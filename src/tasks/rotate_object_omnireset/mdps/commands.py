"""Fixed object-pose command paired with recorded OmniReset states."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_mul,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class RecordedObjectPoseCommand(CommandTerm):
    """Hold the recorded goal injected by the reset event for the full episode."""

    cfg: RecordedObjectPoseCommandCfg

    def __init__(self, cfg: RecordedObjectPoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.hand_base_command_b = self.pose_command_b.clone()
        self._command_b = torch.cat((self.pose_command_b, self.hand_base_command_b), dim=1)
        self._pending_goal_b = torch.zeros_like(self.pose_command_b)
        self._pending_hand_base_command_b = torch.zeros_like(self.pose_command_b)
        self._has_pending_goal = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self._marker_z_to_y_quat = self.pose_command_b.new_tensor(
            (0.70710678, -0.70710678, 0.0, 0.0)
        ).repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        if not self.cfg.include_hand_base_command:
            return self.pose_command_b
        goal_pos_h, goal_quat_h = subtract_frame_transforms(
            self.hand_base_command_b[:, :3],
            self.hand_base_command_b[:, 3:7],
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:7],
        )
        self._command_b[:, :7] = torch.cat((goal_pos_h, goal_quat_h), dim=1)
        self._command_b[:, 7:14] = self.hand_base_command_b
        return self._command_b

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def set_pending_goals(
        self,
        env_ids: torch.Tensor,
        goal_pose_b: torch.Tensor,
        hand_base_command_b: torch.Tensor | None = None,
    ) -> None:
        """Stage paired goals for consumption by the following command-manager reset."""
        env_ids = self._env_ids_tensor(env_ids)
        expected_shape = (env_ids.numel(), 7)
        if goal_pose_b.shape != expected_shape:
            raise ValueError(
                f"Recorded goals must have shape {expected_shape}, got {goal_pose_b.shape}."
            )
        if not bool(torch.isfinite(goal_pose_b).all()):
            raise ValueError("Recorded goals must contain only finite values.")
        if self.cfg.include_hand_base_command and hand_base_command_b is None:
            raise ValueError("BC commands require a paired hand-base command.")
        if hand_base_command_b is not None:
            if hand_base_command_b.shape != expected_shape:
                raise ValueError(
                    f"Recorded hand-base commands must have shape {expected_shape}, "
                    f"got {hand_base_command_b.shape}."
                )
            if not bool(torch.isfinite(hand_base_command_b).all()):
                raise ValueError("Recorded hand-base commands must contain only finite values.")
            self._pending_hand_base_command_b[env_ids] = hand_base_command_b
        self._pending_goal_b[env_ids] = goal_pose_b
        self._has_pending_goal[env_ids] = True

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = self._env_ids_tensor(env_ids)
        missing = env_ids[~self._has_pending_goal[env_ids]]
        if missing.numel() > 0:
            raise RuntimeError(
                f"Reset event did not provide recorded goals for environments {missing.tolist()}."
            )
        self.pose_command_b[env_ids] = self._pending_goal_b[env_ids]
        self.hand_base_command_b[env_ids] = self._pending_hand_base_command_b[env_ids]
        self._has_pending_goal[env_ids] = False

    def _update_metrics(self) -> None:
        desired_pos_w, desired_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:7],
        )
        position_error, orientation_error = compute_pose_error(
            desired_pos_w,
            desired_quat_w,
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
        )
        self.metrics["position_error"] = torch.linalg.vector_norm(position_error, dim=1)
        self.metrics["orientation_error"] = torch.linalg.vector_norm(
            orientation_error, dim=1
        )

    def _update_command(self) -> None:
        pass

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis:
            if not hasattr(self, "goal_axis_visualizer"):
                self.goal_axis_visualizer = VisualizationMarkers(
                    self.cfg.goal_axis_visualizer_cfg
                )
                self.current_axis_visualizer = VisualizationMarkers(
                    self.cfg.current_axis_visualizer_cfg
                )
            self.goal_axis_visualizer.set_visibility(True)
            self.current_axis_visualizer.set_visibility(True)
        elif hasattr(self, "goal_axis_visualizer"):
            self.goal_axis_visualizer.set_visibility(False)
            self.current_axis_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event) -> None:
        if not self.robot.is_initialized:
            return
        _, goal_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:7],
        )
        marker_pos_w = self.object.data.root_pos_w
        self.goal_axis_visualizer.visualize(
            marker_pos_w, quat_mul(goal_quat_w, self._marker_z_to_y_quat)
        )
        self.current_axis_visualizer.visualize(
            marker_pos_w, quat_mul(self.object.data.root_quat_w, self._marker_z_to_y_quat)
        )


@configclass
class RecordedObjectPoseCommandCfg(CommandTermCfg):
    class_type: type = RecordedObjectPoseCommand
    asset_name: str = MISSING
    object_name: str = MISSING
    include_hand_base_command: bool = False
    """Expose the 14-D object-in-hand and hand-base command used by BC."""
    goal_axis_visualizer_cfg: VisualizationMarkersCfg = (
        GREEN_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/rotate_object_goal_y_axis"
        )
    )
    goal_axis_visualizer_cfg.markers["arrow"].scale = (0.02, 0.02, 0.2)
    current_axis_visualizer_cfg: VisualizationMarkersCfg = (
        BLUE_ARROW_X_MARKER_CFG.replace(
            prim_path="/Visuals/Command/rotate_object_current_y_axis"
        )
    )
    current_axis_visualizer_cfg.markers["arrow"].scale = (0.025, 0.025, 0.18)


__all__ = ["RecordedObjectPoseCommand", "RecordedObjectPoseCommandCfg"]
