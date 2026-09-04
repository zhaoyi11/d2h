"""Continuous signed world-Z knob targets."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms


def yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


class KnobTargetCommand(CommandTerm):
    """Continuously sample signed yaw deltas and expose their targets in the LEAP base frame."""

    cfg: "KnobTargetCommandCfg"

    def __init__(self, cfg: "KnobTargetCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.target_angle = torch.zeros(self.num_envs, device=self.device)
        self.metrics["angle_error"] = torch.zeros_like(self.target_angle)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        magnitude = torch.empty(env_ids.numel(), device=self.device).uniform_(*self.cfg.magnitude_range)
        sign = torch.randint(0, 2, (env_ids.numel(),), device=self.device) * 2 - 1
        current_angle = yaw_from_quat(self.object.data.root_quat_w[env_ids])
        self.target_angle[env_ids] = wrap_to_pi(current_angle + magnitude * sign)

        goal_pos_w = self.object.data.root_pos_w[env_ids]
        zero = torch.zeros_like(magnitude)
        goal_quat_w = quat_from_euler_xyz(zero, zero, self.target_angle[env_ids])
        goal_pos_b, goal_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            goal_pos_w,
            goal_quat_w,
        )
        self.pose_command_b[env_ids] = torch.cat((goal_pos_b, goal_quat_b), dim=-1)
        self.pose_command_w[env_ids] = torch.cat((goal_pos_w, goal_quat_w), dim=-1)

    def _update_metrics(self):
        angle = yaw_from_quat(self.object.data.root_quat_w)
        self.metrics["angle_error"] = wrap_to_pi(self.target_angle - angle).abs()

    def _update_command(self):
        achieved = self.metrics["angle_error"] <= self.cfg.angle_threshold
        env_ids = achieved.nonzero().flatten()
        if env_ids.numel() > 0:
            self._resample_command(env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
                self.current_visualizer = VisualizationMarkers(self.cfg.current_visualizer_cfg)
            self.goal_visualizer.set_visibility(True)
            self.current_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)
            self.current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        marker_pos_w = self.object.data.root_pos_w
        self.goal_visualizer.visualize(marker_pos_w, self.pose_command_w[:, 3:])
        self.current_visualizer.visualize(marker_pos_w, self.object.data.root_quat_w)


@configclass
class KnobTargetCommandCfg(CommandTermCfg):
    class_type: type = KnobTargetCommand
    asset_name: str = MISSING
    object_name: str = MISSING
    magnitude_range: tuple[float, float] = (math.pi / 3.0, math.pi / 2.0)
    angle_threshold: float = 0.1
    goal_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/knob_goal"
    )
    goal_visualizer_cfg.markers["arrow"].scale = (0.06, 0.06, 0.18)
    current_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/knob_current"
    )
    current_visualizer_cfg.markers["arrow"].scale = (0.06, 0.06, 0.18)


__all__ = ["KnobTargetCommand", "KnobTargetCommandCfg", "wrap_to_pi", "yaw_from_quat"]
