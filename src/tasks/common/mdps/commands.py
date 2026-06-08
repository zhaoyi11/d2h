# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from dataclasses import MISSING
import importlib.util
from pathlib import Path
import sys
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_from_euler_xyz,
    quat_unique,
    subtract_frame_transforms,
)
import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


DEFAULT_HAND_BASE_TO_ANCHOR_POSE = (0.10623648, 0.01035594, 0.07579897, 1.0, 0.0, 0.0, 0.0)
DEFAULT_TARGET_ANCHOR_QUAT = (0.70710678, 0.0, 0.70710678, 0.0)
DEFAULT_ANCHOR_CLEARANCE = 0.02
DEFAULT_PICK_INSERT_RECEPTIVE_POSE = (0.35, 0.0, 0.285, 1.0, 0.0, 0.0, 0.0)
DEFAULT_PICK_INSERT_SEGMENT_STEPS = (20, 20, 10, 20, 5)


def _load_pick_insert_object_trajectory_module():
    module_name = "pick_insert_demo_object_trajectory_for_command"
    if module_name in sys.modules:
        return sys.modules[module_name]

    trajectory_path = Path(__file__).resolve().parents[2] / "pick_insert demo" / "object_trajectory.py"
    spec = importlib.util.spec_from_file_location(module_name, trajectory_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pick-insert object trajectory helper from {trajectory_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def hand_base_pose_from_object_command_b(
    object_pose_b: torch.Tensor,
    hand_base_to_anchor_pose: torch.Tensor,
    target_anchor_quat: torch.Tensor | tuple[float, float, float, float] = DEFAULT_TARGET_ANCHOR_QUAT,
    anchor_clearance: float = DEFAULT_ANCHOR_CLEARANCE,
) -> torch.Tensor:
    """Compute target hand-base pose from target object pose and a fixed anchor offset."""
    if hand_base_to_anchor_pose.ndim == 1:
        hand_base_to_anchor_pose = hand_base_to_anchor_pose.unsqueeze(0).repeat(object_pose_b.shape[0], 1)
    hand_base_to_anchor_pose = hand_base_to_anchor_pose.to(dtype=object_pose_b.dtype, device=object_pose_b.device)
    if not isinstance(target_anchor_quat, torch.Tensor):
        target_anchor_quat = torch.tensor(target_anchor_quat, dtype=object_pose_b.dtype, device=object_pose_b.device)
    if target_anchor_quat.ndim == 1:
        target_anchor_quat = target_anchor_quat.unsqueeze(0).repeat(object_pose_b.shape[0], 1)
    target_anchor_quat = target_anchor_quat.to(dtype=object_pose_b.dtype, device=object_pose_b.device)

    target_anchor_pos_b = object_pose_b[:, :3].clone()
    target_anchor_pos_b[:, 2] += object_pose_b.new_tensor(anchor_clearance)
    anchor_to_hand_base_pos, anchor_to_hand_base_quat = subtract_frame_transforms(
        hand_base_to_anchor_pose[:, :3],
        hand_base_to_anchor_pose[:, 3:7],
    )
    hand_base_pos_b, hand_base_quat_b = combine_frame_transforms(
        target_anchor_pos_b,
        target_anchor_quat,
        anchor_to_hand_base_pos,
        anchor_to_hand_base_quat,
    )
    return torch.cat((hand_base_pos_b, hand_base_quat_b), dim=1)


class ObjectUniformPoseCommand(CommandTerm):
    """Uniform pose command generator for an object (in the robot base frame).

    This command term samples target object poses by:
      • Drawing (x, y, z) uniformly within configured Cartesian bounds, and
      • Drawing roll-pitch-yaw uniformly within configured ranges, then converting
        to a quaternion (w, x, y, z). Optionally makes quaternions unique by enforcing
        a positive real part.

    Frames:
        Targets are defined in the robot's *base frame*. For metrics/visualization,
        targets are transformed into the *world frame* using the robot root pose.

    Outputs:
        The command buffer has shape (num_envs, 7): `(x, y, z, qw, qx, qy, qz)`.

    Metrics:
        `position_error` and `orientation_error` are computed between the commanded
        world-frame pose and the object's current world-frame pose.

    Config:
        `cfg` must provide the sampling ranges, whether to enforce quaternion uniqueness,
        and optional visualization settings.
    """

    cfg: ObjectUniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: ObjectUniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.success_vis_asset: RigidObject = env.scene[cfg.success_vis_asset_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
        self.success_visualizer.set_visibility(True)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3],
            self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        success_id = self.metrics["position_error"] < 0.05
        if not self.cfg.position_only:
            success_id &= self.metrics["orientation_error"] < 0.5
        self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success_id.int())

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                # -- goal pose
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_visualizer.set_visibility(True)
            self.curr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                self.goal_visualizer.set_visibility(False)
                self.curr_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        if not self.cfg.position_only:
            # -- goal pose
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
            # -- current object pose
            self.curr_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        else:
            distance = torch.norm(self.pose_command_w[:, :3] - self.object.data.root_pos_w[:, :3], dim=1)
            success_id = (distance < 0.05).int()
            # note: since marker indices for position is 1(far) and 2(near), we can simply shift the success_id by 1.
            # -- goal position
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)
            # -- current object position
            self.curr_visualizer.visualize(self.object.data.root_pos_w, marker_indices=success_id + 1)


class ObjectAndHandBasePoseCommand(ObjectUniformPoseCommand):
    """Object pose command augmented with a target robot hand-base pose."""

    cfg: ObjectAndHandBasePoseCommandCfg

    def __init__(self, cfg: ObjectAndHandBasePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.hand_base_pose_command_b = torch.zeros_like(self.pose_command_b)
        self.hand_base_pose_command_b[:, 3] = 1.0
        self._command_b = torch.zeros(self.num_envs, 14, device=self.device)
        self._command_b[:, 3] = 1.0
        self._command_b[:, 10] = 1.0
        self._hand_base_to_anchor_pose = torch.tensor(cfg.hand_base_to_anchor_pose, device=self.device).repeat(
            self.num_envs, 1
        )
        self._update_hand_base_pose_command()

    @property
    def command(self) -> torch.Tensor:
        self._command_b[:, :7] = self.pose_command_b
        self._command_b[:, 7:14] = self.hand_base_pose_command_b
        return self._command_b

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self._update_hand_base_pose_command(env_ids)

    def _update_command(self):
        self._update_hand_base_pose_command()

    def _update_hand_base_pose_command(self, env_ids: Sequence[int] | slice = slice(None)) -> None:
        self.hand_base_pose_command_b[env_ids] = hand_base_pose_from_object_command_b(
            self.pose_command_b[env_ids],
            self._hand_base_to_anchor_pose[env_ids],
            target_anchor_quat=self.cfg.target_anchor_quat,
            anchor_clearance=self.cfg.anchor_clearance,
        )


class PickInsertTrajectoryObjectAndHandBasePoseCommand(ObjectAndHandBasePoseCommand):
    """Object and hand-base command that follows the pick-insert demo trajectory."""

    cfg: PickInsertTrajectoryObjectAndHandBasePoseCommandCfg

    def __init__(self, cfg: PickInsertTrajectoryObjectAndHandBasePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        body_ids, body_names = self.robot.find_bodies(self.cfg.hand_base_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the hand base body name: {self.cfg.hand_base_body_name}. "
                f"Found {len(body_ids)}: {body_names}."
            )
        self._hand_base_body_idx = body_ids[0]
        self._trajectory_length = 1 + sum(self.cfg.trajectory_segment_steps)
        self._trajectory_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._object_pose_trajectory_b = torch.zeros(self.num_envs, self._trajectory_length, 7, device=self.device)
        self._object_pose_trajectory_b[:, :, 3] = 1.0
        self._trajectory_command_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.metrics["hand_base_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_base_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["trajectory_command_achieved"] = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float):
        self._update_metrics()
        self._update_command()

    def _update_metrics(self):
        super()._update_metrics()
        hand_base_pos_b, hand_base_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.body_pos_w[:, self._hand_base_body_idx],
            self.robot.data.body_quat_w[:, self._hand_base_body_idx],
        )
        hand_base_pos_error, hand_base_rot_error = compute_pose_error(
            self.hand_base_pose_command_b[:, :3],
            self.hand_base_pose_command_b[:, 3:7],
            hand_base_pos_b,
            hand_base_quat_b,
        )
        self.metrics["hand_base_position_error"] = torch.norm(hand_base_pos_error, dim=-1)
        self.metrics["hand_base_orientation_error"] = torch.norm(hand_base_rot_error, dim=-1)

        object_achieved = self.metrics["position_error"] < self.cfg.object_position_tolerance
        if not self.cfg.position_only:
            object_achieved &= self.metrics["orientation_error"] < self.cfg.object_orientation_tolerance
        hand_base_achieved = self.metrics["hand_base_position_error"] < self.cfg.hand_base_position_tolerance
        hand_base_achieved &= self.metrics["hand_base_orientation_error"] < self.cfg.hand_base_orientation_tolerance
        self._trajectory_command_achieved[:] = object_achieved & hand_base_achieved
        self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = self._env_ids_tensor(env_ids)
        if env_ids_tensor.numel() == 0:
            return

        current_pos_b, current_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids_tensor],
            self.robot.data.root_quat_w[env_ids_tensor],
            self.object.data.root_pos_w[env_ids_tensor],
            self.object.data.root_quat_w[env_ids_tensor],
        )
        current_pose_b = torch.cat((current_pos_b, current_quat_b), dim=1)
        trajectory_module = _load_pick_insert_object_trajectory_module()
        receptive_pose = torch.tensor(self.cfg.receptive_pose, dtype=current_pose_b.dtype, device=self.device)

        trajectories = [
            trajectory_module.build_pick_insert_object_pose_sequence(
                current_pose_b[env_idx],
                receptive_pose=receptive_pose,
                segment_steps=self.cfg.trajectory_segment_steps,
                above_offset=self.cfg.above_offset,
                insertion_depth=self.cfg.insertion_depth,
                approach_height=self.cfg.approach_height,
            )
            for env_idx in range(env_ids_tensor.numel())
        ]
        self._object_pose_trajectory_b[env_ids_tensor] = torch.stack(trajectories, dim=0)
        self._trajectory_step[env_ids_tensor] = 0
        self.pose_command_b[env_ids_tensor] = self._object_pose_trajectory_b[env_ids_tensor, 0]
        self._update_hand_base_pose_command(env_ids_tensor)
        self._trajectory_command_achieved[env_ids_tensor] = False

    def _update_command(self):
        advance_env_ids = self._trajectory_command_achieved.nonzero().flatten()
        if advance_env_ids.numel() == 0:
            self._update_hand_base_pose_command()
            return
        self._trajectory_step[advance_env_ids] = torch.clamp(
            self._trajectory_step[advance_env_ids] + 1,
            max=self._trajectory_length - 1,
        )
        self.pose_command_b[advance_env_ids] = self._object_pose_trajectory_b[
            advance_env_ids,
            self._trajectory_step[advance_env_ids],
        ]
        self._update_hand_base_pose_command()

    def _env_ids_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)


#############
# ConfigClass
#############

ALIGN_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.1, 0.1, 0.1),
        ),
        "position_far": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        "position_near": sim_utils.SphereCfg(
            radius=0.01,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
        ),
    }
)


@configclass
class ObjectUniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = ObjectUniformPoseCommand

    asset_name: str = MISSING
    """Name of the coordinate referencing asset in the environment for which the commands are generated respect to."""

    object_name: str = MISSING
    """Name of the object in the environment for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    position_only: bool = True
    """Command goal position only. Command includes goal quat if False"""

    # Pose Markers
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    curr_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    success_vis_asset_name: str = MISSING
    """Name of the asset in the environment for which the success color are indicated."""

    # success markers
    success_visualizer_cfg = VisualizationMarkersCfg(prim_path="/Visuals/SuccessMarkers", markers={})
    """The configuration for the success visualization marker. User needs to add the markers"""


@configclass
class ObjectAndHandBasePoseCommandCfg(ObjectUniformPoseCommandCfg):
    """Configuration for object and target hand-base pose commands."""

    class_type: type = ObjectAndHandBasePoseCommand

    hand_base_to_anchor_pose: tuple[float, float, float, float, float, float, float] = DEFAULT_HAND_BASE_TO_ANCHOR_POSE
    """Anchor pose w.r.t. the robot hand base as ``(x, y, z, qw, qx, qy, qz)``."""

    anchor_clearance: float = DEFAULT_ANCHOR_CLEARANCE
    """Target anchor height above the target object pose in the command frame."""

    target_anchor_quat: tuple[float, float, float, float] = DEFAULT_TARGET_ANCHOR_QUAT
    """Target anchor-frame orientation as ``(qw, qx, qy, qz)`` in the command frame."""


@configclass
class PickInsertTrajectoryObjectAndHandBasePoseCommandCfg(ObjectAndHandBasePoseCommandCfg):
    """Configuration for the pick-insert trajectory command."""

    class_type: type = PickInsertTrajectoryObjectAndHandBasePoseCommand

    receptive_pose: tuple[float, float, float, float, float, float, float] = DEFAULT_PICK_INSERT_RECEPTIVE_POSE
    """Target receptacle pose used by the pick-insert object trajectory."""

    hand_base_body_name: str = "base"
    """Robot body whose pose must reach the target hand-base command before advancing."""

    trajectory_segment_steps: tuple[int, int, int, int, int] = DEFAULT_PICK_INSERT_SEGMENT_STEPS
    """Interpolation samples for move, align, approach, insert, and hold segments."""

    above_offset: float = 0.15
    """Height above the receptacle for the initial move and orientation alignment."""

    insertion_depth: float = 0.015
    """Inserted object height offset above the receptacle pose."""

    approach_height: float = 0.08
    """Height above the insertion pose used before final descent."""

    object_position_tolerance: float = 0.02
    """Object position tolerance in meters for advancing the command trajectory."""

    object_orientation_tolerance: float = 0.2
    """Object orientation tolerance in radians for advancing the command trajectory."""

    hand_base_position_tolerance: float = 0.02
    """Hand-base position tolerance in meters for advancing the command trajectory."""

    hand_base_orientation_tolerance: float = 0.2
    """Hand-base orientation tolerance in radians for advancing the command trajectory."""
