# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

from dataclasses import MISSING
import importlib.util
import math
from pathlib import Path
import sys
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    apply_delta_pose,
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
# Anchor pose offset in the robot root frame as (x, y, z, qw, qx, qy, qz).
# Position (0, 0, -0.01) is a fixed -1 cm z-offset in root; orientation (√2/2, 0, √2/2, 0) is 90° rotation about root +Y axis.
DEFAULT_OBJECT_TO_ANCHOR_POSE = (0.0, 0.0, -0.01, 0.70710678, 0.0, 0.70710678, 0.0)
DEFAULT_PICK_INSERT_RECEPTIVE_POSE = (0.35, 0.0, 0.27, 1.0, 0.0, 0.0, 0.0)
# DEFAULT_PICK_INSERT_SEGMENT_STEPS = (20, 20, 10, 50, 20)
DEFAULT_PICK_INSERT_SEGMENT_STEPS = (2, 1, 1, 1, 1)


class StageObjTol(NamedTuple):
    """Per-stage object tolerances for advancing the pick-insert trajectory."""

    object_position: float
    object_orientation: float


DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES = (
    StageObjTol(0.02, 0.3),  # move
    StageObjTol(0.02, 0.2),  # align
    StageObjTol(0.01, 0.2),  # approach
    StageObjTol(0.005, 0.1),  # insert
    StageObjTol(0.005, 0.1),  # hold
)


def _stage_obj_tor_tensors(
    stage_obj_tors: Sequence[StageObjTol],
    expected_count: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(stage_obj_tors) != expected_count:
        raise ValueError(
            "stage_object_tolerances must have one entry per trajectory segment "
            f"({expected_count}); got {len(stage_obj_tors)}."
        )

    values = []
    for stage_idx, stage_obj_tor in enumerate(stage_obj_tors):
        if not isinstance(stage_obj_tor, StageObjTol):
            raise TypeError(
                f"stage_object_tolerances[{stage_idx}] must be a StageObjTol; "
                f"got {type(stage_obj_tor).__name__}."
            )
        position = float(stage_obj_tor.object_position)
        orientation = float(stage_obj_tor.object_orientation)
        if not math.isfinite(position) or not math.isfinite(orientation):
            raise ValueError(f"stage_object_tolerances[{stage_idx}] values must be finite.")
        if position < 0.0 or orientation < 0.0:
            raise ValueError(f"stage_object_tolerances[{stage_idx}] values must be non-negative.")
        values.append((position, orientation))

    stage_obj_tor_tensor = torch.tensor(values, device=device)
    return stage_obj_tor_tensor[:, 0], stage_obj_tor_tensor[:, 1]


def _load_object_trajectory_module():
    """Load the peg-insertion trajectory helper (sibling module) by file path.

    Loaded lazily by path rather than imported at module top level so that unit
    tests can exec ``commands.py`` in isolation without importing the whole
    ``src.tasks.common.mdps`` package (and thus isaaclab).
    """
    module_name = "common_mdps_object_trajectory_for_command"
    if module_name in sys.modules:
        return sys.modules[module_name]

    trajectory_path = Path(__file__).resolve().parent / "object_trajectory.py"
    spec = importlib.util.spec_from_file_location(module_name, trajectory_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load object trajectory helper from {trajectory_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def hand_base_pose_from_object_command_b(
    object_pose_b: torch.Tensor,
    hand_base_to_anchor_pose: torch.Tensor,
    object_to_anchor_pose: torch.Tensor | Sequence[float] = DEFAULT_OBJECT_TO_ANCHOR_POSE,
    anchor_correction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target hand-base pose from target object pose and the object->anchor offset.

    The anchor pose is computed in the *robot root frame*: position = ``object_goal_pos +
    offset_pos`` (offset_pos is a fixed root-frame offset), and orientation = ``offset_quat``
    (fixed root-aligned grasp orientation, decoupled from the object goal). The anchor orientation
    is thus constant along the trajectory while the object goal reorients. The hand base is then
    recovered by composing the inverse of the fixed hand-base->anchor transform. ``object_to_anchor_pose``
    may be a single ``(7,)`` offset (broadcast to all envs) or a per-env ``(N, 7)`` offset.
    ``anchor_correction`` is an optional per-env ``(N, 6)`` bounded delta (pos[3], axis-angle[3])
    applied to the nominal anchor pose *in the anchor frame*.

    Returns:
        Tuple of (hand_base_pose, anchor_pose), each shape (N, 7) in robot root frame.
    """
    if hand_base_to_anchor_pose.ndim == 1:
        hand_base_to_anchor_pose = hand_base_to_anchor_pose.unsqueeze(0).repeat(object_pose_b.shape[0], 1)
    hand_base_to_anchor_pose = hand_base_to_anchor_pose.to(dtype=object_pose_b.dtype, device=object_pose_b.device)
    offset = torch.as_tensor(object_to_anchor_pose, dtype=object_pose_b.dtype, device=object_pose_b.device)
    if offset.ndim == 1:
        offset = offset.unsqueeze(0).repeat(object_pose_b.shape[0], 1)

    # Nominal anchor: both position and orientation are expressed in the robot root frame.
    # Position = object_goal_pos + offset_pos (offset_pos is a fixed root-frame z-offset).
    # Orientation = offset_quat (root-aligned, fixed grasp orientation).
    target_anchor_pos_b = object_pose_b[:, :3] + offset[:, :3]
    target_anchor_quat = offset[:, 3:7]

    # Bounded correction applied *in the anchor frame*. The anchor orientation is root-aligned, so
    # apply_delta_pose (position added directly, orientation left-multiplied) realises an
    # anchor-frame == root-frame delta on the anchor pose.
    if anchor_correction is not None:
        target_anchor_pos_b, target_anchor_quat = apply_delta_pose(
            target_anchor_pos_b, target_anchor_quat, anchor_correction
        )

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
    hand_base_pose = torch.cat((hand_base_pos_b, hand_base_quat_b), dim=1)
    anchor_pose = torch.cat((target_anchor_pos_b, target_anchor_quat), dim=1)
    return hand_base_pose, anchor_pose


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
    """Object pose command augmented with a target robot hand-base pose.

    The internal object target is stored in the robot root frame. The public
    command exposes the object target in the target hand-base frame followed by
    the target hand-base pose in the robot root frame.
    """

    cfg: ObjectAndHandBasePoseCommandCfg

    def __init__(self, cfg: ObjectAndHandBasePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.hand_base_pose_command_b = torch.zeros_like(self.pose_command_b)
        self.hand_base_pose_command_b[:, 3] = 1.0
        self.anchor_pose_command_b = torch.zeros_like(self.pose_command_b)
        self.anchor_pose_command_b[:, 3] = 1.0
        self._command_b = torch.zeros(self.num_envs, 14, device=self.device)
        self._command_b[:, 3] = 1.0
        self._command_b[:, 10] = 1.0
        self._hand_base_to_anchor_pose = torch.tensor(cfg.hand_base_to_anchor_pose, device=self.device).repeat(
            self.num_envs, 1
        )
        self._update_hand_base_pose_command()

    @property
    def command(self) -> torch.Tensor:
        self._command_b[:, :7] = self._object_pose_command_hand_base_b()
        self._command_b[:, 7:14] = self.hand_base_pose_command_b
        return self._command_b

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self._update_hand_base_pose_command(env_ids)

    def _update_command(self):
        self._update_hand_base_pose_command()

    def _update_hand_base_pose_command(self, env_ids: Sequence[int] | slice = slice(None)) -> None:
        hand_base_pose, anchor_pose = hand_base_pose_from_object_command_b(
            self.pose_command_b[env_ids],
            self._hand_base_to_anchor_pose[env_ids],
            self._effective_object_to_anchor_pose(env_ids),
            self._anchor_correction(env_ids),
        )
        self.hand_base_pose_command_b[env_ids] = hand_base_pose
        self.anchor_pose_command_b[env_ids] = anchor_pose

    def _effective_object_to_anchor_pose(self, env_ids: Sequence[int] | slice = slice(None)) -> torch.Tensor:
        """Object->anchor offset used to build the nominal anchor target.

        Position is expressed in the object-goal frame, orientation directly in the robot root
        frame. Returns the static configured offset; any adaptive adjustment is applied as an
        anchor-frame correction via :meth:`_anchor_correction`, not by changing this offset.
        """
        return self.pose_command_b.new_tensor(self.cfg.object_to_anchor_pose)

    def _anchor_correction(self, env_ids: Sequence[int] | slice = slice(None)) -> torch.Tensor | None:
        """Bounded anchor-frame correction (pos[3], axis-angle[3]). ``None`` => no correction."""
        return None

    def _object_pose_command_hand_base_b(self) -> torch.Tensor:
        object_pos_h, object_quat_h = subtract_frame_transforms(
            self.hand_base_pose_command_b[:, :3],
            self.hand_base_pose_command_b[:, 3:7],
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:7],
        )
        return torch.cat((object_pos_h, object_quat_h), dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set visualization for anchor and hand-base command frames."""
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis:
            if not hasattr(self, "anchor_visualizer"):
                self.anchor_visualizer = VisualizationMarkers(self.cfg.anchor_pose_visualizer_cfg)
                self.hand_base_visualizer = VisualizationMarkers(self.cfg.hand_base_pose_visualizer_cfg)
            self.anchor_visualizer.set_visibility(True)
            self.hand_base_visualizer.set_visibility(True)
        else:
            if hasattr(self, "anchor_visualizer"):
                self.anchor_visualizer.set_visibility(False)
                self.hand_base_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualize anchor and hand-base command frames (in addition to object goal/current)."""
        super()._debug_vis_callback(event)
        if not self.robot.is_initialized:
            return
        # Convert root-frame command poses to world frame
        anchor_pos_w, anchor_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.anchor_pose_command_b[:, :3],
            self.anchor_pose_command_b[:, 3:7],
        )
        hand_base_pos_w, hand_base_quat_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.hand_base_pose_command_b[:, :3],
            self.hand_base_pose_command_b[:, 3:7],
        )
        self.anchor_visualizer.visualize(anchor_pos_w, anchor_quat_w)
        self.hand_base_visualizer.visualize(hand_base_pos_w, hand_base_quat_w)


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
        stage_ids = [0]
        for stage_idx, steps in enumerate(self.cfg.trajectory_segment_steps):
            stage_ids.extend([stage_idx] * steps)
        self._step_to_stage = torch.tensor(stage_ids, dtype=torch.long, device=self.device)

        (
            self._stage_object_position_tolerance,
            self._stage_object_orientation_tolerance,
        ) = _stage_obj_tor_tensors(
            self.cfg.stage_object_tolerances,
            len(self.cfg.trajectory_segment_steps),
            self.device,
        )
        self._trajectory_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._object_pose_trajectory_b = torch.zeros(self.num_envs, self._trajectory_length, 7, device=self.device)
        self._object_pose_trajectory_b[:, :, 3] = 1.0
        self._trajectory_command_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Adaptive anchor-frame correction (pos[3], axis-angle[3], root-aligned anchor frame).
        # A bounded PI(D) controller on the measured object->goal error writes these:
        #   _objanchor_integral   -- the integral accumulator (anti-windup clamped)
        #   _objanchor_correction -- the slew-limited applied output (added to the nominal anchor)
        #   _objanchor_prev_err / _objanchor_deriv -- for the optional derivative term
        # All zero => sit at the nominal anchor (object goal + configured offset).
        self._objanchor_integral = torch.zeros(self.num_envs, 6, device=self.device)
        self._objanchor_correction = torch.zeros(self.num_envs, 6, device=self.device)
        self._objanchor_prev_err = torch.zeros(self.num_envs, 6, device=self.device)
        self._objanchor_deriv = torch.zeros(self.num_envs, 6, device=self.device)
        # Stall detection: track object error history and stall counter for correction gating.
        self._objanchor_err_history_pos = torch.zeros(self.num_envs, device=self.device)
        self._objanchor_err_history_rot = torch.zeros(self.num_envs, device=self.device)
        self._objanchor_stall_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["hand_base_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_base_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["trajectory_command_achieved"] = torch.zeros(self.num_envs, device=self.device)

    def compute(self, dt: float):
        self._update_metrics()
        self._update_command()

    def _update_metrics(self):
        super()._update_metrics()
        hand_base_pos_b, hand_base_quat_b = self._current_hand_base_pose_b()
        hand_base_pos_error, hand_base_rot_error = compute_pose_error(
            self.hand_base_pose_command_b[:, :3],
            self.hand_base_pose_command_b[:, 3:7],
            hand_base_pos_b,
            hand_base_quat_b,
        )
        self.metrics["hand_base_position_error"] = torch.norm(hand_base_pos_error, dim=-1)
        self.metrics["hand_base_orientation_error"] = torch.norm(hand_base_rot_error, dim=-1)

        object_achieved = self._object_target_achieved()
        hand_base_achieved = self.metrics["hand_base_position_error"] < self.cfg.hand_base_position_tolerance
        hand_base_achieved &= self.metrics["hand_base_orientation_error"] < self.cfg.hand_base_orientation_tolerance
        achieved = object_achieved & hand_base_achieved
        
        self._trajectory_command_achieved[:] = achieved
        self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()

    def _object_target_achieved(self) -> torch.Tensor:
        stage = self._step_to_stage[self._trajectory_step]
        achieved = self.metrics["position_error"] < self._stage_object_position_tolerance[stage]
        if not self.cfg.position_only:
            achieved &= self.metrics["orientation_error"] < self._stage_object_orientation_tolerance[stage]
        return achieved

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
        trajectory_module = _load_object_trajectory_module()
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
        if self.cfg.enable_pregrasp_reach:
            # Reach phase: seed the step-0 hand-base with a grasp pose over the object (waypoint 0 is
            # the object's current pose) so the arm first descends to the object, instead of freezing
            # at the current hand pose. An optional standoff hovers above the object first.
            reach_object_pose_b = self.pose_command_b[env_ids_tensor].clone()
            reach_object_pose_b[:, 2] += self.cfg.pregrasp_approach_height
            hand_base_pose, anchor_pose = hand_base_pose_from_object_command_b(
                reach_object_pose_b,
                self._hand_base_to_anchor_pose[env_ids_tensor],
                self.cfg.object_to_anchor_pose,
                None,
            )
            self.hand_base_pose_command_b[env_ids_tensor] = hand_base_pose
            self.anchor_pose_command_b[env_ids_tensor] = anchor_pose
        else:
            hand_base_pos_b, hand_base_quat_b = self._current_hand_base_pose_b(env_ids_tensor)
            self.hand_base_pose_command_b[env_ids_tensor] = torch.cat((hand_base_pos_b, hand_base_quat_b), dim=1)
        self._objanchor_integral[env_ids_tensor] = 0.0
        self._objanchor_correction[env_ids_tensor] = 0.0
        self._objanchor_prev_err[env_ids_tensor] = 0.0
        self._objanchor_deriv[env_ids_tensor] = 0.0
        self._objanchor_err_history_pos[env_ids_tensor] = 0.0
        self._objanchor_err_history_rot[env_ids_tensor] = 0.0
        self._objanchor_stall_counter[env_ids_tensor] = 0
        self._trajectory_command_achieved[env_ids_tensor] = False

    def _update_command(self):
        advance_env_ids = self._trajectory_command_achieved.nonzero().flatten()
        if advance_env_ids.numel() == 0:
            active_env_ids = (self._trajectory_step > 0).nonzero().flatten()
            if active_env_ids.numel() > 0:
                self._apply_objanchor_correction(active_env_ids)
                self._update_hand_base_pose_command(active_env_ids)
            return
        self._trajectory_step[advance_env_ids] = torch.clamp(
            self._trajectory_step[advance_env_ids] + 1,
            max=self._trajectory_length - 1,
        )
        self.pose_command_b[advance_env_ids] = self._object_pose_trajectory_b[
            advance_env_ids,
            self._trajectory_step[advance_env_ids],
        ]
        self._objanchor_stall_counter[advance_env_ids] = 0
        active_env_ids = (self._trajectory_step > 0).nonzero().flatten()
        if active_env_ids.numel() > 0:
            self._apply_objanchor_correction(active_env_ids)
            self._update_hand_base_pose_command(active_env_ids)

    @property
    def objanchor_correction(self) -> torch.Tensor:
        """The applied anchor-frame correction (pos[3], axis-angle[3], root-aligned anchor frame)."""
        return self._objanchor_correction

    def _anchor_correction(self, env_ids: Sequence[int] | slice = slice(None)) -> torch.Tensor | None:
        """Bounded PI(D) correction applied to the nominal anchor pose in the anchor frame.

        During ``super().__init__`` the correction buffer does not exist yet -- the ``getattr``
        guard then returns ``None`` (no correction), so no special-casing of the first update is
        needed.
        """
        correction = getattr(self, "_objanchor_correction", None)
        if correction is None:
            return None
        return correction[env_ids]

    def _apply_objanchor_correction(self, env_ids: torch.Tensor) -> None:
        """Nudge the anchor pose so the *measured* object reaches its goal pose.

        The correction is gated by two conditions:
        1. **Anchor achieved**: the hand-base (arm) has settled within tolerance of the commanded pose.
        2. **Object stalled**: the object error has not improved by at least ``corr_stall_delta``
           over the last ``corr_stall_window`` steps. When ``corr_stall_window=0``, this gate is disabled.

        When both conditions hold, a bounded **PI(D) controller** runs on the measured object->goal
        error, expressed in the root-aligned anchor frame (the frame the correction is applied in,
        so a delta on the anchor maps 1:1 onto the resulting anchor/object motion). The proportional
        term gives responsiveness, the integral removes steady-state error (the object actually reaches
        the goal), an optional filtered derivative damps overshoot. A deadband makes the in-hand policy
        own small errors (the arm only acts when the hand *can't*), an anti-windup + output clamp
        bound the deviation to +-``corr_max_pos`` / ``corr_max_rot`` (the 5 cm / 10 deg budget),
        and an output slew limit makes the correction change gradually.

        The correction feeds :meth:`_anchor_correction` and thus the anchor / hand-base target
        (``command[:, 7:14]`` the arm target and, via :meth:`_object_pose_command_hand_base_b`,
        ``command[:, :7]`` the in-hand target), so arm and hand cooperate. Keep the gains/slew slow
        relative to the hand's response.
        """
        if not self.cfg.enable_object_goal_correction or env_ids.numel() == 0:
            return

        # --- Gate 1: anchor achieved (arm has settled at commanded hand-base pose) ---
        hand_base_pos_b, hand_base_quat_b = self._current_hand_base_pose_b(env_ids)
        pos_err, rot_err = compute_pose_error(
            self.hand_base_pose_command_b[env_ids, :3],
            self.hand_base_pose_command_b[env_ids, 3:7],
            hand_base_pos_b,
            hand_base_quat_b,
        )
        anchor_achieved = (
            torch.norm(pos_err, dim=-1) < self.cfg.corr_anchor_achieved_pos
        ) & (
            torch.norm(rot_err, dim=-1) < self.cfg.corr_anchor_achieved_rot
        )

        # --- Gate 2: object stall detection ---
        # Compute current object error magnitude for envs in env_ids.
        object_pos_b, object_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.object.data.root_pos_w[env_ids],
            self.object.data.root_quat_w[env_ids],
        )
        err_pos_b, err_rot_b = compute_pose_error(
            object_pos_b,
            object_quat_b,
            self.pose_command_b[env_ids, :3],
            self.pose_command_b[env_ids, 3:7],
            rot_error_type="axis_angle",
        )
        cur_err_pos = torch.norm(err_pos_b, dim=-1)
        cur_err_rot = torch.norm(err_rot_b, dim=-1)

        if self.cfg.corr_stall_window > 0:
            # Stall counter: increment if error didn't improve, reset if it did.
            improved = (
                (self._objanchor_err_history_pos[env_ids] - cur_err_pos > self.cfg.corr_stall_delta_pos) |
                (self._objanchor_err_history_rot[env_ids] - cur_err_rot > self.cfg.corr_stall_delta_rot)
            )
            self._objanchor_stall_counter[env_ids] = torch.where(
                improved,
                torch.zeros_like(self._objanchor_stall_counter[env_ids]),
                self._objanchor_stall_counter[env_ids] + 1,
            )
            self._objanchor_err_history_pos[env_ids] = cur_err_pos
            self._objanchor_err_history_rot[env_ids] = cur_err_rot
            object_stalled = self._objanchor_stall_counter[env_ids] >= self.cfg.corr_stall_window
        else:
            object_stalled = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)

        # Only apply correction where both conditions hold.
        gate_mask = anchor_achieved & object_stalled
        active_mask_ids = env_ids[gate_mask]
        if active_mask_ids.numel() == 0:
            return
        env_ids = active_mask_ids
        # The correction is applied in the anchor frame, which is root-aligned (the anchor
        # orientation lives in the robot root frame), so the root-frame error is used directly.
        err_pos = err_pos_b
        err_rot = err_rot_b

        # Per-axis deadband: within tolerance the in-hand policy owns the error; the arm waits.
        err_pos = torch.sign(err_pos) * torch.clamp(err_pos.abs() - self.cfg.corr_deadband_pos, min=0.0)
        err_rot = torch.sign(err_rot) * torch.clamp(err_rot.abs() - self.cfg.corr_deadband_rot, min=0.0)
        err = torch.cat((err_pos, err_rot), dim=-1)

        # Integral term (anti-windup clamped) -- removes steady-state error.
        integ = self._objanchor_integral[env_ids].clone()
        integ[:, :3] = self._clamp_norm(integ[:, :3] + self.cfg.corr_ki_pos * err_pos, self.cfg.corr_max_pos)
        integ[:, 3:] = self._clamp_norm(integ[:, 3:] + self.cfg.corr_ki_rot * err_rot, self.cfg.corr_max_rot)
        self._objanchor_integral[env_ids] = integ

        # Proportional + Integral term.
        raw = torch.zeros_like(integ)
        raw[:, :3] = self.cfg.corr_kp_pos * err_pos
        raw[:, 3:] = self.cfg.corr_kp_rot * err_rot
        raw[:, :3] = raw[:, :3] + integ[:, :3]
        raw[:, 3:] = raw[:, 3:] + integ[:, 3:]

        # Optional filtered-derivative term (off by default; pose-error derivatives are noisy).
        if self.cfg.corr_kd_pos != 0.0 or self.cfg.corr_kd_rot != 0.0:
            d_err = err - self._objanchor_prev_err[env_ids]
            deriv = (1.0 - self.cfg.corr_d_lowpass) * self._objanchor_deriv[env_ids] + self.cfg.corr_d_lowpass * d_err
            self._objanchor_deriv[env_ids] = deriv
            raw[:, :3] = raw[:, :3] + self.cfg.corr_kd_pos * deriv[:, :3]
            raw[:, 3:] = raw[:, 3:] + self.cfg.corr_kd_rot * deriv[:, 3:]
        self._objanchor_prev_err[env_ids] = err

        # Output clamp to the +-5 cm / +-10 deg budget.
        raw[:, :3] = self._clamp_norm(raw[:, :3], self.cfg.corr_max_pos)
        raw[:, 3:] = self._clamp_norm(raw[:, 3:], self.cfg.corr_max_rot)

        # Slew-limit the applied correction toward the target so it changes gradually.
        prev = self._objanchor_correction[env_ids]
        corr = torch.cat(
            (
                self._slew_limit(prev[:, :3], raw[:, :3], self.cfg.corr_slew_pos),
                self._slew_limit(prev[:, 3:], raw[:, 3:], self.cfg.corr_slew_rot),
            ),
            dim=-1,
        )
        self._objanchor_correction[env_ids] = corr

    @staticmethod
    def _clamp_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
        norm = torch.norm(vec, dim=-1, keepdim=True)
        scale = torch.clamp(max_norm / torch.clamp(norm, min=1e-8), max=1.0)
        return vec * scale

    @staticmethod
    def _slew_limit(prev: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
        """Move ``prev`` toward ``target`` by at most ``max_step`` (per-vector norm) this step."""
        delta = target - prev
        norm = torch.norm(delta, dim=-1, keepdim=True)
        scale = torch.clamp(max_step / torch.clamp(norm, min=1e-8), max=1.0)
        return prev + delta * scale

    def _env_ids_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _current_hand_base_pose_b(
        self,
        env_ids: Sequence[int] | slice = slice(None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.robot.data.body_pos_w[env_ids, self._hand_base_body_idx],
            self.robot.data.body_quat_w[env_ids, self._hand_base_body_idx],
        )


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

    object_to_anchor_pose: tuple[float, float, float, float, float, float, float] = DEFAULT_OBJECT_TO_ANCHOR_POSE
    """Anchor pose offset in the **robot root frame** as ``(x, y, z, qw, qx, qy, qz)``.

    Position is a fixed root-frame offset added to the object goal position; orientation is a
    fixed grasp orientation in the root frame, decoupled from the object goal orientation. The
    anchor is thus ``object_goal_pos + offset_pos`` for position and ``offset_quat`` for
    orientation. When the adaptive correction is enabled, it adjusts around this nominal anchor."""

    anchor_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/anchor_pose")
    """The configuration for the anchor pose visualization marker."""

    hand_base_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/hand_base_pose")
    """The configuration for the hand-base pose visualization marker."""


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

    above_offset: float = 0.10
    """Height above the receptacle for the initial move and orientation alignment."""

    insertion_depth: float = 0.06
    """Inserted object height offset above the receptacle pose."""

    approach_height: float = 0.01
    """Height above the insertion pose used before final descent."""

    stage_object_tolerances: tuple[StageObjTol, ...] = DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES
    """Per-stage object position and orientation tolerances for advancing the command trajectory.

    One entry is required for each trajectory segment: move, align, approach, insert, and hold.
    """

    hand_base_position_tolerance: float = 0.02
    """Hand-base position tolerance in meters for advancing the command trajectory."""

    hand_base_orientation_tolerance: float = 0.2
    """Hand-base orientation tolerance in radians for advancing the command trajectory."""

    # -- Initial reach-to-grasp phase (used when the object spawns on the table, not in-hand) --
    enable_pregrasp_reach: bool = False
    """When True, seed the initial (trajectory step 0) hand-base command with a grasp pose over the
    object -- derived from the object's current pose via the object->anchor / hand-base->anchor
    offsets -- instead of freezing at the current hand pose. This makes the arm first reach down to
    the object before the pick->insert trajectory. When False the legacy behavior (hold the current
    hand pose at step 0) is kept."""

    pregrasp_approach_height: float = 0.0
    """Optional standoff (m) added to the object z when deriving the step-0 reach goal, so the hand
    first hovers above the object. Zero => go directly to the grasp pose."""

    # -- Adaptive object->anchor correction (bounded PI(D) so the object reaches its goal) --
    enable_object_goal_correction: bool = False
    """Whether to adapt the object->anchor offset (PI(D) on the measured object->goal error) so
    the object reaches its goal. When False the command exposes the bare initial offset."""

    corr_deadband_pos: float = 0.00
    """Per-axis object position error (m) below which the controller idles -- the in-hand policy
    owns sub-deadband error, so the arm only acts when the hand can't (residual tolerance)."""

    corr_deadband_rot: float = 0.0
    """Per-axis object orientation error (rad) below which the controller idles (residual tol)."""

    '''TODO: !!!! Tune the PI(D) gains and correction limits. The current values are a starting point based on intuition and preliminary experiments, but systematic tuning is needed for best performance.'''
    corr_kp_pos: float = 0.1
    """Proportional gain on the (deadbanded) position error -- responsiveness."""

    corr_kp_rot: float = 0.05
    """Proportional gain on the rotation error (smaller so the hand does most reorientation)."""

    corr_ki_pos: float = 0.1
    """Integral gain on the position error -- removes steady-state error (object reaches goal)."""

    corr_ki_rot: float = 0.05
    """Integral gain on the rotation error (smaller -- rotation commits slowly/reluctantly)."""

    corr_kd_pos: float = 0.1
    """Derivative gain on the position error (off by default; pose-error derivatives are noisy)."""

    corr_kd_rot: float = 0.05
    """Derivative gain on the rotation error (off by default)."""

    corr_d_lowpass: float = 0.2
    """EMA factor for the derivative term (smaller = more smoothing). Used only when Kd != 0."""

    corr_max_pos: float = 0.05
    """Output + integral bound: max position deviation (m) of the offset from its initial value."""

    corr_max_rot: float = 0.1745
    """Output + integral bound: max rotation deviation (rad, ~10 deg) of the offset from initial."""

    corr_slew_pos: float = 0.002
    """Max change (m) of the applied position correction per step -- keeps adjustment gradual."""

    corr_slew_rot: float = 0.005
    """Max change (rad) of the applied rotation correction per step -- keeps adjustment gradual."""

    corr_anchor_achieved_pos: float = 0.01
    """Hand-base position tolerance (m) to consider the anchor achieved (arm settled)."""

    corr_anchor_achieved_rot: float = 0.05
    """Hand-base orientation tolerance (rad) to consider the anchor achieved."""

    corr_stall_window: int = 30
    """Number of steps over which the object error must not improve by corr_stall_delta to
    be considered stalled. Zero disables the stall gate (correction fires whenever anchor
    is achieved)."""

    corr_stall_delta_pos: float = 0.003
    """Object position improvement (m) below which the object is considered stalled."""

    corr_stall_delta_rot: float = 0.01
    """Object orientation improvement (rad) below which the object is considered stalled."""
