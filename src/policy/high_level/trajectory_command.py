# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Scripted-trajectory object/hand-base command term (the high-level HRL adapter).

This is the isaaclab ``CommandTerm`` that drives the pure-``torch`` high-level "brains" in this
package -- the per-env trajectory stepper (:mod:`.trajectory_stepper`), the bounded PI(D) anchor
correction (:mod:`.anchor_correction`) and the grasp-anchor kinematics (:mod:`.utils`).
It lives in :mod:`src.policy.high_level` so the whole HRL reference-generation stack sits together,
but it depends on isaaclab, so -- like :mod:`.gate` and :mod:`.utils` -- it is *not*
re-exported from the package ``__init__`` (importing it eagerly would pull isaaclab into the
otherwise pure-torch package). It is re-exported through ``src.tasks.common.mdps.commands`` so the
``mdp.*`` namespace keeps exposing it to tasks and scripts.
"""

from __future__ import annotations

from dataclasses import MISSING
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    subtract_frame_transforms,
)
import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from src.policy.high_level.anchor_correction import AnchorCorrectionCfg, ObjectAnchorPIDController
from src.policy.high_level.utils import (
    DEFAULT_HAND_BASE_TO_ANCHOR_POSE,
    DEFAULT_OBJECT_TO_ANCHOR_POSE,
    hand_base_pose_from_object_command_b,
)
from src.policy.high_level.trajectory_stepper import (
    StageObjTol,
    TrajectoryStepper,
    stage_tolerance_tensors,
)


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _update_object_pose_metrics(cmd) -> None:
    """Compute the object-goal pose metrics and update the success marker.

    Used by :class:`TrajectoryObjectAndHandBasePoseCommand` here (a duplicate of the same helper in
    ``src.tasks.common.mdps.commands`` that serves ``ObjectUniformPoseCommand``). Transforms the
    root-frame object goal to world, computes the position/orientation error against the measured
    object pose, writes ``metrics["position_error"|"orientation_error"]``, and drives the success
    visualizer. ``cmd`` is duck-typed on ``robot``, ``object``, ``pose_command_b``,
    ``pose_command_w``, ``metrics``, ``cfg.position_only``, ``success_visualizer`` and
    ``success_vis_asset``.
    """
    cmd.pose_command_w[:, :3], cmd.pose_command_w[:, 3:] = combine_frame_transforms(
        cmd.robot.data.root_pos_w,
        cmd.robot.data.root_quat_w,
        cmd.pose_command_b[:, :3],
        cmd.pose_command_b[:, 3:],
    )
    pos_error, rot_error = compute_pose_error(
        cmd.pose_command_w[:, :3],
        cmd.pose_command_w[:, 3:],
        cmd.object.data.root_state_w[:, :3],
        cmd.object.data.root_state_w[:, 3:7],
    )
    cmd.metrics["position_error"] = torch.norm(pos_error, dim=-1)
    cmd.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    success_id = cmd.metrics["position_error"] < 0.05
    if not cmd.cfg.position_only:
        success_id &= cmd.metrics["orientation_error"] < 0.5
    cmd.success_visualizer.visualize(cmd.success_vis_asset.data.root_pos_w, marker_indices=success_id.int())


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


class TrajectoryObjectAndHandBasePoseCommand(CommandTerm):
    """Object and hand-base command that follows a scripted, task-defined object-pose trajectory.

    Inherits directly from :class:`~isaaclab.managers.CommandTerm` -- it is *not* an
    ``ObjectUniformPoseCommand`` (it never samples a uniform goal). It owns the object-goal
    pose buffers/metrics (the metric computation is shared with ``ObjectUniformPoseCommand``
    via the module-level :func:`_update_object_pose_metrics`), the hand-base/anchor kinematics and
    the 14-D public command, plus all the task-agnostic machinery: the per-env stepper
    (:class:`~src.policy.high_level.trajectory_stepper.TrajectoryStepper`), the bounded PI(D) anchor
    correction (:class:`~src.policy.high_level.anchor_correction.ObjectAnchorPIDController`), the
    hand-base targeting/metrics, the optional pregrasp reach, and the advance logic. The only
    task-specific piece is the object-pose trajectory itself, produced by the overridable
    :meth:`_build_object_trajectories` hook. Subclass it per task (see e.g.
    ``src.tasks.pick_insert.mdps.commands.PickInsertTrajectoryObjectAndHandBasePoseCommand``) and
    supply a matching ``trajectory_segment_steps`` / ``stage_object_tolerances`` on the config.
    """

    cfg: TrajectoryObjectAndHandBasePoseCommandCfg

    def __init__(self, cfg: TrajectoryObjectAndHandBasePoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # -- Object-goal pose plumbing (same buffers/handles as ObjectUniformPoseCommand) --
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.success_vis_asset: RigidObject = env.scene[cfg.success_vis_asset_name]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
        self.success_visualizer.set_visibility(True)
        # -- Hand-base / anchor target buffers and the 14-D public command buffer --
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
        # Seed the first hand-base target from the (identity) object goal. ``_anchor_correction``'s
        # getattr guard returns None here because ``_corr`` does not exist yet.
        self._update_hand_base_pose_command()
        # -- Trajectory stepper (per-env waypoints, step index, per-stage advance tolerances) lives
        # in the pure-torch high-level package; the command term orchestrates it.
        body_ids, body_names = self.robot.find_bodies(self.cfg.hand_base_body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the hand base body name: {self.cfg.hand_base_body_name}. "
                f"Found {len(body_ids)}: {body_names}."
            )
        self._hand_base_body_idx = body_ids[0]
        stage_position_tolerance, stage_orientation_tolerance = stage_tolerance_tensors(
            self.cfg.stage_object_tolerances,
            len(self.cfg.trajectory_segment_steps),
            self.device,
        )
        self._stepper = TrajectoryStepper(
            self.num_envs,
            self.device,
            self.cfg.trajectory_segment_steps,
            stage_position_tolerance,
            stage_orientation_tolerance,
        )
        self._trajectory_command_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Adaptive anchor-frame correction: a bounded PI(D) controller (pure-torch high-level module)
        # nudges the anchor pose so the measured object reaches its goal. All-zero output => sit at
        # the nominal anchor (object goal + configured offset). Built after the buffer setup so the
        # ``_anchor_correction`` override's getattr guard returns None during that first update.
        self._corr = ObjectAnchorPIDController(self.num_envs, self.device, self.cfg.correction)
        self.metrics["hand_base_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_base_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["trajectory_command_achieved"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "TrajectoryObjectAndHandBasePoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        self._command_b[:, :7] = self._object_pose_command_hand_base_b()
        self._command_b[:, 7:14] = self.hand_base_pose_command_b
        return self._command_b

    def _update_hand_base_pose_command(self, env_ids: Sequence[int] | slice = slice(None)) -> None:
        hand_base_pose, anchor_pose = hand_base_pose_from_object_command_b(
            self.pose_command_b[env_ids],
            self._hand_base_to_anchor_pose[env_ids],
            self.pose_command_b.new_tensor(self.cfg.object_to_anchor_pose),
            self._anchor_correction(env_ids),
        )
        self.hand_base_pose_command_b[env_ids] = hand_base_pose
        self.anchor_pose_command_b[env_ids] = anchor_pose

    def _object_pose_command_hand_base_b(self) -> torch.Tensor:
        object_pos_h, object_quat_h = subtract_frame_transforms(
            self.hand_base_pose_command_b[:, :3],
            self.hand_base_pose_command_b[:, 3:7],
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:7],
        )
        return torch.cat((object_pos_h, object_quat_h), dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create/toggle the object goal/current and anchor/hand-base command markers."""
        markers = ("goal_visualizer", "curr_visualizer", "anchor_visualizer", "hand_base_visualizer")
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                self.goal_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.curr_visualizer = VisualizationMarkers(self.cfg.curr_pose_visualizer_cfg)
                self.anchor_visualizer = VisualizationMarkers(self.cfg.anchor_pose_visualizer_cfg)
                self.hand_base_visualizer = VisualizationMarkers(self.cfg.hand_base_pose_visualizer_cfg)
            for name in markers:
                getattr(self, name).set_visibility(True)
        else:
            if hasattr(self, "goal_visualizer"):
                for name in markers:
                    getattr(self, name).set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualize the object goal/current and anchor/hand-base command frames."""
        if not self.robot.is_initialized:
            return
        # -- object goal + current object pose (same convention as ObjectUniformPoseCommand)
        if not self.cfg.position_only:
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
            self.curr_visualizer.visualize(self.object.data.root_pos_w, self.object.data.root_quat_w)
        else:
            distance = torch.norm(self.pose_command_w[:, :3] - self.object.data.root_pos_w[:, :3], dim=1)
            success_id = (distance < 0.05).int()
            # marker indices for position are 1 (far) and 2 (near); shift the success_id by 1.
            self.goal_visualizer.visualize(self.pose_command_w[:, :3], marker_indices=success_id + 1)
            self.curr_visualizer.visualize(self.object.data.root_pos_w, marker_indices=success_id + 1)
        # -- anchor + hand-base command frames (root -> world)
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

    def compute(self, dt: float):
        self._update_metrics()
        self._update_command()

    def _update_metrics(self):
        _update_object_pose_metrics(self)
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
        return self._stepper.object_target_achieved(
            self.metrics["position_error"],
            self.metrics["orientation_error"],
            self.cfg.position_only,
        )

    def _build_object_trajectories(
        self, env_ids: torch.Tensor, current_pose_b: torch.Tensor
    ) -> torch.Tensor:
        """Build the per-env object-pose waypoint sequences to follow (task-specific hook).

        Override this in a task subclass to produce the scripted object trajectory for that task.
        Given the current object poses in the robot root frame (``current_pose_b``, shape
        ``(len(env_ids), 7)`` as ``(x, y, z, qw, qx, qy, qz)``, one row per env in ``env_ids``),
        return a stacked tensor of shape ``(len(env_ids), 1 + sum(trajectory_segment_steps), 7)``.
        The waypoint count must match the stepper built from ``cfg.trajectory_segment_steps``; the
        number and meaning of the segments is entirely task-defined. Use
        :func:`~src.policy.high_level.utils.build_object_pose_sequence_from_keyframes` to
        turn task keyframes into waypoints.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_object_trajectories to supply its "
            "task-specific object-pose trajectory."
        )

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

        trajectories = self._build_object_trajectories(env_ids_tensor, current_pose_b)
        self._stepper.reset(env_ids_tensor, trajectories)
        self.pose_command_b[env_ids_tensor] = self._stepper.current_object_pose(env_ids_tensor)
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
        self._corr.reset(env_ids_tensor)
        self._trajectory_command_achieved[env_ids_tensor] = False

    def _update_command(self):
        advance_env_ids = self._trajectory_command_achieved.nonzero().flatten()
        if advance_env_ids.numel() == 0:
            active_env_ids = (self._stepper.step > 0).nonzero().flatten()
            if active_env_ids.numel() > 0:
                self._apply_objanchor_correction(active_env_ids)
                self._update_hand_base_pose_command(active_env_ids)
            return
        self._stepper.advance(advance_env_ids)
        self.pose_command_b[advance_env_ids] = self._stepper.current_object_pose(advance_env_ids)
        self._corr.clear_stall(advance_env_ids)
        active_env_ids = (self._stepper.step > 0).nonzero().flatten()
        if active_env_ids.numel() > 0:
            self._apply_objanchor_correction(active_env_ids)
            self._update_hand_base_pose_command(active_env_ids)

    @property
    def objanchor_correction(self) -> torch.Tensor:
        """The applied anchor-frame correction (pos[3], axis-angle[3], root-aligned anchor frame)."""
        return self._corr.correction

    def _anchor_correction(self, env_ids: Sequence[int] | slice = slice(None)) -> torch.Tensor | None:
        """Bounded PI(D) correction applied to the nominal anchor pose in the anchor frame.

        During ``super().__init__`` the controller does not exist yet -- the ``getattr`` guard then
        returns ``None`` (no correction), so no special-casing of the first update is needed.
        """
        controller = getattr(self, "_corr", None)
        if controller is None:
            return None
        return controller.correction[env_ids]

    def _apply_objanchor_correction(self, env_ids: torch.Tensor) -> None:
        """Nudge the anchor pose so the *measured* object reaches its goal pose.

        Computes the two frame-coupled inputs and delegates the stateful, gated PI(D) update to
        :class:`ObjectAnchorPIDController`:

        1. **Anchor achieved** gate -- the hand-base (arm) has settled within tolerance of its
           commanded pose (needs the robot pose, so it is computed here).
        2. The **measured object->goal error** in the root-aligned anchor frame (the frame the
           correction is applied in, so a delta on the anchor maps 1:1 onto the object motion --
           the root-frame error is used directly).

        The controller owns the object-stall gate and the bounded PI(D) math (deadband, anti-windup
        integral, optional filtered derivative, output clamp, slew). Its output feeds
        :meth:`_anchor_correction` and thus both the arm target (``command[:, 7:14]``) and, via
        :meth:`_object_pose_command_hand_base_b`, the in-hand target (``command[:, :7]``), so arm
        and hand cooperate.
        """
        if not self.cfg.correction.enable or env_ids.numel() == 0:
            return

        # Gate 1: anchor achieved (arm has settled at the commanded hand-base pose).
        hand_base_pos_b, hand_base_quat_b = self._current_hand_base_pose_b(env_ids)
        pos_err, rot_err = compute_pose_error(
            self.hand_base_pose_command_b[env_ids, :3],
            self.hand_base_pose_command_b[env_ids, 3:7],
            hand_base_pos_b,
            hand_base_quat_b,
        )
        anchor_achieved = (
            torch.norm(pos_err, dim=-1) < self.cfg.correction.anchor_achieved_pos
        ) & (
            torch.norm(rot_err, dim=-1) < self.cfg.correction.anchor_achieved_rot
        )

        # Measured object->goal error in the root-aligned anchor frame.
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

        # Object-stall gate + bounded PI(D) update (writes the correction for the gated subset).
        self._corr.update(env_ids, err_pos_b, err_rot_b, anchor_achieved)

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


@configclass
class TrajectoryObjectAndHandBasePoseCommandCfg(CommandTermCfg):
    """Configuration for the generic scripted-trajectory object/hand-base command.

    Inherits directly from :class:`~isaaclab.managers.CommandTermCfg` (mirroring the command, which
    inherits :class:`~isaaclab.managers.CommandTerm` directly). It is *not* a uniform-pose command,
    so it declares only the object-goal fields the shared plumbing needs and omits the
    uniform-sampling-only ``ranges`` / ``make_quat_unique``. A task subclass sets ``class_type`` to
    its command, fills ``trajectory_segment_steps`` / ``stage_object_tolerances`` (one entry per
    segment, task-defined), and adds any task-specific trajectory parameters."""

    class_type: type = TrajectoryObjectAndHandBasePoseCommand

    # -- Object-goal command fields (consumed by the shared object-pose plumbing / metrics) --
    asset_name: str = MISSING
    """Name of the coordinate-referencing asset (robot) the commands are generated w.r.t."""

    object_name: str = MISSING
    """Name of the object the commands are generated for."""

    position_only: bool = True
    """Command goal position only. Command includes goal quat if False."""

    success_vis_asset_name: str = MISSING
    """Name of the asset in the environment for which the success color is indicated."""

    success_visualizer_cfg = VisualizationMarkersCfg(prim_path="/Visuals/SuccessMarkers", markers={})
    """The configuration for the success visualization marker. User needs to add the markers."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker."""

    curr_pose_visualizer_cfg: VisualizationMarkersCfg = ALIGN_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker."""

    # -- Hand-base / anchor grasp kinematics (formerly ObjectAndHandBasePoseCommandCfg) --
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

    hand_base_body_name: str = "base"
    """Robot body whose pose must reach the target hand-base command before advancing."""

    trajectory_segment_steps: tuple[int, ...] = MISSING
    """Interpolation samples per trajectory segment. Length and meaning are task-defined; the task's
    command builds ``1 + sum(trajectory_segment_steps)`` waypoints to match the stepper."""

    stage_object_tolerances: tuple[StageObjTol, ...] = MISSING
    """Per-stage object position and orientation tolerances for advancing the command trajectory.

    One :class:`StageObjTol` entry is required for each trajectory segment (matching
    ``trajectory_segment_steps``)."""

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
    correction: AnchorCorrectionCfg = AnchorCorrectionCfg()
    """Bounded PI(D) correction on the measured object->goal error. Disabled by default
    (``correction.enable = False``); see :class:`AnchorCorrectionCfg` for the gains/limits."""


__all__ = [
    "ALIGN_MARKER_CFG",
    "TrajectoryObjectAndHandBasePoseCommand",
    "TrajectoryObjectAndHandBasePoseCommandCfg",
]
