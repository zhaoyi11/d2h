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
from src.policy.high_level.drop_recovery import DropRecoveryTracker
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
    hand-base targeting/metrics, and the advance logic. The pre-grasp reach is not special-cased
    here -- it is simply the trajectory's first stage (the object goal held at its spawn pose while
    the arm reaches to grasp). The only task-specific piece is the object-pose trajectory itself,
    produced by the overridable
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
        # Drop-recovery state machine (pure-torch high-level module): arms once the object is secured
        # (hand at the object) and, if it later leaves the hand and comes to rest, emits the env so
        # compute() can regenerate its trajectory from the new object pose. Disabled unless the cfg
        # opts in (see _update_drop_recovery); the tracker itself is cheap so it is always built.
        self._recovery = DropRecoveryTracker(self.num_envs, self.device, self.cfg.recovery_settle_steps)
        # Initial settle-capture bookkeeping: after a reset the object is still falling, so the
        # reset-time trajectory build captures the spawn pose. These buffers let compute() rebuild the
        # reach goal once the object has come to rest (see _update_settle_capture / reset).
        self._steps_since_reset = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_goal_captured = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        # Consecutive grasp-establishment steps the object has sat off its goal while at rest: a stuck
        # lift (grip never took / slipped and fell back). Reaching grasp_stall_steps replans the
        # trajectory from the object's settled pose (see _update_grasp_stall). Reset on every resample.
        self._grasp_stall_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Grasp reference (reach) pose captured at each (re)sample: the object's settled pose. While
        # the current stage is within cfg.hand_base_hold_until_stage the hand-base is held here (so the
        # arm stays at the grasp pose and the correction lifts the object). See _hand_base_reference_pose.
        self._grasp_ref_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._grasp_ref_pose_b[:, 3] = 1.0
        self.metrics["hand_base_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hand_base_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        # How far the *live* object sits (measured in the actual hand-base frame) from the commanded
        # object-in-hand pose: large during the pre-grasp reach, small once grasped, and -- unlike an
        # object-root distance -- invariant to in-hand reorientation. The hand-stretch gate thresholds it.
        self.metrics["hand_base_object_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["trajectory_command_achieved"] = torch.zeros(self.num_envs, device=self.device)
        # Per-env signal to the low-level hand gate: hold the hand open (stretch) on the approach stages.
        self.metrics["keep_hand_open"] = torch.zeros(self.num_envs, device=self.device)

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
            self._hand_base_reference_pose(env_ids),
            self._hand_base_to_anchor_pose[env_ids],
            self.pose_command_b.new_tensor(self.cfg.object_to_anchor_pose),
            self._anchor_correction(env_ids),
        )
        self.hand_base_pose_command_b[env_ids] = hand_base_pose
        self.anchor_pose_command_b[env_ids] = anchor_pose

    def _hand_base_reference_pose(self, env_ids: Sequence[int] | slice = slice(None)) -> torch.Tensor:
        """Object pose the hand-base target is derived from.

        Normally the commanded object goal (``pose_command_b``). While the current stage is within
        ``cfg.hand_base_hold_until_stage`` the hand-base is instead held at the grasp reference (reach)
        pose captured at resample, so the arm stays at the grasp pose and the bounded anchor correction
        -- not a hand-base jump ahead of the grip -- lifts the object to the raised goal. Guarded for
        the ``__init__``-ordering call before the stepper / reference buffer exist.
        """
        goal = self.pose_command_b[env_ids]
        stepper = getattr(self, "_stepper", None)
        grasp_ref = getattr(self, "_grasp_ref_pose_b", None)
        if self.cfg.hand_base_hold_until_stage < 0 or stepper is None or grasp_ref is None:
            return goal
        stage = stepper.step_to_stage[stepper.step[env_ids]]
        hold = (stage <= self.cfg.hand_base_hold_until_stage).unsqueeze(-1)
        return torch.where(hold, grasp_ref[env_ids], goal)

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
        self._update_settle_capture()
        self._update_drop_recovery()
        self._update_grasp_stall()
        self._update_keep_hand_open_metric()

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

        # Orientation-invariant "grasp held?" proxy: the *live* object's position in the *actual*
        # hand-base frame vs. the commanded object-in-hand position (command[:, :7]). Under a rigid
        # grasp both reorient together, so this stays small through an in-hand flip and only grows on a
        # real slip/drop (or during the pre-grasp reach, when the arm is still far from the grasp) --
        # exactly what the stretch gate and drop recovery must react to. Keying off the object *root*
        # position + a fixed offset instead would swing with the object origin during a large
        # reorientation (the origin sits off the grasp point) and spuriously trip the gate. Position only.
        object_pos_b, object_quat_b = self._current_object_pose_b()
        object_in_hand_pos, _ = subtract_frame_transforms(
            hand_base_pos_b, hand_base_quat_b, object_pos_b, object_quat_b
        )
        commanded_object_in_hand = self._object_pose_command_hand_base_b()
        self.metrics["hand_base_object_error"] = torch.norm(
            object_in_hand_pos - commanded_object_in_hand[:, :3], dim=-1
        )

        object_achieved = self._object_target_achieved()
        hand_base_achieved = self.metrics["hand_base_position_error"] < self.cfg.hand_base_position_tolerance
        hand_base_achieved &= self.metrics["hand_base_orientation_error"] < self.cfg.hand_base_orientation_tolerance
        achieved = object_achieved & hand_base_achieved

        self._trajectory_command_achieved[:] = achieved
        self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()

    def _update_keep_hand_open_metric(self, env_ids: Sequence[int] | slice = slice(None)) -> None:
        """Publish the hand-open signal from the final stage state for this control step."""
        stage = self._stepper.step_to_stage[self._stepper.step[env_ids]]
        self.metrics["keep_hand_open"][env_ids] = (stage <= self.cfg.hand_open_until_stage).float()

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
        # Capture the reach (grasp reference) pose = the step-0 object goal, so the hand-base can be
        # held here through the grasp-establishment stages while the correction lifts the object.
        self._grasp_ref_pose_b[env_ids_tensor] = self.pose_command_b[env_ids_tensor].clone()
        self._corr.reset(env_ids_tensor)
        # Reach (trajectory step 0): the object goal is held at its settled pose, so deriving the
        # hand-base from it produces a grasp pose over the object -- the (open) hand reaches to grasp.
        # Reset the corrector first so this uses zero correction (the nominal grasp anchor).
        self._update_hand_base_pose_command(env_ids_tensor)
        self._trajectory_command_achieved[env_ids_tensor] = False
        self._update_keep_hand_open_metric(env_ids_tensor)
        # Disarm the drop-recovery detector for the (re)sampled envs so the fresh pre-grasp reach is a
        # non-event until the object is secured again. Covers env-reset / timer resamples and the
        # on-demand recovery resample alike. Guarded for init ordering (the tracker is built late in
        # __init__, though _resample_command is not called during construction).
        recovery = getattr(self, "_recovery", None)
        if recovery is not None:
            recovery.reset(env_ids_tensor)
        # Clear the lift-stall counter too, so a replan from either detector starts the fresh reach with
        # a clean grasp-establishment timeout. Guarded for the same __init__-ordering reason.
        grasp_stall_counter = getattr(self, "_grasp_stall_counter", None)
        if grasp_stall_counter is not None:
            grasp_stall_counter[env_ids_tensor] = 0

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

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the command and arm the initial settle-capture for the reset envs.

        ``super().reset`` resamples the command -- the provisional build from the object's spawn pose,
        which is still falling. Arming the settle-capture makes :meth:`_update_settle_capture` rebuild
        the reach goal from the object's resting pose once it comes to rest.
        """
        extras = super().reset(env_ids)
        if self.cfg.capture_goal_after_settle:
            ids = (
                torch.arange(self.num_envs, device=self.device)
                if env_ids is None
                else self._env_ids_tensor(env_ids)
            )
            self._steps_since_reset[ids] = 0
            self._grasp_goal_captured[ids] = False
        return extras

    def _update_settle_capture(self) -> None:
        """Rebuild the reach goal from the object's settled pose once, shortly after a reset.

        The reset-time build captures the object at its spawn pose while it is still falling. Waiting
        for it to come to rest -- past a short ``settle_min_steps`` guard that skips the zero-velocity
        reset instant -- and rebuilding via :meth:`_resample_command` makes the reach goal sit on the
        object's true resting pose. Fires at most once per episode (the per-env ``_grasp_goal_captured``
        latch is cleared only in :meth:`reset`).
        """
        if not self.cfg.capture_goal_after_settle:
            return
        self._steps_since_reset += 1
        pending = ~self._grasp_goal_captured
        if not bool(pending.any()):
            return
        speed = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
        ready = pending & (self._steps_since_reset >= self.cfg.settle_min_steps) & (
            speed < self.cfg.recovery_settle_speed
        )
        ids = ready.nonzero().flatten()
        if ids.numel() > 0:
            self._resample_command(ids)
            self._grasp_goal_captured[ids] = True

    def _update_drop_recovery(self) -> None:
        """Detect a dropped object and regenerate its trajectory from the new resting pose.

        Runs after :meth:`_update_metrics` / :meth:`_update_command` each step. Feeds the per-env
        drop-recovery state machine three masks and resamples the envs it emits:

        * ``secured`` -- past the grasp-establishment stages (current stage index above
          ``recovery_arm_after_stage``) *and* the hand is at the live object's grasp anchor
          (``hand_base_object_error`` below ``drop_object_hand_distance``). Gating on the stage keeps
          the reach/lift phase -- where the hand rises to the lifted goal while the grasp is still
          forming -- from ever arming the detector and collapsing the goal.
        * ``dropped`` -- past those stages and the object has left the hand (distance *above* the
          threshold). Only meaningful for an already-armed env.
        * ``at_rest`` -- the object's speed is below ``recovery_settle_speed``; the machine waits for
          ``recovery_settle_steps`` consecutive at-rest steps before firing so it replans from the
          settled pose, not mid-bounce.

        :meth:`_resample_command` rebuilds the trajectory from the (new) live object pose and disarms
        the env via :meth:`DropRecoveryTracker.reset`.
        """
        if not self.cfg.enable_drop_recovery:
            return
        stage = self._stepper.step_to_stage[self._stepper.step]
        in_transport = stage > self.cfg.recovery_arm_after_stage
        near = self.metrics["hand_base_object_error"] < self.cfg.drop_object_hand_distance
        speed = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
        at_rest = speed < self.cfg.recovery_settle_speed
        settled = self._recovery.update(
            secured=in_transport & near, dropped=in_transport & ~near, at_rest=at_rest
        )
        if settled.numel() > 0:
            self._resample_command(settled)

    def _update_grasp_stall(self) -> None:
        """Replan when the grasp fails to lift the object out of the establishment stages.

        The hand<->object-distance drop detector (:meth:`_update_drop_recovery`) is structurally blind to
        a lift-stage failure: with the arm held over the peg only the bounded correction raises it, so a
        failed lift never grows ``hand_base_object_error`` past ``drop_object_hand_distance`` and the
        stepper stays stuck at ``settled + lift_height``. This detector instead keys off the object not
        reaching its goal, covering the grasp-establishment stages (``stage <= recovery_arm_after_stage``,
        the complement of the drop detector's transport range).

        Per env it counts consecutive steps where an establishment-stage object sits **off its goal**
        (the stepper can't advance) **and at rest** (``recovery_settle_speed``). A peg being lifted
        successfully is moving, so it doesn't accumulate and reaches its goal (advancing, which clears
        the count) well before the threshold; only a peg that stopped short of its lift goal -- grip
        never took, or slipped and fell back -- accumulates. At ``grasp_stall_steps`` the grasp is deemed
        failed and :meth:`_resample_command` rebuilds the trajectory from the (settled) object pose,
        restarting the reach so the hand re-opens and re-grasps. Reach never accumulates (its goal is the
        object's own settled pose, so the object is already at goal). Disabled unless the cfg opts in.
        """
        if not self.cfg.enable_drop_recovery or self.cfg.grasp_stall_steps <= 0:
            return
        stage = self._stepper.step_to_stage[self._stepper.step]
        in_establishment = stage <= self.cfg.recovery_arm_after_stage
        off_goal = ~self._object_target_achieved()
        speed = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
        at_rest = speed < self.cfg.recovery_settle_speed
        stalling = in_establishment & off_goal & at_rest
        self._grasp_stall_counter = torch.where(
            stalling,
            self._grasp_stall_counter + 1,
            torch.zeros_like(self._grasp_stall_counter),
        )
        failed = (self._grasp_stall_counter >= self.cfg.grasp_stall_steps).nonzero().flatten()
        if failed.numel() > 0:
            self._resample_command(failed)

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

    def _current_object_pose_b(
        self,
        env_ids: Sequence[int] | slice = slice(None),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.object.data.root_pos_w[env_ids],
            self.object.data.root_quat_w[env_ids],
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

    # -- Drop recovery: regenerate the trajectory from the object's new pose if it leaves the hand --
    enable_drop_recovery: bool = False
    """Detect a dropped object and replan the trajectory from its new resting pose (see
    :meth:`TrajectoryObjectAndHandBasePoseCommand._update_drop_recovery`). Disabled by default."""

    recovery_settle_speed: float = 0.05
    """Object speed (m/s) below which a dropped object counts as at-rest for recovery."""

    recovery_settle_steps: int = 5
    """Consecutive at-rest steps a dropped object must hold before the trajectory is regenerated."""

    drop_object_hand_distance: float = 0.20
    """Hand-base-to-live-object grasp-anchor distance (m) above which the object counts as dropped
    (and, below which, as secured to arm the detector)."""

    recovery_arm_after_stage: int = 0
    """Drop recovery arms only once the trajectory's current stage index exceeds this. Grasp-
    establishment stages (e.g. reach/lift) should be excluded so the hand rising to the lifted goal
    while the grasp is still forming is not mistaken for a drop (which would collapse the goal)."""

    capture_goal_after_settle: bool = False
    """After a reset, rebuild the reach goal from the object's settled resting pose (once it comes to
    rest) instead of the spawn pose captured while it is still falling. See ``_update_settle_capture``."""

    settle_min_steps: int = 5
    """Minimum steps after a reset before the settle-capture rebuild may fire, skipping the
    zero-velocity reset instant so a still-to-fall object is not treated as already settled."""

    grasp_stall_steps: int = -1
    """Lift-stall recovery: consecutive grasp-establishment-stage steps the object may sit off its goal
    (and at rest) before the grasp is deemed failed and the trajectory replanned from its settled pose.

    Complements the hand<->object-distance drop detector, which is blind to a lift-stage failure: while
    the arm is held over the peg (``hand_base_hold_until_stage``) only the bounded correction raises it,
    so a failed lift never grows ``hand_base_object_error`` past ``drop_object_hand_distance`` and the
    stepper stays stuck at ``settled + lift_height``. This detector instead keys off the object not
    reaching its lift goal. The two partition the stages at ``recovery_arm_after_stage``: distance drop
    for transport (stage above), lift-stall for establishment (stage at or below). ``-1``/``0`` disables
    it (default); requires ``enable_drop_recovery``. See ``_update_grasp_stall``."""

    # -- Grasp sequencing: keep the hand open on approach, hold the arm at the grasp pose on lift --
    hand_open_until_stage: int = -1
    """Publish ``metrics["keep_hand_open"]`` while the current stage index is <= this, so the
    low-level hand gate holds the hand open (stretch) through those (approach) stages instead of
    closing on distance. Default -1 disables it (gate stays distance-only)."""

    hand_base_hold_until_stage: int = -1
    """Hold the hand-base target at the captured grasp reference (reach) pose while the current stage
    index is <= this, instead of deriving it from the (lifted) object goal. The bounded anchor
    correction then lifts the object without the arm jumping up ahead of the grip. Default -1 disables
    it (hand-base always follows the object goal). Requires ``correction.enable`` to actually lift."""

    # -- Adaptive object->anchor correction (bounded PI(D) so the object reaches its goal) --
    correction: AnchorCorrectionCfg = AnchorCorrectionCfg()
    """Bounded PI(D) correction on the measured object->goal error. Disabled by default
    (``correction.enable = False``); see :class:`AnchorCorrectionCfg` for the gains/limits."""


__all__ = [
    "ALIGN_MARKER_CFG",
    "TrajectoryObjectAndHandBasePoseCommand",
    "TrajectoryObjectAndHandBasePoseCommandCfg",
]
