# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-insert scripted-trajectory command term.

Thin task layer over the generic
:class:`~src.policy.high_level.trajectory_command.TrajectoryObjectAndHandBasePoseCommand`: it overrides
:meth:`_build_object_trajectories` with the pick-insert scripted trajectory and optionally gates
its establish-grasp phase on stable fingertip contact. The trajectory is built by
:func:`build_pick_insert_object_pose_sequence`; all shared machinery
(stepper, PI(D) anchor correction, hand-base targeting, advance logic) is inherited.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.tasks.common.mdps.contacts import contacts as good_object_contact
from src.tasks.pick_insert.mdps.trajectory import (build_pick_insert_object_pose_sequence,
                                                   DEFAULT_PICK_INSERT_RECEPTIVE_POSE,
                                                   DEFAULT_PICK_INSERT_SEGMENT_STEPS,
                                                   DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES)


class PickInsertTrajectoryObjectAndHandBasePoseCommand(TrajectoryObjectAndHandBasePoseCommand):
    """Object and hand-base command that follows the pick-insert demo trajectory."""

    cfg: PickInsertTrajectoryObjectAndHandBasePoseCommandCfg

    _GRASP_ESTABLISH_STAGE = 1

    def __init__(self, cfg, env) -> None:
        if cfg.enable_grasp_establish:
            if len(cfg.trajectory_segment_steps) != 8:
                raise ValueError("trajectory_segment_steps must contain 8 values when grasp establish is enabled.")
            if len(cfg.stage_object_tolerances) != 8:
                raise ValueError("stage_object_tolerances must contain 8 values when grasp establish is enabled.")
            if cfg.grasp_contact_force_threshold < 0.0:
                raise ValueError("grasp_contact_force_threshold must be non-negative.")
            if cfg.grasp_contact_stable_steps < 1:
                raise ValueError("grasp_contact_stable_steps must be at least 1.")
            if cfg.grasp_establish_timeout_steps < cfg.grasp_contact_stable_steps:
                raise ValueError("grasp_establish_timeout_steps must be at least grasp_contact_stable_steps.")
        super().__init__(cfg, env)
        self._grasp_contact_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.cfg.enable_grasp_establish and self.cfg.capture_goal_after_settle:
            self._trajectory_command_achieved &= self._grasp_goal_captured
            self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()
        self._update_grasp_establish()

    def _update_grasp_establish(self) -> None:
        if not self.cfg.enable_grasp_establish:
            return

        stage = self._stepper.step_to_stage[self._stepper.step]
        establishing = stage == self._GRASP_ESTABLISH_STAGE
        contact = good_object_contact(self._env, self.cfg.grasp_contact_force_threshold)
        self._grasp_phase_steps = torch.where(
            establishing, self._grasp_phase_steps + 1, torch.zeros_like(self._grasp_phase_steps)
        )
        self._grasp_contact_streak = torch.where(
            establishing & contact,
            self._grasp_contact_streak + 1,
            torch.zeros_like(self._grasp_contact_streak),
        )
        confirmed = self._grasp_contact_streak >= self.cfg.grasp_contact_stable_steps
        self._trajectory_command_achieved[establishing] = confirmed[establishing]
        self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()

        timed_out = establishing & ~confirmed & (
            self._grasp_phase_steps >= self.cfg.grasp_establish_timeout_steps
        )
        retry_env_ids = timed_out.nonzero().flatten()
        if retry_env_ids.numel() > 0:
            self._resample_command(retry_env_ids)
            self._steps_since_reset[retry_env_ids] = 0
            self._grasp_goal_captured[retry_env_ids] = False

    def _resample_command(self, env_ids) -> None:
        env_ids_tensor = self._env_ids_tensor(env_ids)
        super()._resample_command(env_ids_tensor)
        contact_streak = getattr(self, "_grasp_contact_streak", None)
        phase_steps = getattr(self, "_grasp_phase_steps", None)
        if contact_streak is not None:
            contact_streak[env_ids_tensor] = 0
        if phase_steps is not None:
            phase_steps[env_ids_tensor] = 0

    def _apply_objanchor_correction(self, env_ids: torch.Tensor) -> None:
        if self.cfg.enable_grasp_establish:
            stage = self._stepper.step_to_stage[self._stepper.step[env_ids]]
            env_ids = env_ids[stage != self._GRASP_ESTABLISH_STAGE]
        super()._apply_objanchor_correction(env_ids)

    def _build_object_trajectories(self, env_ids: torch.Tensor, current_pose_b: torch.Tensor) -> torch.Tensor:
        receptive_pose = torch.tensor(self.cfg.receptive_pose, dtype=current_pose_b.dtype, device=self.device)
        trajectories = [
            build_pick_insert_object_pose_sequence(
                current_pose_b[env_idx],
                receptive_pose=receptive_pose,
                segment_steps=self.cfg.trajectory_segment_steps,
                above_offset=self.cfg.above_offset,
                insertion_depth=self.cfg.insertion_depth,
                approach_height=self.cfg.approach_height,
                lift_height=self.cfg.lift_height,
                enable_grasp_establish=self.cfg.enable_grasp_establish,
            )
            for env_idx in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)


@configclass
class PickInsertTrajectoryObjectAndHandBasePoseCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    """Configuration for the pick-insert trajectory command."""

    class_type: type = PickInsertTrajectoryObjectAndHandBasePoseCommand

    receptive_pose: tuple[float, float, float, float, float, float, float] = DEFAULT_PICK_INSERT_RECEPTIVE_POSE
    """Target receptacle pose used by the pick-insert object trajectory."""

    trajectory_segment_steps: tuple[int, ...] = DEFAULT_PICK_INSERT_SEGMENT_STEPS
    """Interpolation samples for the task stages. Eight entries are required when grasp establish is enabled."""

    above_offset: float = 0.10
    """Height above the receptacle for the initial move and orientation alignment."""

    insertion_depth: float = 0.06
    """Inserted object height offset above the receptacle pose."""

    approach_height: float = 0.01
    """Height above the insertion pose used before final descent."""

    lift_height: float = 0.03
    """Height (m) the lift-stage goal sits above the object's settled (reach) pose, so the grip must
    lift the object this far to advance out of the lift stage (an implicit grip-secured check)."""

    enable_grasp_establish: bool = False
    """Insert a contact-confirmed establish-grasp stage between reach and lift."""

    grasp_contact_force_threshold: float = 1.0
    """Per-fingertip object-contact force threshold (N) used to confirm the grasp."""

    grasp_contact_stable_steps: int = 3
    """Consecutive good-contact control steps required before advancing to lift."""

    grasp_establish_timeout_steps: int = 30
    """Establish-grasp steps allowed before reopening the hand and retrying from the settled object."""

    stage_object_tolerances: tuple[StageObjTol, ...] = DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES
    """Per-stage object position and orientation tolerances for advancing the command trajectory.

    One entry is required for every configured trajectory segment. The optional establish-grasp
    entry sits between reach and lift.
    """


__all__ = [
    "DEFAULT_PICK_INSERT_RECEPTIVE_POSE",
    "DEFAULT_PICK_INSERT_SEGMENT_STEPS",
    "DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES",
    "PickInsertTrajectoryObjectAndHandBasePoseCommand",
    "PickInsertTrajectoryObjectAndHandBasePoseCommandCfg",
]
