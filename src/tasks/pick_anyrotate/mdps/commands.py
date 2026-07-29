# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory command for picking an object and reaching six orientations."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.policy.high_level.trajectory_stepper import StageObjTol
from src.tasks.common.mdps.rewards import contacts as good_object_contact
from src.tasks.pick_anyrotate.mdps.trajectory import (
    DEFAULT_PICK_ANYROTATE_SEGMENT_STEPS,
    DEFAULT_PICK_ANYROTATE_STAGE_OBJECT_TOLERANCES,
    DEFAULT_ROTATION_DELTA_RANGE,
    build_pick_anyrotate_object_pose_sequence,
)


class PickAnyRotateTrajectoryObjectAndHandBasePoseCommand(TrajectoryObjectAndHandBasePoseCommand):
    """Pick, lift, and advance through six pre-sampled orientation targets."""

    cfg: PickAnyRotateTrajectoryObjectAndHandBasePoseCommandCfg

    _GRASP_ESTABLISH_STAGE = 1
    _FIRST_REORIENTATION_STAGE = 3
    _FINAL_REORIENTATION_STAGE = 8

    def __init__(self, cfg, env) -> None:
        if len(cfg.trajectory_segment_steps) != 9:
            raise ValueError("trajectory_segment_steps must contain 9 values.")
        if len(cfg.stage_object_tolerances) != 9:
            raise ValueError("stage_object_tolerances must contain 9 values.")
        if cfg.grasp_contact_force_threshold < 0.0:
            raise ValueError("grasp_contact_force_threshold must be non-negative.")
        if cfg.grasp_contact_stable_steps < 1:
            raise ValueError("grasp_contact_stable_steps must be at least 1.")
        if cfg.grasp_establish_timeout_steps < cfg.grasp_contact_stable_steps:
            raise ValueError("grasp_establish_timeout_steps must be at least grasp_contact_stable_steps.")
        if cfg.goal_hold_steps < 0:
            raise ValueError("goal_hold_steps must be non-negative.")

        super().__init__(cfg, env)
        self._grasp_contact_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._tracked_rotation_stage = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._rotation_goal_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rotation_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.metrics["reorientation_active"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reorientation_goal_achieved"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reorientation_targets_completed"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sequence_complete"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.cfg.capture_goal_after_settle:
            self._trajectory_command_achieved &= self._grasp_goal_captured
            self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()
        self._update_grasp_establish()
        self._update_reorientation_progress()

    def _update_grasp_establish(self) -> None:
        stage = self._stepper.step_to_stage[self._stepper.step]
        establishing = stage == self._GRASP_ESTABLISH_STAGE
        contact = good_object_contact(self._env, self.cfg.grasp_contact_force_threshold)
        self._grasp_phase_steps = torch.where(
            establishing,
            self._grasp_phase_steps + 1,
            torch.zeros_like(self._grasp_phase_steps),
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

    def _update_reorientation_progress(self) -> None:
        stage = self._stepper.step_to_stage[self._stepper.step]
        active = stage >= self._FIRST_REORIENTATION_STAGE
        stage_changed = stage != self._tracked_rotation_stage
        reset = ~active | stage_changed
        self._rotation_goal_latched[reset] = False
        self._rotation_hold_counter[reset] = 0
        self._tracked_rotation_stage[:] = stage

        first_success = active & self._trajectory_command_achieved & ~self._rotation_goal_latched
        self._rotation_goal_latched[first_success] = True
        self._rotation_hold_counter[first_success] = self.cfg.goal_hold_steps

        holding = active & self._rotation_goal_latched & ~first_success & (self._rotation_hold_counter > 0)
        self._rotation_hold_counter[holding] -= 1
        ready = active & self._rotation_goal_latched & (self._rotation_hold_counter == 0)
        self._trajectory_command_achieved[active] = ready[active]
        self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()
        self.metrics["reorientation_active"] = active.float()
        self.metrics["reorientation_goal_achieved"] = ready.float()

        completed_before_current = (stage - self._FIRST_REORIENTATION_STAGE).clamp(min=0, max=5)
        completed = torch.where(active, completed_before_current + ready.long(), torch.zeros_like(stage))
        self.metrics["reorientation_targets_completed"] = completed.float()
        self.metrics["sequence_complete"] = (ready & (stage == self._FINAL_REORIENTATION_STAGE)).float()

    def _resample_command(self, env_ids) -> None:
        env_ids_tensor = self._env_ids_tensor(env_ids)
        super()._resample_command(env_ids_tensor)
        for name, value in (
            ("_grasp_contact_streak", 0),
            ("_grasp_phase_steps", 0),
            ("_tracked_rotation_stage", -1),
            ("_rotation_goal_latched", False),
            ("_rotation_hold_counter", 0),
        ):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer[env_ids_tensor] = value
        for metric_name in (
            "reorientation_active",
            "reorientation_goal_achieved",
            "reorientation_targets_completed",
            "sequence_complete",
        ):
            metric = self.metrics.get(metric_name)
            if metric is not None:
                metric[env_ids_tensor] = 0.0

    def _apply_objanchor_correction(self, env_ids: torch.Tensor) -> None:
        stage = self._stepper.step_to_stage[self._stepper.step[env_ids]]
        super()._apply_objanchor_correction(env_ids[stage != self._GRASP_ESTABLISH_STAGE])

    def _build_object_trajectories(self, env_ids: torch.Tensor, current_pose_b: torch.Tensor) -> torch.Tensor:
        trajectories = [
            build_pick_anyrotate_object_pose_sequence(
                current_pose_b[env_idx],
                segment_steps=self.cfg.trajectory_segment_steps,
                lift_height=self.cfg.lift_height,
                rotation_delta_range=self.cfg.rotation_delta_range,
            )
            for env_idx in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)


@configclass
class PickAnyRotateTrajectoryObjectAndHandBasePoseCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    """Configuration for the pick-anyrotate trajectory command."""

    class_type: type = PickAnyRotateTrajectoryObjectAndHandBasePoseCommand
    trajectory_segment_steps: tuple[int, ...] = DEFAULT_PICK_ANYROTATE_SEGMENT_STEPS
    stage_object_tolerances: tuple[StageObjTol, ...] = DEFAULT_PICK_ANYROTATE_STAGE_OBJECT_TOLERANCES
    lift_height: float = 0.10
    rotation_delta_range: tuple[float, float] = DEFAULT_ROTATION_DELTA_RANGE
    grasp_contact_force_threshold: float = 1.0
    grasp_contact_stable_steps: int = 3
    grasp_establish_timeout_steps: int = 30
    goal_hold_steps: int = 20


__all__ = [
    "PickAnyRotateTrajectoryObjectAndHandBasePoseCommand",
    "PickAnyRotateTrajectoryObjectAndHandBasePoseCommandCfg",
]
