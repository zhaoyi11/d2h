# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted unscrew command with contact-confirmed grasp establishment."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.policy.high_level.trajectory_stepper import StageObjTol
from src.tasks.common.mdps.contacts import contacts as good_object_contact
from src.tasks.unscrew.geometry import bolt_aabb_corners, bolt_bottom_clearance
from src.tasks.unscrew.mdps.trajectory import (
    DEFAULT_UNSCREW_EXTRACTION_HEIGHT,
    DEFAULT_UNSCREW_SEGMENT_STEPS,
    DEFAULT_UNSCREW_STAGE_OBJECT_TOLERANCES,
    DEFAULT_UNSCREW_THREAD_PITCH,
    DEFAULT_UNSCREW_TWIST_SEGMENTS,
    DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE,
    build_unscrew_object_pose_sequence,
    build_unscrew_vertical_lift_pose_sequence,
)


class UnscrewTrajectoryObjectAndHandBasePoseCommand(TrajectoryObjectAndHandBasePoseCommand):
    """Drive the hand to an installed leg, confirm grasp, then follow a helical extraction path."""

    cfg: UnscrewTrajectoryObjectAndHandBasePoseCommandCfg
    _GRASP_ESTABLISH_STAGE = 1

    def __init__(self, cfg, env) -> None:
        expected_segments = cfg.twist_segments + 4
        if len(cfg.trajectory_segment_steps) != expected_segments:
            raise ValueError(f"trajectory_segment_steps must contain {expected_segments} values.")
        if len(cfg.stage_object_tolerances) != expected_segments:
            raise ValueError(f"stage_object_tolerances must contain {expected_segments} values.")
        if cfg.grasp_contact_force_threshold < 0.0:
            raise ValueError("grasp_contact_force_threshold must be non-negative.")
        if cfg.grasp_contact_stable_steps < 1:
            raise ValueError("grasp_contact_stable_steps must be at least 1.")
        if cfg.grasp_establish_timeout_steps < cfg.grasp_contact_stable_steps:
            raise ValueError("grasp_establish_timeout_steps must be at least grasp_contact_stable_steps.")
        if cfg.clearance_transition_margin < 0.0:
            raise ValueError("clearance_transition_margin must be non-negative.")
        if cfg.clearance_transition_stable_steps < 1:
            raise ValueError("clearance_transition_stable_steps must be at least 1.")
        if cfg.enable_clearance_transition and cfg.trajectory_segment_steps[-2] < 1:
            raise ValueError("the vertical lift segment must contain at least one step.")
        super().__init__(cfg, env)
        self._grasp_contact_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._clearance_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._clearance_transition_pending = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._clearance_transitioned = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.metrics["bolt_clearance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["clearance_transitioned"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["vertical_lift_complete"] = torch.zeros(self.num_envs, device=self.device)
        if cfg.enable_clearance_transition:
            self.receptive = env.scene[cfg.receptive_name]
            self._bolt_corners = bolt_aabb_corners(self.device)

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.cfg.capture_goal_after_settle:
            self._trajectory_command_achieved &= self._grasp_goal_captured
            self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()
        self._update_grasp_establish()
        self._update_clearance_transition()
        self._update_vertical_lift_complete()

    def _compute_bolt_clearance(self) -> torch.Tensor:
        object_pos_r, object_quat_r = subtract_frame_transforms(
            self.receptive.data.root_pos_w,
            self.receptive.data.root_quat_w,
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
        )
        return bolt_bottom_clearance(object_pos_r, object_quat_r, self._bolt_corners)

    def _update_clearance_transition(self) -> None:
        if not self.cfg.enable_clearance_transition:
            return
        clearance = self._compute_bolt_clearance()
        self.metrics["bolt_clearance"] = clearance
        stage = self._stepper.step_to_stage[self._stepper.step]
        in_twist = (stage >= 2) & (stage < self.cfg.twist_segments + 2)
        grasped = good_object_contact(self._env, self.cfg.grasp_contact_force_threshold)
        confirmed = (
            in_twist
            & ~self._clearance_transitioned
            & (clearance >= self.cfg.clearance_transition_margin)
            & grasped
        )
        self._clearance_streak = torch.where(
            confirmed,
            self._clearance_streak + 1,
            torch.zeros_like(self._clearance_streak),
        )
        self._clearance_transition_pending = (
            self._clearance_streak >= self.cfg.clearance_transition_stable_steps
        )

    def _update_vertical_lift_complete(self) -> None:
        hold_stage = self.cfg.twist_segments + 3
        stage = self._stepper.step_to_stage[self._stepper.step]
        complete = (stage == hold_stage) & self._trajectory_command_achieved
        self.metrics["vertical_lift_complete"] = complete.float()

    def _transition_cleared_to_lift(self) -> None:
        env_ids = self._clearance_transition_pending.nonzero().flatten()
        if env_ids.numel() == 0:
            return

        current_pos_b, current_quat_b = self._current_object_pose_b(env_ids)
        current_pose_b = torch.cat((current_pos_b, current_quat_b), dim=1)
        lift_stage = self.cfg.twist_segments + 2
        lift_step = 1 + sum(self.cfg.trajectory_segment_steps[:lift_stage])
        remaining_segment_steps = self.cfg.trajectory_segment_steps[lift_stage:]
        remaining_waypoints = torch.stack(
            [
                build_unscrew_vertical_lift_pose_sequence(
                    current_pose_b[index],
                    segment_steps=remaining_segment_steps,
                    extraction_height=self.cfg.extraction_height,
                )[1:]
                for index in range(env_ids.numel())
            ],
            dim=0,
        )
        self._stepper.transition_to_step(env_ids, lift_step, remaining_waypoints)
        self.pose_command_b[env_ids] = self._stepper.current_object_pose(env_ids)
        self._corr.reset(env_ids)
        self._grasp_stall_counter[env_ids] = 0
        self._trajectory_command_achieved[env_ids] = False
        self._clearance_transition_pending[env_ids] = False
        self._clearance_transitioned[env_ids] = True
        self.metrics["clearance_transitioned"] = self._clearance_transitioned.float()

    def _update_command(self) -> None:
        if self.cfg.enable_clearance_transition:
            self._transition_cleared_to_lift()
        super()._update_command()

    def _update_grasp_establish(self) -> None:
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
        if hasattr(self, "_grasp_contact_streak"):
            self._grasp_contact_streak[env_ids_tensor] = 0
            self._grasp_phase_steps[env_ids_tensor] = 0
        if hasattr(self, "_clearance_streak"):
            self._clearance_streak[env_ids_tensor] = 0
            self._clearance_transition_pending[env_ids_tensor] = False
            self._clearance_transitioned[env_ids_tensor] = False
            self.metrics["bolt_clearance"][env_ids_tensor] = 0.0
            self.metrics["clearance_transitioned"] = self._clearance_transitioned.float()
            self.metrics["vertical_lift_complete"][env_ids_tensor] = 0.0

    def _apply_objanchor_correction(self, env_ids: torch.Tensor) -> None:
        stage = self._stepper.step_to_stage[self._stepper.step[env_ids]]
        super()._apply_objanchor_correction(env_ids[stage != self._GRASP_ESTABLISH_STAGE])

    def _build_object_trajectories(self, env_ids: torch.Tensor, current_pose_b: torch.Tensor) -> torch.Tensor:
        trajectories = [
            build_unscrew_object_pose_sequence(
                current_pose_b[env_index],
                segment_steps=self.cfg.trajectory_segment_steps,
                twist_total_angle=self.cfg.twist_total_angle,
                twist_segments=self.cfg.twist_segments,
                thread_pitch=self.cfg.thread_pitch,
                extraction_height=self.cfg.extraction_height,
            )
            for env_index in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)


@configclass
class UnscrewTrajectoryObjectAndHandBasePoseCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    """Configuration for the installed-leg unscrew trajectory."""

    class_type: type = UnscrewTrajectoryObjectAndHandBasePoseCommand

    trajectory_segment_steps: tuple[int, ...] = DEFAULT_UNSCREW_SEGMENT_STEPS
    twist_total_angle: float = DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE
    twist_segments: int = DEFAULT_UNSCREW_TWIST_SEGMENTS
    thread_pitch: float = DEFAULT_UNSCREW_THREAD_PITCH
    extraction_height: float = DEFAULT_UNSCREW_EXTRACTION_HEIGHT

    enable_clearance_transition: bool = False
    receptive_name: str = "receptive_object"
    clearance_transition_margin: float = 0.005
    clearance_transition_stable_steps: int = 3

    grasp_contact_force_threshold: float = 1.0
    grasp_contact_stable_steps: int = 3
    grasp_establish_timeout_steps: int = 30

    stage_object_tolerances: tuple[StageObjTol, ...] = DEFAULT_UNSCREW_STAGE_OBJECT_TOLERANCES


__all__ = [
    "DEFAULT_UNSCREW_EXTRACTION_HEIGHT",
    "DEFAULT_UNSCREW_SEGMENT_STEPS",
    "DEFAULT_UNSCREW_STAGE_OBJECT_TOLERANCES",
    "DEFAULT_UNSCREW_THREAD_PITCH",
    "DEFAULT_UNSCREW_TWIST_SEGMENTS",
    "DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE",
    "UnscrewTrajectoryObjectAndHandBasePoseCommand",
    "UnscrewTrajectoryObjectAndHandBasePoseCommandCfg",
]
