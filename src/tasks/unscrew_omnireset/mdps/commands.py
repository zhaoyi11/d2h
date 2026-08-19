# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Object-pose trajectory command for unscrew OmniReset."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    subtract_frame_transforms,
)

from src.policy.high_level.trajectory_stepper import TrajectoryStepper
from src.tasks.unscrew.mdps.trajectory import build_unscrew_object_pose_sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UnscrewOmniResetTrajectoryCommand(CommandTerm):
    """Track a reset-relative helical unscrew path with a final vertical lift."""

    cfg: UnscrewOmniResetTrajectoryCommandCfg

    def __init__(
        self,
        cfg: UnscrewOmniResetTrajectoryCommandCfg,
        env: ManagerBasedRLEnv,
    ) -> None:
        expected_segments = cfg.twist_segments + 2
        if len(cfg.trajectory_segment_steps) != expected_segments:
            raise ValueError(
                f"trajectory_segment_steps must contain {expected_segments} values."
            )
        if cfg.position_threshold < 0.0 or cfg.orientation_threshold < 0.0:
            raise ValueError("trajectory thresholds must be non-negative.")

        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        self._stepper = TrajectoryStepper(
            num_envs=self.num_envs,
            device=self.device,
            segment_steps=cfg.trajectory_segment_steps,
            stage_position_tolerance=torch.full(
                (expected_segments,), cfg.position_threshold, device=self.device
            ),
            stage_orientation_tolerance=torch.full(
                (expected_segments,), cfg.orientation_threshold, device=self.device
            ),
        )

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _env_ids_tensor(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids_tensor = self._env_ids_tensor(env_ids)
        object_pos_b, object_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids_tensor],
            self.robot.data.root_quat_w[env_ids_tensor],
            self.object.data.root_pos_w[env_ids_tensor],
            self.object.data.root_quat_w[env_ids_tensor],
        )
        current_pose_b = torch.cat((object_pos_b, object_quat_b), dim=1)
        builder_segment_steps = (0, 0) + tuple(self.cfg.trajectory_segment_steps)
        trajectories = torch.stack(
            [
                build_unscrew_object_pose_sequence(
                    current_pose_b[index],
                    segment_steps=builder_segment_steps,
                    twist_total_angle=self.cfg.twist_total_angle,
                    twist_segments=self.cfg.twist_segments,
                    thread_pitch=self.cfg.thread_pitch,
                    extraction_height=self.cfg.extraction_height,
                )
                for index in range(env_ids_tensor.numel())
            ]
        )
        self._stepper.reset(env_ids_tensor, trajectories)
        self.pose_command_b[env_ids_tensor] = self._stepper.current_object_pose(
            env_ids_tensor
        )

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
        achieved = self._stepper.object_target_achieved(
            self.metrics["position_error"],
            self.metrics["orientation_error"],
            position_only=False,
        )
        env_ids = achieved.nonzero().flatten()
        self._stepper.advance(env_ids)
        self.pose_command_b[env_ids] = self._stepper.current_object_pose(env_ids)


@configclass
class UnscrewOmniResetTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the reset-relative twist/lift object trajectory."""

    class_type: type = UnscrewOmniResetTrajectoryCommand

    asset_name: str = MISSING
    object_name: str = MISSING
    trajectory_segment_steps: tuple[int, ...] = (2,) * 6 + (1, 1)
    twist_total_angle: float = 2.0 * math.pi
    twist_segments: int = 6
    thread_pitch: float = 0.02
    extraction_height: float = 0.10
    position_threshold: float = 0.005
    orientation_threshold: float = 0.20


__all__ = [
    "UnscrewOmniResetTrajectoryCommand",
    "UnscrewOmniResetTrajectoryCommandCfg",
]
