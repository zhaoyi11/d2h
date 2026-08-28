# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single yaw-goal command for the Z-axis-constrained object task."""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.policy.high_level.trajectory_stepper import StageObjTol
from src.tasks.rotate_object_once.mdps.trajectory import (
    DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_SEGMENT_STEPS,
    DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_STAGE_OBJECT_TOLERANCES,
    DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_YAW_DELTA_RANGE,
    build_rotate_object_once_z_axis_object_pose_sequence,
    sample_signed_yaw_deltas,
)


class RotateObjectOnceTrajectoryObjectAndHandBasePoseCommand(TrajectoryObjectAndHandBasePoseCommand):
    """Expose one yaw-only object goal while preserving an open-hand reach stage."""

    cfg: RotateObjectOnceTrajectoryObjectAndHandBasePoseCommandCfg

    def __init__(self, cfg, env):
        runtime_cfg = cfg.replace(
            stage_object_tolerances=tuple(StageObjTol(*values) for values in cfg.stage_object_tolerances)
        )
        super().__init__(runtime_cfg, env)
        self.metrics["yaw_target_active"] = torch.zeros(self.num_envs, device=self.device)

    def _build_object_trajectories(self, env_ids: torch.Tensor, current_pose_b: torch.Tensor) -> torch.Tensor:
        yaw_deltas = sample_signed_yaw_deltas(
            env_ids.numel(),
            self.cfg.yaw_delta_range,
            dtype=current_pose_b.dtype,
            device=self.device,
        )
        trajectories = [
            build_rotate_object_once_z_axis_object_pose_sequence(
                current_pose_b[env_idx],
                yaw_delta=yaw_deltas[env_idx],
                segment_steps=self.cfg.trajectory_segment_steps,
            )
            for env_idx in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)

    def _object_target_achieved(self) -> torch.Tensor:
        """Make stage 0 hand-arrival-only while stage 1 tracks the sampled yaw target."""
        achieved = super()._object_target_achieved()
        return achieved | (self._stepper.step == 0)

    def _activate_sampled_yaw_target(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._stepper.advance(env_ids)
        self.pose_command_b[env_ids] = self._stepper.current_object_pose(env_ids)
        self._corr.clear_stall(env_ids)
        self._trajectory_command_achieved[env_ids] = False
        self.metrics["trajectory_command_achieved"][env_ids] = 0.0

    def _update_command(self) -> None:
        achieved_env_ids = self._trajectory_command_achieved.nonzero().flatten()
        if achieved_env_ids.numel() > 0:
            at_final_target = self._stepper.step[achieved_env_ids] == self._stepper.length - 1
            reach_env_ids = achieved_env_ids[~at_final_target]
            self._activate_sampled_yaw_target(reach_env_ids)

        active_env_ids = (self._stepper.step > 0).nonzero().flatten()
        if active_env_ids.numel() > 0:
            self._apply_objanchor_correction(active_env_ids)
            self._update_hand_base_pose_command(active_env_ids)
        self.metrics["yaw_target_active"] = (self._stepper.step > 0).float()


@configclass
class RotateObjectOnceTrajectoryObjectAndHandBasePoseCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    """Configuration for one fixed-position object yaw goal."""

    class_type: type = RotateObjectOnceTrajectoryObjectAndHandBasePoseCommand

    yaw_delta_range: tuple[float, float] = DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_YAW_DELTA_RANGE
    """Uniform absolute yaw delta range in radians; direction is sampled independently."""

    trajectory_segment_steps: tuple[int, ...] = DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_SEGMENT_STEPS
    """Reach and yaw-target interpolation samples."""

    stage_object_tolerances: tuple[tuple[float, float], ...] = DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_STAGE_OBJECT_TOLERANCES
    """Object pose tolerances for the reach and yaw-target stages."""


__all__ = [
    "DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_SEGMENT_STEPS",
    "DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_STAGE_OBJECT_TOLERANCES",
    "DEFAULT_ROTATE_OBJECT_ONCE_Z_AXIS_YAW_DELTA_RANGE",
    "RotateObjectOnceTrajectoryObjectAndHandBasePoseCommand",
    "RotateObjectOnceTrajectoryObjectAndHandBasePoseCommandCfg",
]
