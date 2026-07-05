# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment stage/step state machine for a scripted object-pose trajectory.

Pure ``torch`` (no isaaclab), so it can be loaded by file path and unit-tested in isolation --
mirroring ``object_trajectory.py`` / ``anchor_correction.py``. It owns the per-env waypoint buffer,
the current step index, the step->stage map, and the per-stage advance tolerances; the command term
orchestrates it (it still owns ``pose_command_b`` and the hand-base/correction targets). It is
task-agnostic: any trajectory expressed as ``1 + sum(segment_steps)`` waypoints can be stepped here.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


class TrajectoryStepper:
    """Holds the per-env object-pose waypoint sequence and advances a step index through it.

    A trajectory has ``1 + sum(segment_steps)`` waypoints: the initial pose followed by the
    interpolation samples of each segment (the number and meaning of segments is task-defined). Each
    step maps to a stage (via :attr:`step_to_stage`) whose position/orientation tolerance decides when
    the object target counts as achieved and the step may advance.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        segment_steps: Sequence[int],
        stage_position_tolerance: torch.Tensor,
        stage_orientation_tolerance: torch.Tensor,
    ) -> None:
        self.num_envs = num_envs
        self.device = device
        self._length = 1 + sum(segment_steps)
        stage_ids = [0]
        for stage_idx, steps in enumerate(segment_steps):
            stage_ids.extend([stage_idx] * steps)
        self._step_to_stage = torch.tensor(stage_ids, dtype=torch.long, device=device)
        self._stage_position_tolerance = stage_position_tolerance
        self._stage_orientation_tolerance = stage_orientation_tolerance
        self._step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._trajectory = torch.zeros(num_envs, self._length, 7, device=device)
        self._trajectory[:, :, 3] = 1.0

    @property
    def length(self) -> int:
        """Number of waypoints per trajectory (``1 + sum(segment_steps)``)."""
        return self._length

    @property
    def step(self) -> torch.Tensor:
        """Per-env current step index. Shape ``(num_envs,)``."""
        return self._step

    @property
    def step_to_stage(self) -> torch.Tensor:
        """Map from step index to stage index. Shape ``(length,)``."""
        return self._step_to_stage

    def reset(self, env_ids: torch.Tensor, trajectories: torch.Tensor) -> None:
        """Store new waypoint sequences for ``env_ids`` and rewind them to step 0."""
        self._trajectory[env_ids] = trajectories
        self._step[env_ids] = 0

    def current_object_pose(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Waypoint at the current step for ``env_ids``. Shape ``(len(env_ids), 7)``."""
        return self._trajectory[env_ids, self._step[env_ids]]

    def object_target_achieved(
        self,
        position_error: torch.Tensor,
        orientation_error: torch.Tensor,
        position_only: bool,
    ) -> torch.Tensor:
        """Per-env mask: is the object within the current stage's tolerance? Shape ``(num_envs,)``."""
        stage = self._step_to_stage[self._step]
        achieved = position_error < self._stage_position_tolerance[stage]
        if not position_only:
            achieved = achieved & (orientation_error < self._stage_orientation_tolerance[stage])
        return achieved

    def advance(self, env_ids: torch.Tensor) -> None:
        """Advance the step for ``env_ids`` by one, clamped at the final waypoint."""
        self._step[env_ids] = torch.clamp(self._step[env_ids] + 1, max=self._length - 1)


__all__ = ["TrajectoryStepper"]
