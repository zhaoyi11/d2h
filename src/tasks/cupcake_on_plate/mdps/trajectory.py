# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch reference generation for the Z-axis-constrained cupcake task."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from src.policy.high_level.utils import (
    _as_pose_tensor,
    _with_normalized_quat,
    build_object_pose_sequence_from_keyframes,
    compose_world_yaw,
)


DEFAULT_CUPCAKE_Z_AXIS_SEGMENT_STEPS = (0, 1)
"""Reach the fixed cupcake, then activate one yaw target."""

DEFAULT_CUPCAKE_Z_AXIS_STAGE_OBJECT_TOLERANCES = (
    (0.01, 0.2),  # reach
    (0.01, 0.2),  # yaw target
)

DEFAULT_CUPCAKE_Z_AXIS_YAW_DELTA_RANGE = (math.pi / 3.0, math.pi / 2.0)
"""Absolute yaw-goal range: 60 to 90 degrees."""


def sample_signed_yaw_deltas(
    count: int,
    yaw_delta_range: tuple[float, float] = DEFAULT_CUPCAKE_Z_AXIS_YAW_DELTA_RANGE,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniformly sized clockwise/counter-clockwise yaw deltas."""
    min_angle, max_angle = (float(value) for value in yaw_delta_range)
    if not (0.0 <= min_angle <= max_angle <= math.pi):
        raise ValueError("yaw_delta_range must satisfy 0 <= min <= max <= pi.")
    if count < 0:
        raise ValueError("count must be non-negative.")

    magnitude = torch.empty(count, dtype=dtype, device=device).uniform_(
        min_angle, max_angle, generator=generator
    )
    positive = torch.rand(count, device=device, generator=generator) >= 0.5
    sign = torch.where(positive, magnitude.new_tensor(1.0), magnitude.new_tensor(-1.0))
    return magnitude * sign


def build_cupcake_z_axis_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    yaw_delta: float | torch.Tensor,
    segment_steps: Sequence[int] = DEFAULT_CUPCAKE_Z_AXIS_SEGMENT_STEPS,
) -> torch.Tensor:
    """Build reach and fixed-position yaw-target poses in the robot-base frame.

    The robot base is fixed and world-aligned in this task, so composing an extrinsic base ``+Z``
    yaw matches the revolute joint's world ``+Z`` axis.
    """
    if len(segment_steps) != 2:
        raise ValueError("segment_steps must contain 2 values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")

    current = _with_normalized_quat(_as_pose_tensor(current_pose))
    target = torch.cat((current[:3], compose_world_yaw(current[3:7], yaw_delta)))
    key_poses = (current, current, target)
    return build_object_pose_sequence_from_keyframes(key_poses, segment_steps)


__all__ = [
    "DEFAULT_CUPCAKE_Z_AXIS_SEGMENT_STEPS",
    "DEFAULT_CUPCAKE_Z_AXIS_STAGE_OBJECT_TOLERANCES",
    "DEFAULT_CUPCAKE_Z_AXIS_YAW_DELTA_RANGE",
    "build_cupcake_z_axis_object_pose_sequence",
    "sample_signed_yaw_deltas",
]
