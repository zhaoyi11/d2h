# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch pick, lift, and six-target reorientation trajectory builder."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.utils import (
    _as_pose_tensor,
    _with_normalized_quat,
    build_object_pose_sequence_from_keyframes,
)


DEFAULT_PICK_ANYROTATE_SEGMENT_STEPS = (0, 1, 1, 1, 1, 1, 1, 1, 1)
"""Reach, establish grasp, lift, and six reorientation segments."""

DEFAULT_PICK_ANYROTATE_STAGE_OBJECT_TOLERANCES = (
    StageObjTol(0.02, 0.3),  # reach
    StageObjTol(0.02, 0.3),  # establish grasp
    StageObjTol(0.02, 0.3),  # lift
    StageObjTol(0.02, 0.3),  # target 1
    StageObjTol(0.02, 0.3),  # target 2
    StageObjTol(0.02, 0.3),  # target 3
    StageObjTol(0.02, 0.3),  # target 4
    StageObjTol(0.02, 0.3),  # target 5
    StageObjTol(0.02, 0.3),  # target 6
)

DEFAULT_ROTATION_DELTA_RANGE = (0.3 * math.pi, 0.5 * math.pi)


def build_pick_anyrotate_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    segment_steps: Sequence[int] = DEFAULT_PICK_ANYROTATE_SEGMENT_STEPS,
    lift_height: float = 0.10,
    rotation_delta_range: tuple[float, float] = DEFAULT_ROTATION_DELTA_RANGE,
) -> torch.Tensor:
    """Build reach -> establish -> lift -> six chained orientation targets.

    Poses are ``(x, y, z, qw, qx, qy, qz)`` in the robot root frame. All
    reorientation targets share the lifted position. Each target orientation is
    sampled by right-multiplying the preceding target by an isotropic SO(3)
    delta whose geodesic angle lies in ``rotation_delta_range``.
    """
    if len(segment_steps) != 9:
        raise ValueError("segment_steps must contain 9 values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")
    if lift_height <= 0.0:
        raise ValueError("lift_height must be positive.")

    current = _with_normalized_quat(_as_pose_tensor(current_pose))
    lifted_pos = current[:3].clone()
    lifted_pos[2] += current.new_tensor(lift_height)
    lifted = torch.cat((lifted_pos, current[3:7]))

    deltas = _sample_rotation_deltas(
        6,
        rotation_delta_range,
        dtype=current.dtype,
        device=current.device,
    )
    targets = []
    target_quat = current[3:7]
    for delta in deltas:
        target_quat = _normalize_quat(_quat_mul(target_quat, delta))
        targets.append(torch.cat((lifted_pos, target_quat)))

    keyframes = (current, current, current, lifted, *targets)
    return build_object_pose_sequence_from_keyframes(keyframes, segment_steps)


def _sample_rotation_deltas(
    count: int,
    angle_range: tuple[float, float],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    min_angle, max_angle = (float(value) for value in angle_range)
    if not (0.0 <= min_angle <= max_angle <= math.pi):
        raise ValueError("rotation_delta_range must satisfy 0 <= min <= max <= pi.")

    # Haar measure on SO(3): p(theta) is proportional to sin(theta / 2)^2,
    # with unnormalized CDF 0.5 * (theta - sin(theta)).
    lo = 0.5 * (min_angle - math.sin(min_angle))
    hi = 0.5 * (max_angle - math.sin(max_angle))
    u = torch.rand(count, dtype=dtype, device=device) * (hi - lo) + lo
    angle = (12.0 * u).clamp_min(torch.finfo(dtype).eps).pow(1.0 / 3.0)
    for _ in range(4):
        residual = 0.5 * (angle - torch.sin(angle)) - u
        derivative = 0.5 * (1.0 - torch.cos(angle))
        angle = angle - residual / derivative.clamp_min(torch.finfo(dtype).eps)
    angle = angle.clamp(min_angle, max_angle)

    axis = torch.randn(count, 3, dtype=dtype, device=device)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(torch.finfo(dtype).eps)
    half_angle = 0.5 * angle
    return torch.cat((torch.cos(half_angle).unsqueeze(-1), axis * torch.sin(half_angle).unsqueeze(-1)), dim=-1)


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat).clamp_min(torch.finfo(quat.dtype).eps)


__all__ = [
    "DEFAULT_PICK_ANYROTATE_SEGMENT_STEPS",
    "DEFAULT_PICK_ANYROTATE_STAGE_OBJECT_TOLERANCES",
    "DEFAULT_ROTATION_DELTA_RANGE",
    "build_pick_anyrotate_object_pose_sequence",
]
