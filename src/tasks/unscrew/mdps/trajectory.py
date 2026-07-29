# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-Torch object-pose trajectory for unscrewing an installed table leg."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.utils import (
    _as_pose_tensor,
    _with_normalized_quat,
    build_object_pose_sequence_from_keyframes,
    compose_world_yaw,
)


DEFAULT_UNSCREW_TWIST_SEGMENTS = 12
DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE = 6.0 * math.pi
DEFAULT_UNSCREW_THREAD_PITCH = 0.015
DEFAULT_UNSCREW_EXTRACTION_HEIGHT = 0.030

# reach / establish grasp / (quarter-turn x 12) / extract / hold
DEFAULT_UNSCREW_SEGMENT_STEPS = (0, 1) + (1,) * DEFAULT_UNSCREW_TWIST_SEGMENTS + (1, 1)
DEFAULT_UNSCREW_STAGE_OBJECT_TOLERANCES = (
    StageObjTol(0.005, 0.20),  # reach while the hand is held open
    StageObjTol(0.005, 0.20),  # establish grasp at the installed pose
    *(StageObjTol(0.002, 0.35) for _ in range(DEFAULT_UNSCREW_TWIST_SEGMENTS)),
    StageObjTol(0.005, 0.20),  # vertical extraction
    StageObjTol(0.005, 0.20),  # hold
)


def build_unscrew_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    segment_steps: Sequence[int] = DEFAULT_UNSCREW_SEGMENT_STEPS,
    twist_total_angle: float = DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE,
    twist_segments: int = DEFAULT_UNSCREW_TWIST_SEGMENTS,
    thread_pitch: float = DEFAULT_UNSCREW_THREAD_PITCH,
    extraction_height: float = DEFAULT_UNSCREW_EXTRACTION_HEIGHT,
) -> torch.Tensor:
    """Build reach, grasp, positive-yaw helical rise, extraction, and hold waypoints.

    Poses are ``(x, y, z, qw, qx, qy, qz)`` in the robot base frame. The live installed object pose is
    authoritative: x/y remain fixed, positive world/base ``+Z`` yaw advances the authored thread, and
    z rises by ``thread_pitch`` per full turn. The final keyframe lifts vertically after the last turn.
    """
    if twist_segments < 1:
        raise ValueError("twist_segments must be at least 1.")
    expected_segments = twist_segments + 4
    if len(segment_steps) != expected_segments:
        raise ValueError(f"segment_steps must contain {expected_segments} values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")
    if twist_total_angle <= 0.0:
        raise ValueError("twist_total_angle must be positive for unscrewing.")
    if thread_pitch < 0.0:
        raise ValueError("thread_pitch must be non-negative.")
    if extraction_height < 0.0:
        raise ValueError("extraction_height must be non-negative.")

    delta_angle = twist_total_angle / twist_segments
    if abs(delta_angle) >= math.pi:
        raise ValueError(
            "twist_total_angle / twist_segments must be less than pi so quaternion interpolation "
            "follows the intended yaw direction."
        )

    current = _with_normalized_quat(_as_pose_tensor(current_pose))
    twist_poses: list[torch.Tensor] = []
    for index in range(1, twist_segments + 1):
        angle = index * delta_angle
        position = torch.stack(
            (
                current[0],
                current[1],
                current[2] + current.new_tensor(thread_pitch * angle / (2.0 * math.pi)),
            )
        )
        twist_poses.append(torch.cat((position, compose_world_yaw(current[3:7], angle))))

    final_twist = twist_poses[-1]
    extracted = torch.cat(
        (
            torch.stack(
                (
                    final_twist[0],
                    final_twist[1],
                    final_twist[2] + current.new_tensor(extraction_height),
                )
            ),
            final_twist[3:7],
        )
    )

    # The two leading duplicates hold the installed object still during reach and grasp establishment.
    key_poses = (current, current, current, *twist_poses, extracted, extracted)
    return build_object_pose_sequence_from_keyframes(key_poses, segment_steps)


__all__ = [
    "DEFAULT_UNSCREW_EXTRACTION_HEIGHT",
    "DEFAULT_UNSCREW_SEGMENT_STEPS",
    "DEFAULT_UNSCREW_STAGE_OBJECT_TOLERANCES",
    "DEFAULT_UNSCREW_THREAD_PITCH",
    "DEFAULT_UNSCREW_TWIST_SEGMENTS",
    "DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE",
    "build_unscrew_object_pose_sequence",
]
