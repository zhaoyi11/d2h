# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-insert object-pose trajectory builder.

The scripted move->align->approach->insert->hold peg trajectory that is specific to the pick-insert
task, composed from the task-agnostic primitives in :mod:`src.tasks.common.mdps.object_trajectory`.
Pure ``torch`` so it can be unit-tested in isolation; the command term that consumes it lives in
:mod:`src.tasks.pick_insert.mdps.commands`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.tasks.common.mdps.object_trajectory import (
    DEFAULT_SEGMENT_STEPS,
    _as_pose_tensor,
    _with_normalized_quat,
    build_object_pose_sequence_from_keyframes,
)


DEFAULT_RECEPTIVE_POSE = (0.35, 0.0, 0.285, 1.0, 0.0, 0.0, 0.0)


def build_pick_insert_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    receptive_pose: torch.Tensor | Sequence[float] = DEFAULT_RECEPTIVE_POSE,
    segment_steps: Sequence[int] = DEFAULT_SEGMENT_STEPS,
    above_offset: float = 0.15,
    insertion_depth: float = 0.015,
    approach_height: float = 0.08,
) -> torch.Tensor:
    """Build an interpolated peg-insertion object-pose trajectory.

    Poses use ``(x, y, z, qw, qx, qy, qz)`` in the robot base frame. The returned sequence starts at
    ``current_pose``, moves above the receptacle, aligns to the receptacle orientation, descends to
    the inserted pose, and holds there -- 5 segments: move / align / approach / insert / hold. The
    keyframes are interpolated by the shared
    :func:`~src.tasks.common.mdps.object_trajectory.build_object_pose_sequence_from_keyframes` core.
    """
    if len(segment_steps) != 5:
        raise ValueError("segment_steps must contain 5 values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")

    current = _with_normalized_quat(_as_pose_tensor(current_pose))
    receptive = _with_normalized_quat(
        _as_pose_tensor(receptive_pose, dtype=current.dtype, device=current.device)
    )

    above_current = torch.cat(
        (
            torch.stack(
                (
                    receptive[0],
                    receptive[1],
                    receptive[2] + current.new_tensor(above_offset),
                )
            ),
            current[3:7],
        )
    )
    above_aligned = torch.cat((above_current[:3], receptive[3:7]))
    approach = torch.cat(
        (
            torch.stack(
                (
                    receptive[0],
                    receptive[1],
                    receptive[2] + current.new_tensor(insertion_depth + approach_height),
                )
            ),
            receptive[3:7],
        )
    )
    inserted = torch.cat(
        (
            torch.stack(
                (
                    receptive[0],
                    receptive[1],
                    receptive[2] + current.new_tensor(insertion_depth),
                )
            ),
            receptive[3:7],
        )
    )

    key_poses = (current, above_current, above_aligned, approach, inserted, inserted)
    return build_object_pose_sequence_from_keyframes(key_poses, segment_steps)


__all__ = [
    "DEFAULT_RECEPTIVE_POSE",
    "build_pick_insert_object_pose_sequence",
]
