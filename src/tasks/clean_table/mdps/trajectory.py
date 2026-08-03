# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clean-table reach, pick, deposit, release, and retreat trajectory."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.utils.math import quat_apply

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.utils import (
    _as_pose_tensor,
    _with_normalized_quat,
    build_object_pose_sequence_from_keyframes,
)


# reach / establish grasp / lift / carry / deposit / release / retreat
DEFAULT_CLEAN_TABLE_SEGMENT_STEPS = (0, 1, 1, 2, 2, 12, 2)
DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES = (
    StageObjTol(0.02, 0.30),
    StageObjTol(0.02, 0.30),
    StageObjTol(0.03, 0.30),
    StageObjTol(0.04, 0.50),
    StageObjTol(0.04, 0.60),
    StageObjTol(0.04, 0.60),
    StageObjTol(0.04, 0.60),
)

DEFAULT_BOX_TARGET_OFFSET = (0.0, 0.0, 0.065)
DEFAULT_BOX_RETREAT_OFFSET = (0.0, 0.0, 0.25)


def _box_local_position_b(box_pose: torch.Tensor, local_offset: Sequence[float]) -> torch.Tensor:
    offset = box_pose.new_tensor(local_offset)
    if offset.shape != (3,):
        raise ValueError(f"Expected box-local offset shape (3,), got {tuple(offset.shape)}.")
    return box_pose[:3] + quat_apply(box_pose[3:7].unsqueeze(0), offset.unsqueeze(0))[0]


def build_clean_table_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    box_pose: torch.Tensor | Sequence[float],
    segment_steps: Sequence[int] = DEFAULT_CLEAN_TABLE_SEGMENT_STEPS,
    lift_height: float = 0.08,
    box_target_offset: Sequence[float] = DEFAULT_BOX_TARGET_OFFSET,
    retreat_offset: Sequence[float] = DEFAULT_BOX_RETREAT_OFFSET,
) -> torch.Tensor:
    """Build the object-goal trajectory in the robot base frame.

    The object keeps its settled orientation. The release keyframe is repeated so
    the gate can open the hand before the final virtual object target raises the
    now-empty hand out of the box.
    """
    if len(segment_steps) != 7:
        raise ValueError("segment_steps must contain 7 values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")
    if segment_steps[5] < 1:
        raise ValueError("the release segment must contain at least one step.")
    if lift_height < 0.0:
        raise ValueError("lift_height must be non-negative.")

    current = _with_normalized_quat(_as_pose_tensor(current_pose))
    box = _with_normalized_quat(
        _as_pose_tensor(box_pose, dtype=current.dtype, device=current.device)
    )

    lifted = current.clone()
    lifted[2] += current.new_tensor(lift_height)

    deposit_pos = _box_local_position_b(box, box_target_offset)
    retreat_pos = _box_local_position_b(box, retreat_offset)
    above_box = torch.cat((retreat_pos, current[3:7]))
    deposited = torch.cat((deposit_pos, current[3:7]))
    retreated = torch.cat((retreat_pos, current[3:7]))

    keyframes = (
        current,
        current,
        current,
        lifted,
        above_box,
        deposited,
        deposited,
        retreated,
    )
    return build_object_pose_sequence_from_keyframes(keyframes, segment_steps)


__all__ = [
    "DEFAULT_BOX_RETREAT_OFFSET",
    "DEFAULT_BOX_TARGET_OFFSET",
    "DEFAULT_CLEAN_TABLE_SEGMENT_STEPS",
    "DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES",
    "build_clean_table_object_pose_sequence",
]
