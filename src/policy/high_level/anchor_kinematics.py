# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Anchor / hand-base kinematics for the high-level reference generation.

Task-agnostic geometry that maps a target *object* pose to the target *hand-base* pose via a fixed
grasp anchor: :func:`hand_base_pose_from_object_command_b` (the kinematic counterpart to
``object_trajectory.py``), plus the robot-tuned default offsets. The isaaclab ``CommandTerm`` in
``src.tasks.common.mdps.commands`` imports and drives these.

Unlike the pure-``torch`` modules in this package, this one depends on ``isaaclab.utils.math`` for the
frame transforms (like :mod:`.gate`), so it is intentionally **not** re-exported from
``__init__.py`` -- import it directly as
``from src.policy.high_level.anchor_kinematics import hand_base_pose_from_object_command_b`` to keep
the pure-torch subset loadable without isaaclab.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    subtract_frame_transforms,
)


DEFAULT_HAND_BASE_TO_ANCHOR_POSE = (0.10623648, 0.01035594, 0.07579897, 1.0, 0.0, 0.0, 0.0)
# Anchor pose offset in the robot root frame as (x, y, z, qw, qx, qy, qz).
# Position (0, 0, -0.01) is a fixed -1 cm z-offset in root; orientation (√2/2, 0, √2/2, 0) is 90° rotation about root +Y axis.
DEFAULT_OBJECT_TO_ANCHOR_POSE = (0.0, 0.0, -0.01, 0.70710678, 0.0, 0.70710678, 0.0)


def hand_base_pose_from_object_command_b(
    object_pose_b: torch.Tensor,
    hand_base_to_anchor_pose: torch.Tensor,
    object_to_anchor_pose: torch.Tensor | Sequence[float] = DEFAULT_OBJECT_TO_ANCHOR_POSE,
    anchor_correction: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute target hand-base pose from target object pose and the object->anchor offset.

    The anchor pose is computed in the *robot root frame*: position = ``object_goal_pos +
    offset_pos`` (offset_pos is a fixed root-frame offset), and orientation = ``offset_quat``
    (fixed root-aligned grasp orientation, decoupled from the object goal). The anchor orientation
    is thus constant along the trajectory while the object goal reorients. The hand base is then
    recovered by composing the inverse of the fixed hand-base->anchor transform. ``object_to_anchor_pose``
    may be a single ``(7,)`` offset (broadcast to all envs) or a per-env ``(N, 7)`` offset.
    ``anchor_correction`` is an optional per-env ``(N, 6)`` bounded delta (pos[3], axis-angle[3])
    applied to the nominal anchor pose *in the anchor frame*.

    Returns:
        Tuple of (hand_base_pose, anchor_pose), each shape (N, 7) in robot root frame.
    """
    if hand_base_to_anchor_pose.ndim == 1:
        hand_base_to_anchor_pose = hand_base_to_anchor_pose.unsqueeze(0).repeat(object_pose_b.shape[0], 1)
    hand_base_to_anchor_pose = hand_base_to_anchor_pose.to(dtype=object_pose_b.dtype, device=object_pose_b.device)
    offset = torch.as_tensor(object_to_anchor_pose, dtype=object_pose_b.dtype, device=object_pose_b.device)
    if offset.ndim == 1:
        offset = offset.unsqueeze(0).repeat(object_pose_b.shape[0], 1)

    # Nominal anchor: both position and orientation are expressed in the robot root frame.
    # Position = object_goal_pos + offset_pos (offset_pos is a fixed root-frame z-offset).
    # Orientation = offset_quat (root-aligned, fixed grasp orientation).
    target_anchor_pos_b = object_pose_b[:, :3] + offset[:, :3]
    target_anchor_quat = offset[:, 3:7]

    # Bounded correction applied *in the anchor frame*. The anchor orientation is root-aligned, so
    # apply_delta_pose (position added directly, orientation left-multiplied) realises an
    # anchor-frame == root-frame delta on the anchor pose.
    if anchor_correction is not None:
        target_anchor_pos_b, target_anchor_quat = apply_delta_pose(
            target_anchor_pos_b, target_anchor_quat, anchor_correction
        )

    anchor_to_hand_base_pos, anchor_to_hand_base_quat = subtract_frame_transforms(
        hand_base_to_anchor_pose[:, :3],
        hand_base_to_anchor_pose[:, 3:7],
    )
    hand_base_pos_b, hand_base_quat_b = combine_frame_transforms(
        target_anchor_pos_b,
        target_anchor_quat,
        anchor_to_hand_base_pos,
        anchor_to_hand_base_quat,
    )
    hand_base_pose = torch.cat((hand_base_pos_b, hand_base_quat_b), dim=1)
    anchor_pose = torch.cat((target_anchor_pos_b, target_anchor_quat), dim=1)
    return hand_base_pose, anchor_pose


__all__ = [
    "DEFAULT_HAND_BASE_TO_ANCHOR_POSE",
    "DEFAULT_OBJECT_TO_ANCHOR_POSE",
    "hand_base_pose_from_object_command_b",
]
