# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-agnostic utilities for the high-level reference generation.

Two groups of shared, task-agnostic helpers:

* **Scripted object-pose trajectory primitives** -- :func:`build_object_pose_sequence_from_keyframes`
  (interpolate a waypoint sequence through an ordered list of keyframes) and the pose interpolation
  primitives (:func:`interpolate_pose_segment`, :func:`slerp`). Task-specific builders (e.g. the
  pick-insert move/align/approach/insert/hold sequence) live in that task's own ``mdps`` package and
  compose these primitives.
* **Grasp-anchor kinematics** -- :func:`hand_base_pose_from_object_command_b` (the kinematic
  counterpart to the trajectory primitives) plus the robot-tuned default offsets, mapping a target
  *object* pose to the target *hand-base* pose via a fixed grasp anchor.

The kinematics depend on ``isaaclab.utils.math`` for the frame transforms, so this module is
isaaclab-dependent (like :mod:`.gate`) and is intentionally **not** re-exported from ``__init__.py``
-- import it directly as ``from src.policy.high_level.utils import ...`` to keep the pure-torch
subset (``anchor_correction`` / ``trajectory_stepper``) loadable without isaaclab.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    subtract_frame_transforms,
)


#############
# Object-pose trajectory primitives
#############

DEFAULT_SEGMENT_STEPS = (20, 20, 10, 20, 5)


def build_object_pose_sequence_from_keyframes(
    keyframes: Sequence[torch.Tensor],
    segment_steps: Sequence[int],
) -> torch.Tensor:
    """Interpolate a waypoint sequence through an ordered list of keyframe poses.

    Task-agnostic core shared by every task's object-pose trajectory builder. ``keyframes`` is an
    ordered sequence of ``(7,)`` poses ``(x, y, z, qw, qx, qy, qz)`` (same dtype/device); each
    consecutive pair is connected by ``segment_steps[i]`` interpolation samples (position lerp +
    quaternion slerp), so ``len(segment_steps) == len(keyframes) - 1``. Returns a
    ``(1 + sum(segment_steps), 7)`` tensor: the first keyframe followed by every segment's samples.
    The number and meaning of the segments is decided entirely by the caller.
    """
    keyframes = [_with_normalized_quat(pose) for pose in keyframes]
    if len(keyframes) < 2:
        raise ValueError("keyframes must contain at least two poses.")
    if len(segment_steps) != len(keyframes) - 1:
        raise ValueError(
            "segment_steps must have one entry per consecutive keyframe pair "
            f"({len(keyframes) - 1}); got {len(segment_steps)}."
        )
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")

    samples = [keyframes[0].unsqueeze(0)]
    for start, end, steps in zip(keyframes[:-1], keyframes[1:], segment_steps):
        segment = interpolate_pose_segment(start, end, steps)
        if segment.numel() > 0:
            samples.append(segment)
    return torch.cat(samples, dim=0)


def _as_pose_tensor(
    pose: torch.Tensor | Sequence[float],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if isinstance(pose, torch.Tensor):
        tensor = pose.to(dtype=dtype, device=device) if dtype is not None or device is not None else pose
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
    else:
        tensor = torch.tensor(pose, dtype=dtype or torch.float32, device=device)
    if tensor.shape != (7,):
        raise ValueError(f"Expected pose shape (7,), got {tuple(tensor.shape)}.")
    return tensor


def _with_normalized_quat(pose: torch.Tensor) -> torch.Tensor:
    return torch.cat((pose[:3], _normalize_quat(pose[3:7])))


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quat)
    if torch.any(norm <= 0):
        raise ValueError("Quaternion norm must be positive.")
    return quat / norm


def interpolate_pose_segment(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    """Interpolate ``steps`` poses from ``start`` (exclusive) to ``end`` (inclusive).

    Positions are linearly interpolated and orientations are spherically interpolated (:func:`slerp`).
    Returns an empty ``(0, 7)`` tensor when ``steps == 0``.
    """
    if steps == 0:
        return start.new_empty((0, 7))

    segment = []
    for step in range(1, steps + 1):
        alpha = start.new_tensor(step / steps)
        pos = start[:3] + alpha * (end[:3] - start[:3])
        quat = slerp(start[3:7], end[3:7], alpha)
        segment.append(torch.cat((pos, quat)))
    return torch.stack(segment)


def slerp(start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two (w, x, y, z) quaternions."""
    start = _normalize_quat(start)
    end = _normalize_quat(end)
    dot = torch.sum(start * end)
    if dot < 0.0:
        end = -end
        dot = -dot
    dot = torch.clamp(dot, -1.0, 1.0)

    if dot > 0.9995:
        return _normalize_quat(start + alpha * (end - start))

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    start_scale = torch.sin((1.0 - alpha) * theta) / sin_theta
    end_scale = torch.sin(alpha * theta) / sin_theta
    return _normalize_quat(start_scale * start + end_scale * end)


def compose_world_yaw(quat: torch.Tensor, angle: float | torch.Tensor) -> torch.Tensor:
    """Rotate a ``(w, x, y, z)`` orientation by ``angle`` (rad) about the world/base +Z axis.

    Returns ``q_yaw ⊗ quat`` (Hamilton product, left-multiply), i.e. an *extrinsic* yaw about the
    fixed vertical Z axis applied to ``quat`` -- used to build the pick-screw twist keyframes that spin
    the (vertically standing) leg about its long/insertion axis. ``q_yaw = (cos(angle/2), 0, 0,
    sin(angle/2))``. Pure ``torch`` so the trajectory builders that call it stay isaaclab-independent.
    """
    quat = _normalize_quat(quat)
    half = quat.new_tensor(angle) * 0.5
    c, s = torch.cos(half), torch.sin(half)
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # (c, 0, 0, s) ⊗ (w, x, y, z):
    return _normalize_quat(
        torch.stack((c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w))
    )


#############
# Grasp-anchor kinematics (isaaclab.utils.math)
#############

DEFAULT_HAND_BASE_TO_ANCHOR_POSE = (0.10623648, 0.01035594, 0.07579897, 1.0, 0.0, 0.0, 0.0)
# Anchor pose offset in the robot root frame as (x, y, z, qw, qx, qy, qz).
# Position (0, 0, -0.01) is a fixed -1 cm z-offset in root; orientation (√2/2, 0, √2/2, 0) is 90° rotation about root +Y axis.
DEFAULT_OBJECT_TO_ANCHOR_POSE = (0.0, 0.0, 0.01, 0.70710678, 0.0, 0.70710678, 0.0)


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
    "DEFAULT_SEGMENT_STEPS",
    "build_object_pose_sequence_from_keyframes",
    "compose_world_yaw",
    "hand_base_pose_from_object_command_b",
    "interpolate_pose_segment",
    "slerp",
]
