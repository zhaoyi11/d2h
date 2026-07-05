"""Task-agnostic building blocks for scripted object-pose (and LEAP-hand) trajectories.

This module holds only the *generic* pieces shared by every task's trajectory builder:
:func:`build_object_pose_sequence_from_keyframes` (interpolate a waypoint sequence through an ordered
list of keyframes), the pose interpolation primitives (:func:`interpolate_pose_segment`,
:func:`slerp`), the generic anchor-pose builder, and the LEAP open->grasp builder. Task-specific
builders (e.g. the pick-insert move/align/approach/insert/hold sequence) live in that task's own
``mdps`` package and compose these primitives.

Pure ``torch`` apart from :func:`combine_frame_transforms` (used only by
:func:`build_anchor_pose_sequence`), so it can be loaded by file path and unit-tested in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import combine_frame_transforms


DEFAULT_SEGMENT_STEPS = (20, 20, 10, 20, 5)
DEFAULT_ANCHOR_POSE_B = (0.10623648, 0.01035594, 0.07579897, 1.0, 0.0, 0.0, 0.0)
# Anchor pose relative to the object-goal frame as (x, y, z, qw, qx, qy, qz).
DEFAULT_OBJECT_TO_ANCHOR_POSE = (0.0, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0)
DEFAULT_LEAP_OPEN_JOINT_POS = {f"a_{joint_id}": 0.0 for joint_id in range(16)}
DEFAULT_LEAP_GRASP_JOINT_POS = {
    "a_0": -0.250,
    "a_1": 1.000,
    "a_2": 0.750,
    "a_3": 0.150,
    "a_4": 0.000,
    "a_5": 1.000,
    "a_6": 0.700,
    "a_7": 0.150,
    "a_8": 0.250,
    "a_9": 1.000,
    "a_10": 0.750,
    "a_11": 0.150,
    "a_12": 1.250,
    "a_13": 0.000,
    "a_14": 0.150,
    "a_15": 0.050,
}


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


def build_leap_hand_joint_pose_sequence(
    joint_names: Sequence[str],
    segment_steps: Sequence[int] = DEFAULT_SEGMENT_STEPS,
    open_joint_pos: dict[str, float] = DEFAULT_LEAP_OPEN_JOINT_POS,
    grasp_joint_pos: dict[str, float] = DEFAULT_LEAP_GRASP_JOINT_POS,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build a deterministic LEAP hand open-to-grasp joint trajectory.

    Closes during the first segment (``segment_steps[0]`` samples) and holds the grasp for the
    remaining segments. Expects the demo's 5-segment convention.
    """

    _validate_segment_steps(segment_steps)
    names = tuple(joint_names)
    open_pose = _joint_pose_tensor(names, open_joint_pos, dtype=dtype, device=device)
    grasp_pose = _joint_pose_tensor(
        names,
        grasp_joint_pos,
        dtype=open_pose.dtype,
        device=open_pose.device,
    )

    samples = [open_pose]
    close_steps = segment_steps[0]
    for step in range(1, close_steps + 1):
        alpha = open_pose.new_tensor(step / close_steps)
        samples.append(open_pose + alpha * (grasp_pose - open_pose))

    hold_steps = sum(segment_steps[1:])
    samples.extend(grasp_pose.clone() for _ in range(hold_steps))
    return torch.stack(samples, dim=0)


def build_anchor_pose_sequence(
    object_pose_b: torch.Tensor | Sequence[Sequence[float]],
    anchor_pose_b: torch.Tensor | Sequence[float] = DEFAULT_ANCHOR_POSE_B,
    object_to_anchor_pose: torch.Tensor | Sequence[float] = DEFAULT_OBJECT_TO_ANCHOR_POSE,
) -> torch.Tensor:
    """Build anchor pose targets for a demo trajectory.

    ``anchor_pose_b`` is the fixed hand-base to anchor transform (kept for caller
    compatibility). The anchor is placed at ``object_pose ⊕ object_to_anchor_pose`` -- a
    relative pose expressed in the object-goal frame, so the anchor tracks the object's
    orientation.
    """

    object_pose = _as_pose_sequence_tensor(object_pose_b)
    _as_pose_tensor(
        anchor_pose_b,
        dtype=object_pose.dtype,
        device=object_pose.device,
    )
    offset = _as_pose_tensor(
        object_to_anchor_pose,
        dtype=object_pose.dtype,
        device=object_pose.device,
    ).unsqueeze(0).expand(object_pose.shape[0], -1)

    anchor_pos, anchor_quat = combine_frame_transforms(
        object_pose[:, :3], object_pose[:, 3:7], offset[:, :3], offset[:, 3:7]
    )
    return torch.cat((anchor_pos, anchor_quat), dim=1)


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


def _as_pose_sequence_tensor(pose: torch.Tensor | Sequence[Sequence[float]]) -> torch.Tensor:
    if isinstance(pose, torch.Tensor):
        tensor = pose if pose.is_floating_point() else pose.to(dtype=torch.float32)
    else:
        tensor = torch.tensor(pose, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != 7:
        raise ValueError(f"Expected pose sequence shape (N, 7), got {tuple(tensor.shape)}.")
    return tensor


def _validate_segment_steps(segment_steps: Sequence[int]) -> None:
    if len(segment_steps) != 5:
        raise ValueError("segment_steps must contain 5 values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")


def _joint_pose_tensor(
    joint_names: Sequence[str],
    joint_pos: dict[str, float],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    values = [joint_pos.get(name, 0.0) for name in joint_names]
    return torch.tensor(values, dtype=dtype or torch.float32, device=device)


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


__all__ = [
    "DEFAULT_ANCHOR_POSE_B",
    "DEFAULT_OBJECT_TO_ANCHOR_POSE",
    "DEFAULT_LEAP_GRASP_JOINT_POS",
    "DEFAULT_LEAP_OPEN_JOINT_POS",
    "DEFAULT_SEGMENT_STEPS",
    "build_anchor_pose_sequence",
    "build_leap_hand_joint_pose_sequence",
    "build_object_pose_sequence_from_keyframes",
    "interpolate_pose_segment",
    "slerp",
]
