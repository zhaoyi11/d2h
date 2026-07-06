"""Task-agnostic building blocks for scripted object-pose trajectories.

Holds only the generic pieces shared by every task's trajectory builder:
:func:`build_object_pose_sequence_from_keyframes` (interpolate a waypoint sequence through an ordered
list of keyframes) and the pose interpolation primitives (:func:`interpolate_pose_segment`,
:func:`slerp`). Task-specific builders (e.g. the pick-insert move/align/approach/insert/hold sequence)
live in that task's own ``mdps`` package and compose these primitives.

Pure ``torch`` (no isaaclab), so it can be loaded by file path and unit-tested in isolation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


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


__all__ = [
    "DEFAULT_SEGMENT_STEPS",
    "build_object_pose_sequence_from_keyframes",
    "interpolate_pose_segment",
    "slerp",
]
