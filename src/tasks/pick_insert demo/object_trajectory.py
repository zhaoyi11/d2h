from __future__ import annotations

from collections.abc import Sequence

import torch


DEFAULT_RECEPTIVE_POSE = (0.35, 0.0, 0.285, 1.0, 0.0, 0.0, 0.0)
DEFAULT_SEGMENT_STEPS = (20, 20, 10, 20, 5)
DEFAULT_ANCHOR_POSE_B = (0.10623648, 0.01035594, 0.07579897, 1.0, 0.0, 0.0, 0.0)
DEFAULT_ANCHOR_CLEARANCE = 0.02
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


def build_pick_insert_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    receptive_pose: torch.Tensor | Sequence[float] = DEFAULT_RECEPTIVE_POSE,
    segment_steps: Sequence[int] = DEFAULT_SEGMENT_STEPS,
    above_offset: float = 0.15,
    insertion_depth: float = 0.015,
    approach_height: float = 0.08,
) -> torch.Tensor:
    """Build an interpolated peg insertion object pose trajectory.

    Poses use ``(x, y, z, qw, qx, qy, qz)`` in the robot base frame. The returned
    sequence starts at ``current_pose``, moves above the receptacle, aligns to the
    receptacle orientation, descends to the inserted pose, and holds there.
    """

    if len(segment_steps) != 5:
        raise ValueError("segment_steps must contain 5 values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")

    current = _as_pose_tensor(current_pose)
    receptive = _as_pose_tensor(
        receptive_pose,
        dtype=current.dtype,
        device=current.device,
    )

    current = _with_normalized_quat(current)
    receptive = _with_normalized_quat(receptive)

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
    samples = [current.unsqueeze(0)]
    for start, end, steps in zip(key_poses[:-1], key_poses[1:], segment_steps):
        segment = _interpolate_pose_segment(start, end, steps)
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
    """Build a deterministic LEAP hand open-to-grasp joint trajectory."""

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
    anchor_clearance: float = DEFAULT_ANCHOR_CLEARANCE,
) -> torch.Tensor:
    """Build anchor pose targets for the demo.

    ``anchor_pose_b`` supplies the anchor-frame orientation. The anchor position is
    placed above the object pose by ``anchor_clearance``.
    """

    object_pose = _as_pose_sequence_tensor(object_pose_b)
    anchor_pose_cfg = _as_pose_tensor(
        anchor_pose_b,
        dtype=object_pose.dtype,
        device=object_pose.device,
    )
    anchor_pose_cfg = _with_normalized_quat(anchor_pose_cfg)

    anchor_pose = object_pose.clone()
    anchor_pose[:, 2] += object_pose.new_tensor(anchor_clearance)
    anchor_pose[:, 3:7] = anchor_pose_cfg[3:7]
    return anchor_pose


def build_pick_insert_demo_motion(
    current_pose: torch.Tensor | Sequence[float],
    joint_names: Sequence[str],
    receptive_pose: torch.Tensor | Sequence[float] = DEFAULT_RECEPTIVE_POSE,
    segment_steps: Sequence[int] = DEFAULT_SEGMENT_STEPS,
    above_offset: float = 0.15,
    insertion_depth: float = 0.015,
    approach_height: float = 0.08,
    anchor_pose_b: torch.Tensor | Sequence[float] = DEFAULT_ANCHOR_POSE_B,
    anchor_clearance: float = DEFAULT_ANCHOR_CLEARANCE,
) -> dict[str, torch.Tensor]:
    """Build object, LEAP hand, and anchor-frame motion for the pick-insert demo."""

    object_pose_b = build_pick_insert_object_pose_sequence(
        current_pose,
        receptive_pose=receptive_pose,
        segment_steps=segment_steps,
        above_offset=above_offset,
        insertion_depth=insertion_depth,
        approach_height=approach_height,
    )
    leap_joint_pos = build_leap_hand_joint_pose_sequence(
        joint_names,
        segment_steps=segment_steps,
        dtype=object_pose_b.dtype,
        device=object_pose_b.device,
    )
    anchor_pose_sequence_b = build_anchor_pose_sequence(
        object_pose_b,
        anchor_pose_b=anchor_pose_b,
        anchor_clearance=anchor_clearance,
    )
    return {
        "object_pose_b": object_pose_b,
        "leap_joint_pos": leap_joint_pos,
        "anchor_pose_b": anchor_pose_sequence_b,
    }


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


def _interpolate_pose_segment(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    if steps == 0:
        return start.new_empty((0, 7))

    segment = []
    for step in range(1, steps + 1):
        alpha = start.new_tensor(step / steps)
        pos = start[:3] + alpha * (end[:3] - start[:3])
        quat = _slerp(start[3:7], end[3:7], alpha)
        segment.append(torch.cat((pos, quat)))
    return torch.stack(segment)


def _slerp(start: torch.Tensor, end: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
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
    "DEFAULT_ANCHOR_CLEARANCE",
    "DEFAULT_ANCHOR_POSE_B",
    "DEFAULT_LEAP_GRASP_JOINT_POS",
    "DEFAULT_LEAP_OPEN_JOINT_POS",
    "DEFAULT_RECEPTIVE_POSE",
    "DEFAULT_SEGMENT_STEPS",
    "build_anchor_pose_sequence",
    "build_leap_hand_joint_pose_sequence",
    "build_pick_insert_demo_motion",
    "build_pick_insert_object_pose_sequence",
]
