"""Simulator-free loading of rotate-object OmniReset states and goals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


_EXPECTED_TASK = "Rotate_Object_Once_HRL-v0"
_LOW_LEVEL_DIM = 155
_BC_DIM = 176
_BC_ORDER = [
    "low_level",
    "arm_joint_pos",
    "arm_joint_vel",
    "hand_base_command",
]
_FIELD_WIDTHS = {
    "state.articulation.robot.root_pose": 7,
    "state.articulation.robot.root_velocity": 6,
    "state.articulation.robot.joint_position": 23,
    "state.articulation.robot.joint_velocity": 23,
    "state.rigid_object.object.root_pose": 7,
    "state.rigid_object.object.root_velocity": 6,
    "state.rigid_object.receptive_object.root_pose": 7,
    "state.rigid_object.receptive_object.root_velocity": 6,
    "state.rigid_object.table.root_pose": 7,
    "state.rigid_object.table.root_velocity": 6,
}
_OBSERVATION_WIDTHS = {
    "observation.low_level": _LOW_LEVEL_DIM,
    "observation.hand_base_command": 7,
}
_GOAL_CHANGE_THRESHOLD = 0.5
_SUCCESS_TOLERANCE = 0.2


@dataclass(frozen=True)
class ResetStatePool:
    robot_root_pose: torch.Tensor
    robot_joint_position: torch.Tensor
    object_root_pose: torch.Tensor
    receptive_object_root_pose: torch.Tensor
    table_root_pose: torch.Tensor
    goal_pose_b: torch.Tensor
    hand_base_command_b: torch.Tensor
    progress: torch.Tensor

    @property
    def num_states(self) -> int:
        return int(self.robot_joint_position.shape[0])

    def scene_state(self, indices: torch.Tensor) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        count = int(indices.numel())
        device = self.robot_joint_position.device

        def rigid_state(root_pose: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "root_pose": root_pose.index_select(0, indices).clone(),
                "root_velocity": torch.zeros(count, 6, device=device),
            }

        return {
            "articulation": {
                "robot": {
                    "root_pose": self.robot_root_pose.index_select(0, indices).clone(),
                    "root_velocity": torch.zeros(count, 6, device=device),
                    "joint_position": self.robot_joint_position.index_select(0, indices).clone(),
                    "joint_velocity": torch.zeros(count, 23, device=device),
                }
            },
            "rigid_object": {
                "object": rigid_state(self.object_root_pose),
                "receptive_object": rigid_state(self.receptive_object_root_pose),
                "table": rigid_state(self.table_root_pose),
            },
        }


def _quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion * quaternion.new_tensor((1.0, -1.0, -1.0, -1.0))


def _quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _quat_angle(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(first * second, dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure_vector = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_multiply(
        _quat_multiply(_quat_conjugate(quaternion), pure_vector),
        quaternion,
    )[..., 1:]


def _load_metadata(metadata_path: Path, expected_joint_names: Sequence[str]) -> None:
    if not metadata_path.is_file():
        raise ValueError(f"Reset dataset metadata does not exist: {metadata_path}")
    with metadata_path.open() as file:
        metadata = json.load(file)

    if metadata.get("schema_version") != 1:
        raise ValueError("Reset dataset must use schema_version 1.")
    if metadata.get("task") != _EXPECTED_TASK:
        raise ValueError(f"Reset dataset task must be {_EXPECTED_TASK!r}.")
    if metadata.get("num_envs") != 1:
        raise ValueError("Reset dataset must have num_envs == 1.")

    observation = metadata.get("observation", {})
    if observation.get("low_level_dim") != _LOW_LEVEL_DIM:
        raise ValueError(f"Reset dataset low_level_dim must be {_LOW_LEVEL_DIM}.")
    if observation.get("bc_dim") != _BC_DIM:
        raise ValueError(f"Reset dataset bc_dim must be {_BC_DIM}.")
    if observation.get("bc_order") != _BC_ORDER:
        raise ValueError(f"Reset dataset bc_order must be {_BC_ORDER}.")

    scene_state = metadata.get("scene_state", {})
    if scene_state.get("relative_to_env_origin") is not True:
        raise ValueError("Reset dataset scene states must be relative to environment origins.")
    fields = set(scene_state.get("fields", []))
    if fields != set(_FIELD_WIDTHS):
        missing = sorted(set(_FIELD_WIDTHS) - fields)
        extra = sorted(fields - set(_FIELD_WIDTHS))
        raise ValueError(f"Reset dataset scene fields mismatch: missing={missing}, extra={extra}.")

    joint_order = metadata.get("joint_order", {}).get("articulations", {}).get("robot")
    if joint_order != list(expected_joint_names):
        raise ValueError("Reset dataset robot joint order does not match the environment articulation.")


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as episode:
        if "complete" not in episode:
            raise ValueError(f"Episode {path.name} must contain a scalar boolean complete marker.")
        complete = np.asarray(episode["complete"])
        if complete.size != 1 or not np.issubdtype(complete.dtype, np.bool_) or not bool(complete.item()):
            raise ValueError(f"Episode {path.name} complete marker must be true.")
        if "source_env_id" not in episode or np.asarray(episode["source_env_id"]).size != 1:
            raise ValueError(f"Episode {path.name} must contain a scalar source_env_id.")
        if int(np.asarray(episode["source_env_id"]).item()) != 0:
            raise ValueError(f"Episode {path.name} source_env_id must be 0.")

        length: int | None = None
        for field, width in (_FIELD_WIDTHS | _OBSERVATION_WIDTHS).items():
            if field not in episode:
                raise ValueError(f"Episode {path.name} is missing field {field!r}.")
            array = np.asarray(episode[field], dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != width:
                raise ValueError(
                    f"Episode {path.name} field {field!r} must have shape (T, {width}), got {array.shape}."
                )
            if length is None:
                length = int(array.shape[0])
            elif array.shape[0] != length:
                raise ValueError(f"Episode {path.name} fields have inconsistent timestep counts.")
            if not np.isfinite(array).all():
                raise ValueError(f"Episode {path.name} field {field!r} must contain only finite values.")
            arrays[field] = array

    if length is None or length == 0:
        raise ValueError(f"Episode {path.name} contains no reset states.")
    return arrays


def _episode_pool(episode: dict[str, np.ndarray], device: str | torch.device) -> ResetStatePool | None:
    def tensor(field: str) -> torch.Tensor:
        return torch.as_tensor(episode[field], dtype=torch.float32, device=device)

    robot_root_pose = tensor("state.articulation.robot.root_pose")
    object_root_pose = tensor("state.rigid_object.object.root_pose")
    low_level = tensor("observation.low_level")
    hand_base_command = tensor("observation.hand_base_command")

    current_quat_h = _normalize_quaternion(low_level[:, 119:123])
    goal_to_current_quat_h = _normalize_quaternion(low_level[:, 135:139])
    goal_quat_h = _normalize_quaternion(
        _quat_multiply(_quat_conjugate(goal_to_current_quat_h), current_quat_h)
    )
    goal_quat_b = _normalize_quaternion(
        _quat_multiply(_normalize_quaternion(hand_base_command[:, 3:7]), goal_quat_h)
    )
    robot_quat = _normalize_quaternion(robot_root_pose[:, 3:7])
    current_quat_b = _normalize_quaternion(
        _quat_multiply(_quat_conjugate(robot_quat), _normalize_quaternion(object_root_pose[:, 3:7]))
    )
    goal_pos_b = _quat_apply_inverse(
        robot_quat,
        object_root_pose[:, :3] - robot_root_pose[:, :3],
    )
    goal_pose_b = torch.cat((goal_pos_b, goal_quat_b), dim=-1)

    target_changed = _quat_angle(goal_quat_b[1:], goal_quat_b[:-1]) > _GOAL_CHANGE_THRESHOLD
    starts = [0, *(target_changed.nonzero().flatten().add(1).tolist())]
    ends = [*starts[1:], goal_pose_b.shape[0]]
    error = _quat_angle(current_quat_b, goal_quat_b)

    retained_indices: list[torch.Tensor] = []
    retained_progress: list[torch.Tensor] = []
    for start, end in zip(starts, ends, strict=True):
        completed = end < goal_pose_b.shape[0] or bool(error[end - 1] <= _SUCCESS_TOLERANCE)
        indices = torch.arange(start, end, device=device)[error[start:end] > _SUCCESS_TOLERANCE]
        if not completed or indices.numel() == 0:
            continue
        retained_indices.append(indices)
        if indices.numel() == 1:
            retained_progress.append(torch.ones(1, device=device))
        else:
            retained_progress.append(torch.linspace(0.0, 1.0, indices.numel(), device=device))

    if not retained_indices:
        return None
    indices = torch.cat(retained_indices)

    return ResetStatePool(
        robot_root_pose=robot_root_pose.index_select(0, indices),
        robot_joint_position=tensor("state.articulation.robot.joint_position").index_select(0, indices),
        object_root_pose=object_root_pose.index_select(0, indices),
        receptive_object_root_pose=tensor(
            "state.rigid_object.receptive_object.root_pose"
        ).index_select(0, indices),
        table_root_pose=tensor("state.rigid_object.table.root_pose").index_select(0, indices),
        goal_pose_b=goal_pose_b.index_select(0, indices),
        hand_base_command_b=hand_base_command.index_select(0, indices),
        progress=torch.cat(retained_progress),
    )


def _sample_hardest_state_indices(
    progress: torch.Tensor,
    count: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample uniformly from the earliest retained state of each recorded goal."""
    hardest_indices = (progress == 0.0).nonzero().flatten().to(device)
    if hardest_indices.numel() == 0:
        raise ValueError("Reset dataset contains no zero-progress states.")
    positions = torch.randint(hardest_indices.numel(), (count,), device=device)
    return hardest_indices.index_select(0, positions)


def _sample_curriculum_state_indices(
    progress: torch.Tensor,
    count: int,
    stage: int,
    num_stages: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample states from the active per-goal progress suffix, biased toward later states."""
    active_start = 1.0 - (stage + 1) / num_stages
    active_indices = (progress >= active_start).nonzero().flatten()
    active_progress = progress.index_select(0, active_indices)
    weights = 1.0 + 3.0 * (active_progress - active_start) / max(1.0 - active_start, 1.0e-8)
    sampled_positions = torch.multinomial(weights.to(device), count, replacement=True)
    return active_indices.to(device).index_select(0, sampled_positions)


def load_reset_state_pool(
    dataset_path: str | Path,
    expected_joint_names: Sequence[str],
    device: str | torch.device,
) -> ResetStatePool:
    dataset_dir = Path(dataset_path).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise ValueError(f"Reset dataset directory does not exist: {dataset_dir}")
    _load_metadata(dataset_dir / "metadata.json", expected_joint_names)

    pools = []
    for archive_path in sorted(dataset_dir.glob("*.npz")):
        pool = _episode_pool(_load_archive(archive_path), device)
        if pool is not None:
            pools.append(pool)
    if not pools:
        raise ValueError(f"Reset dataset contains no unfinished states from completed goals: {dataset_dir}")

    return ResetStatePool(
        robot_root_pose=torch.cat([pool.robot_root_pose for pool in pools]),
        robot_joint_position=torch.cat([pool.robot_joint_position for pool in pools]),
        object_root_pose=torch.cat([pool.object_root_pose for pool in pools]),
        receptive_object_root_pose=torch.cat([pool.receptive_object_root_pose for pool in pools]),
        table_root_pose=torch.cat([pool.table_root_pose for pool in pools]),
        goal_pose_b=torch.cat([pool.goal_pose_b for pool in pools]),
        hand_base_command_b=torch.cat([pool.hand_base_command_b for pool in pools]),
        progress=torch.cat([pool.progress for pool in pools]),
    )


__all__ = ["ResetStatePool", "load_reset_state_pool"]
