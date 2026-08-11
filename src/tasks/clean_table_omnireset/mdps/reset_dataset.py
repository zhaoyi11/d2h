"""Simulator-free loading for Instant Dexterity reset-state datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


_EXPECTED_TASK = "Clean_Table_HRL-v0"
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


@dataclass(frozen=True)
class ResetStatePool:
    robot_root_pose: torch.Tensor
    robot_joint_position: torch.Tensor
    object_root_pose: torch.Tensor
    receptive_object_root_pose: torch.Tensor
    table_root_pose: torch.Tensor

    @property
    def num_states(self) -> int:
        return int(self.robot_joint_position.shape[0])

    def scene_state(self, indices: torch.Tensor) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        count = int(indices.numel())
        device = self.robot_joint_position.device
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
                "object": {
                    "root_pose": self.object_root_pose.index_select(0, indices).clone(),
                    "root_velocity": torch.zeros(count, 6, device=device),
                },
                "receptive_object": {
                    "root_pose": self.receptive_object_root_pose.index_select(0, indices).clone(),
                    "root_velocity": torch.zeros(count, 6, device=device),
                },
                "table": {
                    "root_pose": self.table_root_pose.index_select(0, indices).clone(),
                    "root_velocity": torch.zeros(count, 6, device=device),
                },
            },
        }


def _load_metadata(dataset_dir: Path, expected_joint_names: Sequence[str]) -> None:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Reset dataset metadata does not exist: {metadata_path}")
    with metadata_path.open() as file:
        metadata = json.load(file)

    if metadata.get("schema_version") != 1:
        raise ValueError("Reset dataset must use schema_version 1.")
    if metadata.get("task") != _EXPECTED_TASK:
        raise ValueError(f"Reset dataset task must be {_EXPECTED_TASK!r}.")
    if metadata.get("num_envs") != 1:
        raise ValueError("Reset dataset must have num_envs == 1 for VisDex object 104738.")

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


def _load_episode(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as episode:
        if "complete" not in episode:
            raise ValueError(f"Episode {path.name} must contain a scalar boolean complete marker.")
        complete = np.asarray(episode["complete"])
        if complete.size != 1 or not np.issubdtype(complete.dtype, np.bool_):
            raise ValueError(f"Episode {path.name} must contain a scalar boolean complete marker.")
        if "source_env_id" not in episode or np.asarray(episode["source_env_id"]).size != 1:
            raise ValueError(f"Episode {path.name} must contain a scalar source_env_id.")
        if int(np.asarray(episode["source_env_id"]).item()) != 0:
            raise ValueError(f"Episode {path.name} source_env_id must be 0 for VisDex object 104738.")

        length: int | None = None
        for field, width in _FIELD_WIDTHS.items():
            if field not in episode:
                raise ValueError(f"Episode {path.name} is missing reset field {field!r}.")
            array = np.asarray(episode[field], dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != width:
                raise ValueError(
                    f"Episode {path.name} field {field!r} must have shape (T, {width}), got {array.shape}."
                )
            if length is None:
                length = int(array.shape[0])
            elif array.shape[0] != length:
                raise ValueError(f"Episode {path.name} reset fields have inconsistent timestep counts.")
            if not np.isfinite(array).all():
                raise ValueError(f"Episode {path.name} field {field!r} must contain only finite values.")
            arrays[field] = array

    if length is None or length == 0:
        raise ValueError(f"Episode {path.name} contains no reset states.")
    return arrays


def load_reset_state_pool(
    dataset_dir: str | Path,
    expected_joint_names: Sequence[str],
    device: str | torch.device,
) -> ResetStatePool:
    dataset_path = Path(dataset_dir).expanduser().resolve()
    if not dataset_path.is_dir():
        raise ValueError(f"Reset dataset directory does not exist: {dataset_path}")
    _load_metadata(dataset_path, expected_joint_names)

    episode_paths = sorted(dataset_path.glob("*.npz"))
    if not episode_paths:
        raise ValueError(f"Reset dataset contains no NPZ episodes: {dataset_path}")
    episodes = [_load_episode(path) for path in episode_paths]

    def tensor(field: str) -> torch.Tensor:
        array = np.concatenate([episode[field] for episode in episodes], axis=0)
        return torch.as_tensor(array, dtype=torch.float32, device=device)

    return ResetStatePool(
        robot_root_pose=tensor("state.articulation.robot.root_pose"),
        robot_joint_position=tensor("state.articulation.robot.joint_position"),
        object_root_pose=tensor("state.rigid_object.object.root_pose"),
        receptive_object_root_pose=tensor("state.rigid_object.receptive_object.root_pose"),
        table_root_pose=tensor("state.rigid_object.table.root_pose"),
    )


__all__ = ["ResetStatePool", "load_reset_state_pool"]
