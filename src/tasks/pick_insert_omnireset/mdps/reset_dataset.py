"""Simulator-free loading for one Instant Dexterity reset-state archive."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from .geometry import (
    RectangularInsertionGeometry,
    unfinished_state_indices,
)


_EXPECTED_TASK = "Pick_Insert_HRL-v0"
_FIELD_WIDTHS = {
    "state.articulation.robot.root_pose": 7,
    "state.articulation.robot.root_velocity": 6,
    "state.articulation.robot.joint_position": 23,
    "state.articulation.robot.joint_velocity": 23,
    "state.rigid_object.object.root_pose": 7,
    "state.rigid_object.object.root_velocity": 6,
    "state.rigid_object.receptive_object.root_pose": 7,
    "state.rigid_object.receptive_object.root_velocity": 6,
    "state.rigid_object.static_obstacle.root_pose": 7,
    "state.rigid_object.static_obstacle.root_velocity": 6,
    "state.rigid_object.table.root_pose": 7,
    "state.rigid_object.table.root_velocity": 6,
}


@dataclass(frozen=True)
class ResetStatePool:
    robot_root_pose: torch.Tensor
    robot_joint_position: torch.Tensor
    object_root_pose: torch.Tensor
    receptive_object_root_pose: torch.Tensor
    static_obstacle_root_pose: torch.Tensor
    table_root_pose: torch.Tensor

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
                "static_obstacle": rigid_state(self.static_obstacle_root_pose),
                "table": rigid_state(self.table_root_pose),
            },
        }


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
        if complete.size != 1 or not np.issubdtype(complete.dtype, np.bool_):
            raise ValueError(f"Episode {path.name} must contain a scalar boolean complete marker.")
        if "source_env_id" not in episode or np.asarray(episode["source_env_id"]).size != 1:
            raise ValueError(f"Episode {path.name} must contain a scalar source_env_id.")
        if int(np.asarray(episode["source_env_id"]).item()) != 0:
            raise ValueError(f"Episode {path.name} source_env_id must be 0.")

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
    dataset_path: str | Path,
    expected_joint_names: Sequence[str],
    device: str | torch.device,
    geometry: RectangularInsertionGeometry,
) -> ResetStatePool:
    archive_path = Path(dataset_path).expanduser().resolve()
    if not archive_path.is_file():
        raise ValueError(f"Reset dataset archive does not exist: {archive_path}")
    _load_metadata(archive_path.parent / "metadata.json", expected_joint_names)
    episode = _load_archive(archive_path)

    def tensor(field: str) -> torch.Tensor:
        return torch.as_tensor(episode[field], dtype=torch.float32, device=device)

    pool = ResetStatePool(
        robot_root_pose=tensor("state.articulation.robot.root_pose"),
        robot_joint_position=tensor("state.articulation.robot.joint_position"),
        object_root_pose=tensor("state.rigid_object.object.root_pose"),
        receptive_object_root_pose=tensor("state.rigid_object.receptive_object.root_pose"),
        static_obstacle_root_pose=tensor("state.rigid_object.static_obstacle.root_pose"),
        table_root_pose=tensor("state.rigid_object.table.root_pose"),
    )
    indices = unfinished_state_indices(
        pool.object_root_pose,
        pool.receptive_object_root_pose,
        geometry,
    )
    return ResetStatePool(
        robot_root_pose=pool.robot_root_pose.index_select(0, indices),
        robot_joint_position=pool.robot_joint_position.index_select(0, indices),
        object_root_pose=pool.object_root_pose.index_select(0, indices),
        receptive_object_root_pose=pool.receptive_object_root_pose.index_select(0, indices),
        static_obstacle_root_pose=pool.static_obstacle_root_pose.index_select(0, indices),
        table_root_pose=pool.table_root_pose.index_select(0, indices),
    )


__all__ = ["ResetStatePool", "load_reset_state_pool"]
