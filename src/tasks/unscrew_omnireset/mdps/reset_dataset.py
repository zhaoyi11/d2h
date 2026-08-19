"""Simulator-free loading for unscrew Instant Dexterity reset datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from src.tasks.unscrew.geometry import (
    bolt_aabb_corners,
    bolt_bottom_clearance,
    relative_pose,
)


_EXPECTED_TASK = "Unscrew_HRL-v0"
_SUCCESS_CLEARANCE = 0.030
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
    progress: torch.Tensor

    @property
    def num_states(self) -> int:
        return int(self.robot_joint_position.shape[0])

    def scene_state(
        self, indices: torch.Tensor
    ) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
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
                    "joint_position": self.robot_joint_position.index_select(
                        0, indices
                    ).clone(),
                    "joint_velocity": torch.zeros(count, 23, device=device),
                }
            },
            "rigid_object": {
                "object": rigid_state(self.object_root_pose),
                "receptive_object": rigid_state(self.receptive_object_root_pose),
                "table": rigid_state(self.table_root_pose),
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
        raise ValueError("Reset dataset must have num_envs == 1.")

    scene_state = metadata.get("scene_state", {})
    if scene_state.get("relative_to_env_origin") is not True:
        raise ValueError("Reset dataset scene states must be relative to environment origins.")
    fields = set(scene_state.get("fields", []))
    if fields != set(_FIELD_WIDTHS):
        missing = sorted(set(_FIELD_WIDTHS) - fields)
        extra = sorted(fields - set(_FIELD_WIDTHS))
        raise ValueError(
            f"Reset dataset scene fields mismatch: missing={missing}, extra={extra}."
        )

    joint_order = metadata.get("joint_order", {}).get("articulations", {}).get("robot")
    if joint_order != list(expected_joint_names):
        raise ValueError(
            "Reset dataset robot joint order does not match the environment articulation."
        )


def _load_episode(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as episode:
        if "complete" not in episode:
            raise ValueError(
                f"Episode {path.name} must contain a scalar boolean complete marker."
            )
        complete = np.asarray(episode["complete"])
        if complete.size != 1 or not np.issubdtype(complete.dtype, np.bool_):
            raise ValueError(
                f"Episode {path.name} must contain a scalar boolean complete marker."
            )
        if not bool(complete.item()):
            raise ValueError(f"Episode {path.name} complete marker must be true.")
        if "source_env_id" not in episode or np.asarray(
            episode["source_env_id"]
        ).size != 1:
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
                    f"Episode {path.name} field {field!r} must have shape "
                    f"(T, {width}), got {array.shape}."
                )
            if length is None:
                length = int(array.shape[0])
            elif array.shape[0] != length:
                raise ValueError(
                    f"Episode {path.name} reset fields have inconsistent timestep counts."
                )
            if not np.isfinite(array).all():
                raise ValueError(
                    f"Episode {path.name} field {field!r} must contain only finite values."
                )
            arrays[field] = array

    if length is None or length == 0:
        raise ValueError(f"Episode {path.name} contains no reset states.")
    return arrays


def _unfinished_state_indices(
    episode: dict[str, torch.Tensor], device: str | torch.device
) -> torch.Tensor:
    object_pose = episode["state.rigid_object.object.root_pose"]
    receptive_pose = episode["state.rigid_object.receptive_object.root_pose"]
    object_pos_r, object_quat_r = relative_pose(receptive_pose, object_pose)
    clearance = bolt_bottom_clearance(
        object_pos_r,
        object_quat_r,
        bolt_aabb_corners(device),
    )
    return torch.nonzero(clearance < _SUCCESS_CLEARANCE, as_tuple=False).flatten()


def _sample_curriculum_state_indices(
    progress: torch.Tensor,
    count: int,
    stage: int,
    num_stages: int,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample later-weighted states from the active per-episode progress band."""
    if num_stages < 1:
        raise ValueError("num_stages must be at least 1.")
    if stage < 0 or stage >= num_stages:
        raise ValueError("stage must be in [0, num_stages).")

    progress = progress.to(device=device)
    active_fraction = (stage + 1) / num_stages
    active_start = 1.0 - active_fraction
    active_indices = torch.nonzero(
        progress >= active_start - torch.finfo(progress.dtype).eps,
        as_tuple=False,
    ).flatten()
    if active_indices.numel() == 0:
        raise ValueError("Reset curriculum has no states in its active progress band.")
    active_progress = progress.index_select(0, active_indices)
    band_progress = ((active_progress - active_start) / active_fraction).clamp(0.0, 1.0)
    weights = 1.0 + 3.0 * band_progress
    sampled_positions = torch.multinomial(weights, count, replacement=True)
    return active_indices.index_select(0, sampled_positions)


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

    retained: list[dict[str, torch.Tensor]] = []
    progress_parts: list[torch.Tensor] = []
    for path in episode_paths:
        arrays = _load_episode(path)
        episode = {
            field: torch.as_tensor(array, dtype=torch.float32, device=device)
            for field, array in arrays.items()
        }
        indices = _unfinished_state_indices(episode, device)
        if indices.numel() == 0:
            continue
        retained.append(
            {field: value.index_select(0, indices) for field, value in episode.items()}
        )
        count = int(indices.numel())
        progress_parts.append(
            torch.ones(1, device=device)
            if count == 1
            else torch.linspace(0.0, 1.0, count, device=device)
        )

    if not retained:
        raise ValueError("Reset dataset contains no unfinished unscrew states.")

    def tensor(field: str) -> torch.Tensor:
        return torch.cat([episode[field] for episode in retained], dim=0)

    return ResetStatePool(
        robot_root_pose=tensor("state.articulation.robot.root_pose"),
        robot_joint_position=tensor("state.articulation.robot.joint_position"),
        object_root_pose=tensor("state.rigid_object.object.root_pose"),
        receptive_object_root_pose=tensor(
            "state.rigid_object.receptive_object.root_pose"
        ),
        table_root_pose=tensor("state.rigid_object.table.root_pose"),
        progress=torch.cat(progress_parts),
    )


__all__ = ["ResetStatePool", "load_reset_state_pool"]
