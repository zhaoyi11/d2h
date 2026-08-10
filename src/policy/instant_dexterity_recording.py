"""Recording helpers for student-policy demonstrations from instant dexterity."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _check_batched_tensors(named_tensors: Mapping[str, torch.Tensor]) -> None:
    batch_size: int | None = None
    for name, tensor in named_tensors.items():
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a batched 2D tensor, got shape {tuple(tensor.shape)}.")
        if batch_size is None:
            batch_size = tensor.shape[0]
        elif tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch size {tensor.shape[0]} does not match {batch_size}.")


def build_bc_observation(
    low_level_obs: torch.Tensor,
    arm_joint_pos: torch.Tensor,
    arm_joint_vel: torch.Tensor,
    hand_base_command: torch.Tensor,
) -> torch.Tensor:
    """Concatenate the teacher observation with arm state and cuRobo's pose goal."""
    tensors = {
        "low_level_obs": low_level_obs,
        "arm_joint_pos": arm_joint_pos,
        "arm_joint_vel": arm_joint_vel,
        "hand_base_command": hand_base_command,
    }
    _check_batched_tensors(tensors)
    for name, tensor in tensors.items():
        expected_width = 7 if name != "low_level_obs" else None
        if expected_width is not None and tensor.shape[1] != expected_width:
            raise ValueError(f"{name} must have width {expected_width}, got {tensor.shape[1]}.")
    return torch.cat(tuple(tensors.values()), dim=1)


def build_bc_action(
    arm_joint_target: torch.Tensor,
    arm_joint_pos: torch.Tensor,
    hand_joint_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[arm delta radians, hand target radians]`` and return its arm delta."""
    tensors = {
        "arm_joint_target": arm_joint_target,
        "arm_joint_pos": arm_joint_pos,
        "hand_joint_target": hand_joint_target,
    }
    _check_batched_tensors(tensors)
    for name in ("arm_joint_target", "arm_joint_pos"):
        if tensors[name].shape[1] != 7:
            raise ValueError(f"{name} must have width 7, got {tensors[name].shape[1]}.")
    if hand_joint_target.shape[1] != 16:
        raise ValueError(f"hand_joint_target must have width 16, got {hand_joint_target.shape[1]}.")
    arm_delta = arm_joint_target - arm_joint_pos
    return torch.cat((arm_delta, hand_joint_target), dim=1), arm_delta


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.floating):
        array = array.astype(np.float32, copy=False)
    return array


def flatten_scene_state(scene_state: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Flatten environment-relative articulation and rigid-object state for NPZ storage."""
    flattened: dict[str, np.ndarray] = {}
    for entity_type in ("articulation", "rigid_object"):
        for asset_name, asset_state in scene_state.get(entity_type, {}).items():
            for field_name, value in asset_state.items():
                key = f"state.{entity_type}.{asset_name}.{field_name}"
                flattened[key] = _to_numpy(value)
    return flattened


class InstantDexterityEpisodeRecorder:
    """Buffer vectorized transitions and write one compressed NPZ per episode."""

    def __init__(self, output_dir: str | Path, num_envs: int, metadata: Mapping[str, Any]) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_dir / "metadata.json"
        if metadata_path.exists() or next(self.output_dir.glob("*.npz"), None) is not None:
            raise FileExistsError(f"Recording directory already contains recorded data: {self.output_dir}")
        self.num_envs = int(num_envs)
        self._buffers: list[list[dict[str, np.ndarray]]] = [[] for _ in range(self.num_envs)]
        self._next_episode_index = 0
        self._closed = False
        with metadata_path.open("w") as file:
            json.dump(dict(metadata), file, indent=2, sort_keys=True)

    @property
    def num_saved(self) -> int:
        return self._next_episode_index

    def add_step(
        self,
        step_data: Mapping[str, Any],
        reward: Any,
        terminated: Any,
        truncated: Any,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot add data after the recorder is closed.")

        arrays = {key: _to_numpy(value) for key, value in step_data.items()}
        arrays["reward"] = _to_numpy(reward)
        arrays["terminated"] = _to_numpy(terminated).astype(bool, copy=False)
        arrays["truncated"] = _to_numpy(truncated).astype(bool, copy=False)
        for key, array in arrays.items():
            if array.ndim == 0 or array.shape[0] != self.num_envs:
                raise ValueError(
                    f"Recorded field '{key}' must have leading dimension {self.num_envs}, got {array.shape}."
                )

        for env_id in range(self.num_envs):
            self._buffers[env_id].append({key: value[env_id].copy() for key, value in arrays.items()})
            if bool(arrays["terminated"][env_id] or arrays["truncated"][env_id]):
                self._flush(env_id, complete=True)

    def _flush(self, env_id: int, complete: bool) -> None:
        frames = self._buffers[env_id]
        if not frames:
            return
        episode = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
        episode["complete"] = np.asarray(complete, dtype=bool)
        episode["source_env_id"] = np.asarray(env_id, dtype=np.int64)
        path = self.output_dir / f"{self._next_episode_index:010d}.npz"
        np.savez_compressed(path, **episode)
        self._buffers[env_id] = []
        self._next_episode_index += 1

    def close(self) -> None:
        if self._closed:
            return
        for env_id in range(self.num_envs):
            self._flush(env_id, complete=False)
        self._closed = True


__all__ = [
    "InstantDexterityEpisodeRecorder",
    "build_bc_action",
    "build_bc_observation",
    "flatten_scene_state",
]
