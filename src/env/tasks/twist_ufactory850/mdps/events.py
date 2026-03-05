# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for loading deterministic reset states from an NPY file."""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_REQUIRED_KEYS = ("object_root_state", "robot_root_state", "robot_joint_pos")


def _normalize_state_array(name: str, value, expected_width: int | None = None) -> np.ndarray:
    """Convert a state value into a float32 2D array with optional width check."""
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    elif array.ndim != 2:
        raise ValueError(f"State key '{name}' must be a 1D or 2D array. Got shape {array.shape}.")

    if expected_width is not None and array.shape[1] != expected_width:
        raise ValueError(
            f"State key '{name}' must have width {expected_width}. Got shape {array.shape}."
        )

    return array


def load_reset_state_npy(state_file_path: str) -> dict[str, np.ndarray]:
    """Load and validate a reset-state dict from an ``.npy`` file.

    Expected schema:
      - object_root_state: (13,) or (N, 13)
      - robot_root_state: (13,) or (N, 13)
      - robot_joint_pos: (J,) or (N, J)
      - robot_joint_vel: optional (J,) or (N, J)
    """
    path = Path(state_file_path)
    if not path.is_file():
        raise ValueError(f"State file does not exist: {state_file_path}")

    payload = np.load(path, allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    if not isinstance(data, dict):
        raise ValueError(
            "State file must contain a dictionary saved via np.save(..., allow_pickle=True)."
        )

    state: dict[str, np.ndarray] = {}
    for key in _REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"Missing required state key: '{key}'.")

    state["object_root_state"] = _normalize_state_array(
        "object_root_state", data["object_root_state"], expected_width=13
    )
    state["robot_root_state"] = _normalize_state_array(
        "robot_root_state", data["robot_root_state"], expected_width=13
    )
    state["robot_joint_pos"] = _normalize_state_array(
        "robot_joint_pos", data["robot_joint_pos"], expected_width=None
    )

    if "robot_joint_vel" in data and data["robot_joint_vel"] is not None:
        state["robot_joint_vel"] = _normalize_state_array(
            "robot_joint_vel", data["robot_joint_vel"], expected_width=state["robot_joint_pos"].shape[1]
        )

    return state


def _env_ids_to_tensor(env_ids, num_envs: int, device: torch.device) -> torch.Tensor:
    """Normalize env_ids to a long tensor on the given device."""
    if env_ids is None or env_ids == slice(None):
        return torch.arange(num_envs, device=device, dtype=torch.long)

    if isinstance(env_ids, slice):
        start = 0 if env_ids.start is None else env_ids.start
        stop = num_envs if env_ids.stop is None else env_ids.stop
        step = 1 if env_ids.step is None else env_ids.step
        return torch.arange(start, stop, step, device=device, dtype=torch.long)

    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)

    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _select_state_rows(state: torch.Tensor, env_ids: torch.Tensor, num_envs: int, key_name: str) -> torch.Tensor:
    """Select per-env rows from either a shared row ``(1, D)`` or full table ``(num_envs, D)``."""
    if state.shape[0] == 1:
        return state.expand(env_ids.shape[0], -1)
    if state.shape[0] == num_envs:
        return state.index_select(0, env_ids)

    raise ValueError(
        f"State key '{key_name}' must have row count 1 or num_envs ({num_envs}), got {state.shape[0]}."
    )


def _map_prim_paths_by_env_index(prim_paths: list[str]) -> dict[int, str]:
    """Map prim paths to environment ids parsed from ``/env_<id>/`` in path."""
    env_map: dict[int, str] = {}
    for prim_path in prim_paths:
        match = re.search(r"/env_(\d+)(?:/|$)", prim_path)
        if match is None:
            continue
        env_map[int(match.group(1))] = prim_path
    if env_map:
        return env_map
    return {idx: prim_path for idx, prim_path in enumerate(prim_paths)}


class filter_collisions_between_assets(ManagerTermBase):
    """Disable collisions between two assets using USD filtered-pairs relationship."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg_a: SceneEntityCfg = cfg.params.get("asset_cfg_a", SceneEntityCfg("object_table"))
        asset_cfg_b: SceneEntityCfg = cfg.params.get("asset_cfg_b", SceneEntityCfg("object"))
        self._asset_a: RigidObject = env.scene[asset_cfg_a.name]
        self._asset_b: RigidObject = env.scene[asset_cfg_b.name]
        self._bidirectional: bool = bool(cfg.params.get("bidirectional", True))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg_a: SceneEntityCfg = SceneEntityCfg("object_table"),
        asset_cfg_b: SceneEntityCfg = SceneEntityCfg("object"),
        bidirectional: bool = True,
    ):
        del env, env_ids, asset_cfg_a, asset_cfg_b, bidirectional
        from pxr import Sdf, UsdPhysics

        prim_paths_a = sim_utils.find_matching_prim_paths(self._asset_a.cfg.prim_path)
        prim_paths_b = sim_utils.find_matching_prim_paths(self._asset_b.cfg.prim_path)
        env_map_a = _map_prim_paths_by_env_index(prim_paths_a)
        env_map_b = _map_prim_paths_by_env_index(prim_paths_b)
        common_env_ids = sorted(set(env_map_a.keys()) & set(env_map_b.keys()))
        if not common_env_ids:
            raise RuntimeError(
                f"No overlapping env instances found for '{self._asset_a.cfg.prim_path}' and "
                f"'{self._asset_b.cfg.prim_path}'."
            )

        stage = self._asset_a.stage
        for env_id in common_env_ids:
            prim_path_a = env_map_a[env_id]
            prim_path_b = env_map_b[env_id]
            prim_a = stage.GetPrimAtPath(prim_path_a)
            prim_b = stage.GetPrimAtPath(prim_path_b)
            if not prim_a.IsValid() or not prim_b.IsValid():
                continue

            target_b = Sdf.Path(prim_path_b)
            rel_a = UsdPhysics.FilteredPairsAPI.Apply(prim_a).CreateFilteredPairsRel()
            if target_b not in rel_a.GetTargets():
                rel_a.AddTarget(target_b)

            if self._bidirectional:
                target_a = Sdf.Path(prim_path_a)
                rel_b = UsdPhysics.FilteredPairsAPI.Apply(prim_b).CreateFilteredPairsRel()
                if target_a not in rel_b.GetTargets():
                    rel_b.AddTarget(target_a)


class reset_root_state_from_npy(ManagerTermBase):
    """Reset an asset root state from a validated NPY state dictionary."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset: RigidObject = env.scene[asset_cfg.name]
        self._num_envs = env.num_envs

        state_file_path = cfg.params.get("state_file_path", None)
        if state_file_path is None:
            raise ValueError("reset_root_state_from_npy requires 'state_file_path'.")

        state_key = cfg.params.get("state_key", "object_root_state")
        state = load_reset_state_npy(state_file_path)
        if state_key not in state:
            raise ValueError(f"State key '{state_key}' was not found in: {state_file_path}")

        self._state_key = state_key
        self._root_state = torch.as_tensor(state[state_key], dtype=torch.float32, device=env.device)

        if self._root_state.shape[0] not in (1, self._num_envs):
            raise ValueError(
                f"State key '{state_key}' row count must be 1 or num_envs ({self._num_envs})."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        state_key: str = "object_root_state",
    ):
        del asset_cfg, state_key

        env_ids_t = _env_ids_to_tensor(env_ids, self._num_envs, env.device)
        root_states = _select_state_rows(self._root_state, env_ids_t, self._num_envs, self._state_key).clone()
        root_states[:, 0:3] += env.scene.env_origins.index_select(0, env_ids_t)
        self._asset.write_root_state_to_sim(root_states, env_ids=env_ids_t)


class reset_joints_from_npy(ManagerTermBase):
    """Reset an articulation joint state from a validated NPY state dictionary."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot", joint_names=".*"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._num_envs = env.num_envs

        state_file_path = cfg.params.get("state_file_path", None)
        if state_file_path is None:
            raise ValueError("reset_joints_from_npy requires 'state_file_path'.")

        joint_pos_key = cfg.params.get("joint_pos_key", "robot_joint_pos")
        joint_vel_key = cfg.params.get("joint_vel_key", "robot_joint_vel")

        state = load_reset_state_npy(state_file_path)
        if joint_pos_key not in state:
            raise ValueError(f"State key '{joint_pos_key}' was not found in: {state_file_path}")

        joint_pos = state[joint_pos_key]
        joint_count = len(self._asset.joint_names)
        if joint_pos.shape[1] != joint_count:
            raise ValueError(
                f"Joint position width mismatch. Expected {joint_count}, got {joint_pos.shape[1]}."
            )

        if joint_vel_key in state:
            joint_vel = state[joint_vel_key]
            if joint_vel.shape[1] != joint_count:
                raise ValueError(
                    f"Joint velocity width mismatch. Expected {joint_count}, got {joint_vel.shape[1]}."
                )
        else:
            joint_vel = np.zeros_like(joint_pos, dtype=np.float32)

        self._joint_pos_key = joint_pos_key
        self._joint_vel_key = joint_vel_key
        self._joint_pos = torch.as_tensor(joint_pos, dtype=torch.float32, device=env.device)
        self._joint_vel = torch.as_tensor(joint_vel, dtype=torch.float32, device=env.device)

        if self._joint_pos.shape[0] not in (1, self._num_envs):
            raise ValueError(
                f"State key '{joint_pos_key}' row count must be 1 or num_envs ({self._num_envs})."
            )
        if self._joint_vel.shape[0] not in (1, self._num_envs):
            raise ValueError(
                f"State key '{joint_vel_key}' row count must be 1 or num_envs ({self._num_envs})."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
        joint_pos_key: str = "robot_joint_pos",
        joint_vel_key: str = "robot_joint_vel",
    ):
        del asset_cfg, joint_pos_key, joint_vel_key

        env_ids_t = _env_ids_to_tensor(env_ids, self._num_envs, env.device)

        joint_pos = _select_state_rows(self._joint_pos, env_ids_t, self._num_envs, self._joint_pos_key).clone()
        joint_vel = _select_state_rows(self._joint_vel, env_ids_t, self._num_envs, self._joint_vel_key).clone()

        joint_pos_limits = self._asset.data.soft_joint_pos_limits[0]
        joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        joint_vel_limits = self._asset.data.soft_joint_vel_limits[0]
        joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)


class record_object_init_quat(ManagerTermBase):
    """Record object's post-reset quaternion for per-episode z-rotation tracking."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset = env.scene[asset_cfg.name]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        root_quat_w = self._asset.data.root_quat_w
        if (
            "object_init_quat" not in env.extras
            or env.extras["object_init_quat"] is None
            or env.extras["object_init_quat"].shape != root_quat_w.shape
        ):
            env.extras["object_init_quat"] = root_quat_w.clone()

        if env_ids is None or env_ids == slice(None):
            env.extras["object_init_quat"][:] = root_quat_w
        else:
            env.extras["object_init_quat"][env_ids] = root_quat_w[env_ids]


class reset_joints_to_init_state(ManagerTermBase):
    """Reset an articulation's joints to the init_state joint_pos from config.

    Use this when init_state.joint_pos in ArticulationCfg is not applied correctly
    at spawn (e.g. due to multi-env cloning or joint name mismatches). This event
    explicitly applies the desired joint positions on every env.reset().

    Requires params:
        asset_cfg: SceneEntityCfg for the robot
        joint_pos: dict[str, float] - joint name (or regex) -> position, from init_state
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._num_envs = env.num_envs

        joint_pos_dict = cfg.params.get("joint_pos", None)
        if joint_pos_dict is None:
            raise ValueError(
                "reset_joints_to_init_state requires 'joint_pos' dict (from init_state)."
            )

        index_list, _, values_list = resolve_matching_names_values(
            joint_pos_dict,
            self._asset.joint_names,
            preserve_order=True,
            strict=False,
        )
        joint_pos_array = [0.0] * len(self._asset.joint_names)
        for idx, val in zip(index_list, values_list):
            joint_pos_array[idx] = val

        self._joint_pos = torch.as_tensor(
            joint_pos_array, dtype=torch.float32, device=env.device
        ).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        joint_pos: dict | None = None,
    ):
        del asset_cfg, joint_pos

        env_ids_t = _env_ids_to_tensor(env_ids, self._num_envs, env.device)
        joint_pos = self._joint_pos.expand(len(env_ids_t), -1).clone()
        joint_vel = torch.zeros_like(joint_pos)

        joint_pos_limits = self._asset.data.soft_joint_pos_limits[0]
        joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
