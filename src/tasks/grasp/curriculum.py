# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import compute_pose_error

import src.tasks.common.mdps as common_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(
    env: ManagerBasedRLEnv,
    env_id,
    data,
    initial_value,
    final_value,
    difficulty_term_str,
):
    """Interpolate nested config values using the scheduler difficulty fraction."""
    difficulty_term: DifficultyScheduler = getattr(
        env.curriculum_manager.cfg, difficulty_term_str
    ).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        return env_mdp.modify_env_param.NO_CHANGE

    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(
        initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac
    )


def _recurse(iv_elem, fv_elem, data_elem, frac):
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        return type(data_elem)(
            _recurse(iv_e, fv_e, d_e, frac)
            for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem)
        )

    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val)
    return float(new_val)


class DifficultyScheduler(ManagerTermBase):
    """Adjust per-environment difficulty from pose-tracking success."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = (
            torch.ones(env.num_envs, device=env.device) * init_difficulty
        )
        self.difficulty_frac = 0.0

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        pos_tol: float = 0.1,
        rot_tol: float | None = None,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 10,
        promotion_only: bool = False,
    ):
        object_asset: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command("object_pose")
        desired_pos_w = command[env_ids, :3] + env.scene.env_origins[env_ids]
        desired_quat_w = command[env_ids, 3:7]
        pos_err, rot_err = compute_pose_error(
            desired_pos_w,
            desired_quat_w,
            object_asset.data.root_pos_w[env_ids],
            object_asset.data.root_quat_w[env_ids],
        )
        pos_dist = torch.norm(pos_err, dim=1)
        rot_dist = torch.norm(rot_err, dim=1)
        promote = (
            (pos_dist < pos_tol) & (rot_dist < rot_tol)
            if rot_tol is not None
            else pos_dist < pos_tol
        )
        demoted = (
            self.current_adr_difficulties[env_ids]
            if promotion_only
            else self.current_adr_difficulties[env_ids] - 1
        )
        self.current_adr_difficulties[env_ids] = torch.where(
            promote,
            self.current_adr_difficulties[env_ids] + 1,
            demoted,
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties).item() / max(
            max_difficulty, 1
        )
        print(f"Difficulty frac: {self.difficulty_frac}")
        return self.difficulty_frac


@configclass
class CurriculumCfg:
    """Curriculum terms for the in-hand reorientation task."""

    adr = CurrTerm(
        func=DifficultyScheduler,
        params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10},
    )

    gravity_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "events.variable_gravity.params.gravity_distribution_params",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                "final_value": ((0.0, 0.0, -9.81), (0.0, 0.0, -9.81)),
                "difficulty_term_str": "adr",
            },
        },
    )
