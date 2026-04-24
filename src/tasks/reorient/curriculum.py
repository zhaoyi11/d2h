# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

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
    """Adjust per-environment difficulty from consecutive task success."""

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
        num_success: int,
        command_name: str = "object_pose",
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 10,
        promotion_only: bool = False,
    ):
        command_term = env.command_manager.get_term(command_name)
        consecutive_success = command_term.metrics["consecutive_success"][env_ids]
        promote = consecutive_success >= num_success
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
        self.difficulty_frac = torch.median(self.current_adr_difficulties).item() / max(
            max_difficulty, 1
        )
        print(f"Difficulty frac: {self.difficulty_frac}")
        return self.difficulty_frac


##############
## ConfigClass
##############
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # adr stands for automatic/adaptive domain randomization
    adr = CurrTerm(
        func=DifficultyScheduler, params={"init_difficulty": 0, "min_difficulty": 0, "max_difficulty": 10}
    )

    # # Observation noise terms
    obj_point_cloud_unoise_min_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.perception.object_point_cloud.noise.n_min",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": -0.01, "difficulty_term_str": "adr"},
        },
    )

    obj_point_cloud_unoise_max_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.perception.object_point_cloud.noise.n_max",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    # # Environment event terms
    # robot_physics_material_static_friction_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_physics_material.params.static_friction_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.7, 1.3),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # robot_physics_material_dynamic_friction_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_physics_material.params.dynamic_friction_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.7, 1.3),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # robot_scale_mass_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_scale_mass.params.mass_distribution_params",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.95, 1.05),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # robot_joint_stiffness_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_joint_stiffness_and_damping.params.stiffness_distribution_params",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.3, 3.0),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # robot_joint_damping_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.robot_joint_stiffness_and_damping.params.damping_distribution_params",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.75, 1.5),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # object_physics_material_static_friction_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.object_physics_material.params.static_friction_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.7, 1.3),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # object_physics_material_dynamic_friction_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.object_physics_material.params.dynamic_friction_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.7, 1.3),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # object_scale_mass_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.object_scale_mass.params.mass_distribution_params",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (1.0, 1.0),
    #             "final_value": (0.4, 1.6),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # hand_object_pose_roll_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.randomize_hand_object_default_pose.params.roll_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-3.141592653589793, 3.141592653589793),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # hand_object_pose_pitch_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.randomize_hand_object_default_pose.params.pitch_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-3.141592653589793, 3.141592653589793),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # hand_object_pose_yaw_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.randomize_hand_object_default_pose.params.yaw_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-3.141592653589793, 3.141592653589793),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

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
