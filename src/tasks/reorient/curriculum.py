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

        self.difficulty_frac = torch.mean(self.current_adr_difficulties).item() / max(max_difficulty, 1)
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

    # policy observation noise terms
    joint_pos_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_pos.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.005, "difficulty_term_str": "adr"},
        },
    )

    joint_vel_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.joint_vel.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    fingertip_pos_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.fingertip_pos.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.002, "difficulty_term_str": "adr"},
        },
    )

    fingertip_lin_vel_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.fingertip_lin_vel.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    fingertip_ang_vel_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.fingertip_ang_vel.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.01, "difficulty_term_str": "adr"},
        },
    )

    fingertip_contact_force_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.fingertip_contact_force_b.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.05, "difficulty_term_str": "adr"},
        },
    )

    contact_force_mag_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.contact_force_mag.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.05, "difficulty_term_str": "adr"},
        },
    )

    object_pos_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_pos.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.002, "difficulty_term_str": "adr"},
        },
    )

    object_lin_vel_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_lin_vel.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.002, "difficulty_term_str": "adr"},
        },
    )

    object_ang_vel_gnoise_std_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "observations.policy.object_ang_vel.noise.std",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {"initial_value": 0.0, "final_value": 0.002, "difficulty_term_str": "adr"},
        },
    )

    # perceptive observation noise terms
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

    # Wrist poses
    # hand_object_pose_roll_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.randomize_hand_object_default_pose.params.roll_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-torch.pi, torch.pi),
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
    #             "final_value": (-torch.pi, torch.pi),
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
    #             "final_value": (-torch.pi, torch.pi),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # # Objct poses
    # reset_object_x_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.reset_object.params.pose_range.x",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-0.01, 0.01),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # reset_object_y_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.reset_object.params.pose_range.y",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-0.01, 0.01),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # reset_object_z_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.reset_object.params.pose_range.z",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-0.01, 0.01),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # reset_object_roll_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.reset_object.params.pose_range.roll",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-torch.pi, torch.pi),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # reset_object_pitch_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.reset_object.params.pose_range.pitch",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-torch.pi, torch.pi),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # reset_object_yaw_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "events.reset_object.params.pose_range.yaw",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.0, 0.0),
    #             "final_value": (-torch.pi, torch.pi),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    # Command range
    # command_random_range_adr = CurrTerm(
    #     func=common_mdp.modify_term_cfg,
    #     params={
    #         "address": "commands.object_pose.random_range",
    #         "modify_fn": initial_final_interpolate_fn,
    #         "modify_params": {
    #             "initial_value": (0.3 * torch.pi, 0.5 * torch.pi),
    #             "final_value": (0.3 * torch.pi, torch.pi),
    #             "difficulty_term_str": "adr",
    #         },
    #     },
    # )

    command_orientation_success_threshold_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "commands.object_pose.orientation_success_threshold",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 0.3,
                "final_value": 0.3,  # todo: 0.1
                "difficulty_term_str": "adr",
            },
        },
    )

    command_position_success_threshold_adr = CurrTerm(
        func=common_mdp.modify_term_cfg,
        params={
            "address": "commands.object_pose.position_success_threshold",
            "modify_fn": initial_final_interpolate_fn,
            "modify_params": {
                "initial_value": 0.1,
                "final_value": 0.1,  # todo: 0.05
                "difficulty_term_str": "adr",
            },
        },
    )

    # Gravity ADR
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
