from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _place_context(env: ManagerBasedRLEnv, command_name: str):
    return env.command_manager.get_term(command_name)


def lift_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    target_height: float = 0.08,
) -> torch.Tensor:
    context = _place_context(env, command_name)
    return torch.clamp(context.object_height_above_table / target_height, 0.0, 1.0)


def transport_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    std: float = 0.35,
) -> torch.Tensor:
    context = _place_context(env, command_name)
    reward = 1.0 - torch.tanh(context.object_to_box_distance / std)
    return reward * context.lifted.float()


def inside_box_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    return _place_context(env, command_name).inside_box.float()


def place_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    return _place_context(env, command_name).success.float()


def object_pos_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    pos_box, _ = subtract_frame_transforms(
        box.data.root_pos_w,
        box.data.root_quat_w,
        obj.data.root_pos_w,
        None,
    )
    return pos_box


def box_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_b = quat_apply_inverse(
        robot.data.root_quat_w, box.data.root_pos_w - robot.data.root_pos_w
    )
    box_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), box.data.root_quat_w)
    return torch.cat((box_pos_b, box_quat_b), dim=1)
