from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from src.tasks.common.mdps.terminations import object_outside_table

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


BOX_TARGET_OFFSET = (0.0, 0.0, 0.065)
BOX_MIN = (-0.09, -0.15, 0.005)
BOX_MAX = (0.09, 0.15, 0.105)


def object_pos_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    position, _ = subtract_frame_transforms(
        box.data.root_pos_w,
        box.data.root_quat_w,
        object_asset.data.root_pos_w,
        None,
    )
    return position


def box_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    box: RigidObject = env.scene[box_cfg.name]
    position = quat_apply_inverse(
        robot.data.root_quat_w,
        box.data.root_pos_w - robot.data.root_pos_w,
    )
    orientation = quat_mul(quat_inv(robot.data.root_quat_w), box.data.root_quat_w)
    return torch.cat((position, orientation), dim=1)


def object_height_above_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    table_half_height: float = 0.02,
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    return object_asset.data.root_pos_w[:, 2] - (table.data.root_pos_w[:, 2] + table_half_height)


def lift_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
) -> torch.Tensor:
    height = object_height_above_table(env, object_cfg, table_cfg)
    return torch.clamp(height / target_height, 0.0, 1.0)


def transport_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.35,
    lift_height: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
) -> torch.Tensor:
    position = object_pos_box(env, object_cfg, box_cfg)
    target = position.new_tensor(BOX_TARGET_OFFSET)
    distance = torch.norm(position - target, dim=1)
    lifted = object_height_above_table(env, object_cfg, table_cfg) > lift_height
    return (1.0 - torch.tanh(distance / std)) * lifted.float()


def inside_box(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    position = object_pos_box(env, object_cfg, box_cfg)
    lower = position.new_tensor(BOX_MIN)
    upper = position.new_tensor(BOX_MAX)
    return ((position >= lower) & (position <= upper)).all(dim=1)


def inside_box_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    return inside_box(env, object_cfg, box_cfg).float()


def placed_and_released(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    hand_base_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    settle_speed: float = 0.05,
    hand_clear_distance: float = 0.10,
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[hand_base_cfg.name]
    hand_position = robot.data.body_pos_w[:, hand_base_cfg.body_ids]
    if hand_position.ndim == 3:
        if hand_position.shape[1] != 1:
            raise ValueError("Success requires hand_base_cfg to resolve exactly one body.")
        hand_position = hand_position[:, 0]
    object_speed = torch.norm(object_asset.data.root_lin_vel_w, dim=1)
    hand_distance = torch.norm(hand_position - object_asset.data.root_pos_w, dim=1)
    return (
        inside_box(env, object_cfg, box_cfg)
        & (object_speed < settle_speed)
        & (hand_distance > hand_clear_distance)
    )


def success_reward(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    hand_base_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    return placed_and_released(env, object_cfg, box_cfg, hand_base_cfg).float()


__all__ = [
    "BOX_MAX",
    "BOX_MIN",
    "BOX_TARGET_OFFSET",
    "box_pose_b",
    "inside_box",
    "inside_box_reward",
    "lift_reward",
    "object_height_above_table",
    "object_outside_table",
    "object_pos_box",
    "placed_and_released",
    "success_reward",
    "transport_reward",
]
