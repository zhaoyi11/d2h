from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def object_lifted_above_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    height: float = 0.06,
    table_half_height: float = 0.02,
) -> torch.Tensor:
    """Reward object lift height above the table top, clamped to [0, 1]."""

    object: RigidObject = env.scene[object_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    table_top_z = table.data.root_pos_w[:, 2] + table_half_height
    height_above_table = object.data.root_pos_w[:, 2] - table_top_z
    return torch.clamp(height_above_table / height, 0.0, 1.0)


def object_to_hole_xy_tanh(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    std: float = 0.08,
    contact_threshold: float = 1.0,
    lift_height: float = 0.06,
    lift_gate: float = 0.5,
) -> torch.Tensor:
    """Reward XY alignment between the object and hole, gated by grasp and lift."""

    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    xy_dist = torch.norm(object.data.root_pos_w[:, :2] - hole.data.root_pos_w[:, :2], dim=1)
    reward = 1.0 - torch.tanh(xy_dist / std)
    return reward * _pick_insert_gate(env, table_cfg, contact_threshold, lift_height, lift_gate)


def peg_hole_axis_alignment(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    std: float = 0.35,
) -> torch.Tensor:
    """Reward alignment of the peg and hole local Z axes."""

    axis_dot = _peg_hole_axis_dot(env, object_cfg, hole_cfg)
    axis_error = 1.0 - axis_dot
    return 1.0 - torch.tanh(axis_error / std)


def peg_insertion_depth(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    target_depth: float = 0.015,
    approach_height: float = 0.08,
    xy_tolerance: float = 0.04,
    axis_tolerance: float = 0.25,
    contact_threshold: float = 1.0,
    lift_height: float = 0.06,
    lift_gate: float = 0.5,
) -> torch.Tensor:
    """Reward downward insertion progress once grasped, lifted, centered, and aligned."""

    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    target_z = hole.data.root_pos_w[:, 2] + target_depth
    approach_z = target_z + approach_height
    progress = torch.clamp((approach_z - object.data.root_pos_w[:, 2]) / approach_height, 0.0, 1.0)

    xy_dist = torch.norm(object.data.root_pos_w[:, :2] - hole.data.root_pos_w[:, :2], dim=1)
    axis_error = 1.0 - _peg_hole_axis_dot(env, object_cfg, hole_cfg)
    insertion_gate = ((xy_dist < xy_tolerance) & (axis_error < axis_tolerance)).float()
    return progress * insertion_gate * _pick_insert_gate(env, table_cfg, contact_threshold, lift_height, lift_gate)


def peg_inserted_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    pos_tol: float = 0.015,
    axis_tol: float = 0.15,
    depth: float = 0.015,
) -> torch.Tensor:
    """Sparse success for a peg centered in the hole at the inserted height."""

    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    target_pos = hole.data.root_pos_w.clone()
    target_pos[:, 2] = target_pos[:, 2] + depth
    pos_dist = torch.norm(object.data.root_pos_w - target_pos, dim=1)
    axis_error = 1.0 - _peg_hole_axis_dot(env, object_cfg, hole_cfg)
    return ((pos_dist < pos_tol) & (axis_error < axis_tol)).float()


def _pick_insert_gate(
    env: ManagerBasedRLEnv,
    table_cfg: SceneEntityCfg,
    contact_threshold: float,
    lift_height: float,
    lift_gate: float,
) -> torch.Tensor:
    contact = _good_finger_contact(env, contact_threshold).float()
    lifted = object_lifted_above_table(env, table_cfg=table_cfg, height=lift_height)
    return contact * (lifted >= lift_gate).float()


def _good_finger_contact(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    thumb_contact = _contact_magnitude(env.scene.sensors["thumb_fingertip_object_s"])
    index_contact = _contact_magnitude(env.scene.sensors["fingertip_object_s"])
    middle_contact = _contact_magnitude(env.scene.sensors["fingertip_2_object_s"])
    ring_contact = _contact_magnitude(env.scene.sensors["fingertip_3_object_s"])
    return (thumb_contact > threshold) & (
        (index_contact > threshold)
        | (middle_contact > threshold)
        | (ring_contact > threshold)
    )


def _contact_magnitude(sensor: ContactSensor) -> torch.Tensor:
    force_w = torch.nan_to_num(sensor.data.force_matrix_w, nan=0.0).reshape(sensor.data.force_matrix_w.shape[0], -1, 3)
    return torch.norm(force_w.sum(dim=1), dim=-1)


def _peg_hole_axis_dot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    local_z = torch.zeros(env.num_envs, 3, device=object.data.root_quat_w.device)
    local_z[:, 2] = 1.0
    object_axis = quat_apply(object.data.root_quat_w, local_z)
    hole_axis = quat_apply(hole.data.root_quat_w, local_z)
    return torch.sum(object_axis * hole_axis, dim=1).abs().clamp(0.0, 1.0)
