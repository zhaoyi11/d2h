from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def reset_object_pose_relative_to_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    local_pos: tuple[float, float, float] = (0.12, 0.0, 0.08),
    local_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    random_orientation: bool = False,
    velocity: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
):
    """Reset an object to a pose expressed relative to one articulated body."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_asset: Articulation = env.scene[body_asset_cfg.name]

    env_ids = _env_ids_tensor(env, env_ids, device=asset.device)
    body_pos_w, body_quat_w = _single_body_state_w(body_asset, body_asset_cfg, env_ids)[:2]

    local_pos_t = torch.tensor(local_pos, dtype=torch.float32, device=asset.device).repeat(len(env_ids), 1)
    if random_orientation:
        local_rot_t = math_utils.random_orientation(len(env_ids), device=asset.device)
    else:
        local_rot_t = torch.tensor(local_rot, dtype=torch.float32, device=asset.device).repeat(len(env_ids), 1)

    object_pos_w, object_quat_w = combine_frame_transforms(body_pos_w, body_quat_w, local_pos_t, local_rot_t)
    object_velocity = torch.tensor(velocity, dtype=torch.float32, device=asset.device).repeat(len(env_ids), 1)

    asset.write_root_pose_to_sim(torch.cat((object_pos_w, object_quat_w), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(object_velocity, env_ids=env_ids)


def move_dynamic_obstacle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("dynamic_obstacle"),
    center: tuple[float, float, float] = (0.4, 0.05, 0.45),
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    amplitude: float = 0.2,
    freq: float = 0.25,
    phase: float = 0.0,
) -> None:
    """Kinematically drive a scene prop along a sinusoid for use as a moving MPC obstacle.

    Per-env (identical across envs) world pose is
        ``pos = env_origin + center + axis * amplitude * sin(2*pi*freq*t + phase)``
    with ``t`` the GLOBAL sim time (``common_step_counter * step_dt``), so every env and the single
    shared cuRobo collision world agree on the obstacle pose. Orientation is identity. Intended as
    an every-step ``interval`` event (``interval_range_s=(0.0, 0.0)``). The matching cuRobo cuboid
    is updated separately by :class:`CommandHandBaseCuroboMpcAction`, which reads this prop's live
    pose each control step (see ``dynamic_obstacle_assets``).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids = _env_ids_tensor(env, env_ids, device=asset.device)

    t = float(env.common_step_counter) * env.step_dt
    offset = math.sin(2.0 * math.pi * freq * t + phase) * amplitude
    center_t = torch.tensor(center, dtype=torch.float32, device=asset.device)
    axis_t = torch.tensor(axis, dtype=torch.float32, device=asset.device)
    pos_local = center_t + axis_t * offset  # (3,) relative to each env origin

    pos_w = env.scene.env_origins[env_ids] + pos_local  # (n, 3)
    quat_w = torch.zeros(len(env_ids), 4, dtype=torch.float32, device=asset.device)
    quat_w[:, 0] = 1.0  # identity orientation (w, x, y, z)
    asset.write_root_pose_to_sim(torch.cat((pos_w, quat_w), dim=-1), env_ids=env_ids)


def _env_ids_tensor(env: ManagerBasedRLEnv, env_ids, device: str) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


def _single_body_state_w(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        body_names = getattr(asset_cfg, "body_names", None)
        body_ids, body_names = asset.find_bodies(body_names)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body matching {body_names}, found {len(body_ids)}.")

    body_pos_w = asset.data.body_pos_w[env_ids][:, body_ids]
    body_quat_w = asset.data.body_quat_w[env_ids][:, body_ids]
    if body_pos_w.ndim == 2:
        body_pos_w = body_pos_w.unsqueeze(1)
        body_quat_w = body_quat_w.unsqueeze(1)
    if body_pos_w.shape[1] != 1:
        raise ValueError(f"Expected one body id for {asset_cfg.name}, found {body_pos_w.shape[1]}.")

    body_lin_vel_w = getattr(asset.data, "body_lin_vel_w", None)
    body_ang_vel_w = getattr(asset.data, "body_ang_vel_w", None)
    if body_lin_vel_w is not None:
        body_lin_vel_w = body_lin_vel_w[env_ids][:, body_ids]
        if body_lin_vel_w.ndim == 2:
            body_lin_vel_w = body_lin_vel_w.unsqueeze(1)
    if body_ang_vel_w is not None:
        body_ang_vel_w = body_ang_vel_w[env_ids][:, body_ids]
        if body_ang_vel_w.ndim == 2:
            body_ang_vel_w = body_ang_vel_w.unsqueeze(1)

    return body_pos_w[:, 0], body_quat_w[:, 0], body_lin_vel_w, body_ang_vel_w


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
