from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .contacts import _compute_contact_metrics


def gravity_dir_b(
    env: ManagerBasedRLEnv,
    base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Unit gravity vector expressed in robot root frame. Shape: (num_envs, 3).

    When the scene is rotated (e.g., by randomize_hand_object_default_pose), the gravity
    direction changes in robot frame. This observation allows the policy to adapt its
    grasp strategy based on orientation (palm-up vs palm-down vs sideways).
    """
    robot: Articulation = env.scene[base_asset_cfg.name]
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, gravity_w)


def recorded_object_scale(
    env: ManagerBasedRLEnv,
    key: str = "object_scale",
) -> torch.Tensor:
    """Recorded object scale from pre-startup randomization. Shape: (num_envs, 3)."""
    scale = env.extras.get(key)
    if scale is None:
        return torch.ones((env.num_envs, 3), device=env.device)
    return scale.to(device=env.device, dtype=torch.float32)


def _cached_privileged_tensor(
    env: ManagerBasedRLEnv,
    key: str,
    fallback: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cached privileged tensor recorded by a randomization event."""
    if key not in env.extras:
        if fallback is None:
            raise RuntimeError(
                f"Missing cached privileged observation '{key}'. "
                "Use the matching randomization recording wrapper before reading this term."
            )
        env.extras[key] = fallback.to(device=env.device, dtype=torch.float32)
    tensor = env.extras[key]
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor, device=env.device, dtype=torch.float32)
    return tensor.to(device=env.device, dtype=torch.float32)


def recorded_asset_masses(
    env: ManagerBasedRLEnv,
    key: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Cached rigid body masses for an asset. Shape: (num_envs, num_bodies)."""
    fallback = None
    if asset_cfg is not None and key not in env.extras:
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        fallback = asset.root_physx_view.get_masses()
    return _cached_privileged_tensor(env, key, fallback=fallback)


def _recorded_asset_material_properties(
    env: ManagerBasedRLEnv,
    key: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    fallback = None
    if asset_cfg is not None and key not in env.extras:
        asset: RigidObject | Articulation = env.scene[asset_cfg.name]
        fallback = asset.root_physx_view.get_material_properties()
    return _cached_privileged_tensor(env, key, fallback=fallback)


def recorded_asset_static_friction(
    env: ManagerBasedRLEnv,
    key: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Cached per-shape static friction coefficients for an asset."""
    return _recorded_asset_material_properties(env, key, asset_cfg=asset_cfg)[:, :, 0]


def recorded_asset_dynamic_friction(
    env: ManagerBasedRLEnv,
    key: str,
    asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Cached per-shape dynamic friction coefficients for an asset."""
    return _recorded_asset_material_properties(env, key, asset_cfg=asset_cfg)[:, :, 1]


def asset_joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint stiffness values for an articulation. Shape: (num_envs, num_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness[:, asset_cfg.joint_ids]


def asset_joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint damping values for an articulation. Shape: (num_envs, num_joints)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping[:, asset_cfg.joint_ids]


def object_lin_vel_robot_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root linear velocity relative to the robot root, expressed in robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    rel_vel_w = obj.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)


def object_ang_vel_robot_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object root angular velocity relative to the robot root, expressed in robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    rel_ang_vel_w = obj.data.root_ang_vel_w - robot.data.root_ang_vel_w
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)


def tip_contact_mask_obs(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
) -> torch.Tensor:
    """Per-fingertip binary contact mask, shape (num_envs, n_fingers)."""
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, 50.0
    )
    return m["in_contact"].float()                               # (N, n)


def tip_contact_force_mag_obs(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
) -> torch.Tensor:
    """Per-fingertip contact force magnitudes, shape (num_envs, n_fingers)."""
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, 50.0
    )
    return m["force_mag"]                                        # (N, n)


def tip_contact_pose_flat(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
    contact_pose_range_deg: float = 50.0,
) -> torch.Tensor:
    """Per-fingertip (theta, phi) flattened; zeroed for non-contacting fingers. Shape (N, n*2)."""
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, contact_pose_range_deg
    )
    pose = m["contact_pose"] * m["in_contact"].unsqueeze(-1).float()
    return pose.flatten(start_dim=1)                             # (N, n*2)


