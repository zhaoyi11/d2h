"""Task-local observations, rewards, and terminations for unscrew OmniReset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from src.tasks.common.mdps.rewards import contacts as good_object_contact
from src.tasks.unscrew.geometry import (
    bolt_aabb_corners,
    bolt_bottom_clearance,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def receptive_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    """Receptive-object pose expressed in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    receptive: RigidObject = env.scene[receptive_cfg.name]
    position = quat_apply_inverse(
        robot.data.root_quat_w,
        receptive.data.root_pos_w - robot.data.root_pos_w,
    )
    orientation = quat_mul(
        quat_inv(robot.data.root_quat_w), receptive.data.root_quat_w
    )
    return torch.cat((position, orientation), dim=1)


def object_pose_receptive(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    """Object pose expressed in the receptive-object frame."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    receptive: RigidObject = env.scene[receptive_cfg.name]
    position, orientation = subtract_frame_transforms(
        receptive.data.root_pos_w,
        receptive.data.root_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    return torch.cat((position, orientation), dim=1)


def object_lin_vel_robot_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear velocity relative to and expressed in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    relative_velocity_w = object_asset.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, relative_velocity_w)


def object_ang_vel_robot_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object angular velocity relative to and expressed in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    relative_velocity_w = object_asset.data.root_ang_vel_w - robot.data.root_ang_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, relative_velocity_w)


def object_outside_table(
    env: ManagerBasedRLEnv,
    table_half_extents: tuple[float, float] = (0.4, 0.75),
    table_half_height: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
) -> torch.Tensor:
    """Return whether the object left or fell below the loaded table frame."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    position, _ = subtract_frame_transforms(
        table.data.root_pos_w,
        table.data.root_quat_w,
        object_asset.data.root_pos_w,
        None,
    )
    half_extents = position.new_tensor(table_half_extents)
    outside_footprint = (position[:, :2].abs() > half_extents).any(dim=1)
    below_surface = position[:, 2] < table_half_height
    return outside_footprint | below_surface


class _UnscrewClearanceTerm(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params.get(
            "object_cfg", SceneEntityCfg("object")
        )
        receptive_cfg: SceneEntityCfg = cfg.params.get(
            "receptive_cfg", SceneEntityCfg("receptive_object")
        )
        self._object: RigidObject = env.scene[object_cfg.name]
        self._receptive: RigidObject = env.scene[receptive_cfg.name]
        self._bolt_corners = bolt_aabb_corners(env.device)

    def _clearance(self) -> torch.Tensor:
        object_pos_r, object_quat_r = subtract_frame_transforms(
            self._receptive.data.root_pos_w,
            self._receptive.data.root_quat_w,
            self._object.data.root_pos_w,
            self._object.data.root_quat_w,
        )
        return bolt_bottom_clearance(
            object_pos_r,
            object_quat_r,
            self._bolt_corners,
        )


class DenseUnscrewClearance(_UnscrewClearanceTerm):
    """Dense extraction progress from installed to successful bolt clearance."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
        start_clearance: float = -0.040,
        target_clearance: float = 0.030,
    ) -> torch.Tensor:
        if target_clearance <= start_clearance:
            raise ValueError("target_clearance must be greater than start_clearance.")
        return (
            (self._clearance() - start_clearance)
            / (target_clearance - start_clearance)
        ).clamp(0.0, 1.0)


class UnscrewSuccess(_UnscrewClearanceTerm):
    """Immediate clearance-and-grasp success with episode-level memory."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.episode_succeeded = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.episode_succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
        clearance_margin: float = 0.030,
        force_threshold: float = 1.0,
    ) -> torch.Tensor:
        if clearance_margin < 0.0:
            raise ValueError("clearance_margin must be non-negative.")
        if force_threshold < 0.0:
            raise ValueError("force_threshold must be non-negative.")
        current_success = (self._clearance() >= clearance_margin) & good_object_contact(
            env, force_threshold
        )
        self.episode_succeeded |= current_success
        return current_success


__all__ = [
    "DenseUnscrewClearance",
    "UnscrewSuccess",
    "bolt_aabb_corners",
    "object_ang_vel_robot_b",
    "object_lin_vel_robot_b",
    "object_outside_table",
    "object_pose_receptive",
    "receptive_pose_b",
]
