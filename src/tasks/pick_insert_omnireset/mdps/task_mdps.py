"""Task-local geometry, rewards, and success terms for pick-insert OmniReset."""

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

from src.tasks.pick_insert_omnireset.mdps.asset_geometry import (
    insertion_geometry_from_assets,
)
from src.tasks.pick_insert_omnireset.mdps.geometry import (
    INSERTION_SHAPING_TARGET_DEPTH,
    peg_assembly_pose_errors,
    peg_inside_rectangular_hole,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def hole_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    position = quat_apply_inverse(
        robot.data.root_quat_w,
        hole.data.root_pos_w - robot.data.root_pos_w,
    )
    orientation = quat_mul(quat_inv(robot.data.root_quat_w), hole.data.root_quat_w)
    return torch.cat((position, orientation), dim=1)


def object_lin_vel_robot_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    relative_velocity_w = object_asset.data.root_lin_vel_w - robot.data.root_lin_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, relative_velocity_w)


def object_ang_vel_robot_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    relative_velocity_w = object_asset.data.root_ang_vel_w - robot.data.root_ang_vel_w
    return quat_apply_inverse(robot.data.root_quat_w, relative_velocity_w)


def object_pose_hole(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    position, orientation = subtract_frame_transforms(
        hole.data.root_pos_w,
        hole.data.root_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    return torch.cat((position, orientation), dim=1)


def object_lifted_above_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    height: float = 0.06,
    table_half_height: float = 0.02,
) -> torch.Tensor:
    object_asset: RigidObject = env.scene[object_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    table_top_z = table.data.root_pos_w[:, 2] + table_half_height
    return torch.clamp((object_asset.data.root_pos_w[:, 2] - table_top_z) / height, 0.0, 1.0)


def object_outside_table(
    env: ManagerBasedRLEnv,
    table_half_extents: tuple[float, float] = (0.4, 0.75),
    table_half_height: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
) -> torch.Tensor:
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


class DenseAssemblyPose(ManagerTermBase):
    """Dense final-pose reward derived from the insertion assets."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        hole_cfg: SceneEntityCfg = cfg.params["hole_cfg"]
        self._geometry = insertion_geometry_from_assets(
            env.scene[object_cfg.name],
            env.scene[hole_cfg.name],
            env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
        position_std: float = 0.08,
        orientation_std: float = 0.35,
        target_depth: float = INSERTION_SHAPING_TARGET_DEPTH,
    ) -> torch.Tensor:
        object_asset: RigidObject = env.scene[object_cfg.name]
        hole: RigidObject = env.scene[hole_cfg.name]
        object_root_pose = torch.cat(
            (object_asset.data.root_pos_w, object_asset.data.root_quat_w),
            dim=1,
        )
        hole_root_pose = torch.cat(
            (hole.data.root_pos_w, hole.data.root_quat_w),
            dim=1,
        )
        position_error, tilt_error = peg_assembly_pose_errors(
            object_root_pose,
            hole_root_pose,
            self._geometry,
            target_depth,
        )
        position_score = torch.exp(-position_error / position_std)
        orientation_score = torch.exp(-tilt_error / orientation_std)
        return 0.5 * (position_score + orientation_score)


class PegInsideHole(ManagerTermBase):
    """Shared asset-derived insertion predicate for rewards and terminations."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        object_cfg: SceneEntityCfg = cfg.params["object_cfg"]
        hole_cfg: SceneEntityCfg = cfg.params["hole_cfg"]
        self._geometry = insertion_geometry_from_assets(
            env.scene[object_cfg.name],
            env.scene[hole_cfg.name],
            env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    ) -> torch.Tensor:
        object_asset: RigidObject = env.scene[object_cfg.name]
        hole: RigidObject = env.scene[hole_cfg.name]
        object_root_pose = torch.cat(
            (object_asset.data.root_pos_w, object_asset.data.root_quat_w),
            dim=1,
        )
        hole_root_pose = torch.cat(
            (hole.data.root_pos_w, hole.data.root_quat_w),
            dim=1,
        )
        return peg_inside_rectangular_hole(
            object_root_pose,
            hole_root_pose,
            self._geometry,
        )


__all__ = [
    "DenseAssemblyPose",
    "hole_pose_b",
    "object_ang_vel_robot_b",
    "object_lin_vel_robot_b",
    "object_lifted_above_table",
    "object_outside_table",
    "object_pose_hole",
    "PegInsideHole",
]
