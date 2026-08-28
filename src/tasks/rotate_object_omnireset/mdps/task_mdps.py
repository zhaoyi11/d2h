"""Task-local observations, rewards, and terminations for rotate-object OmniReset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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


def recorded_goal_yaw_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    std: float = 0.5,
) -> torch.Tensor:
    """Reward orientation tracking of the paired recorded goal."""
    error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    return 1.0 - torch.tanh(error / std)


class RecordedGoalSuccess(ManagerTermBase):
    """Immediate orientation success with episode-level memory for the curriculum."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.episode_succeeded = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.episode_succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str = "object_pose",
        tolerance: float = 0.2,
    ) -> torch.Tensor:
        current_success = (
            env.command_manager.get_term(command_name).metrics["orientation_error"]
            <= tolerance
        )
        self.episode_succeeded |= current_success
        return current_success


__all__ = [
    "RecordedGoalSuccess",
    "object_ang_vel_robot_b",
    "object_lin_vel_robot_b",
    "recorded_goal_yaw_tracking",
]
