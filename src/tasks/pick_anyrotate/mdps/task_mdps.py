from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_lifted_above_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    height: float = 0.10,
    table_half_height: float = 0.02,
) -> torch.Tensor:
    """Return normalized object height above the table top."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    table_top_z = table.data.root_pos_w[:, 2] + table_half_height
    return torch.clamp((object_asset.data.root_pos_w[:, 2] - table_top_z) / height, 0.0, 1.0)


def trajectory_orientation_tracking(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    std: float = 0.5,
) -> torch.Tensor:
    """Track orientation only while a sampled reorientation target is active."""
    command = env.command_manager.get_term(command_name)
    active = command.metrics["reorientation_active"]
    return (1.0 - torch.tanh(command.metrics["orientation_error"] / std)) * active


def reorientation_goal_achieved(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Sparse pulse after one target's dwell finishes."""
    command = env.command_manager.get_term(command_name)
    return command.metrics["reorientation_goal_achieved"]


def sequence_completion_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Sparse reward when all six target dwells finish."""
    command = env.command_manager.get_term(command_name)
    return command.metrics["sequence_complete"]


def reorientation_sequence_complete(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Terminate environments that completed all six orientation targets."""
    command = env.command_manager.get_term(command_name)
    return command.metrics["sequence_complete"].bool()
