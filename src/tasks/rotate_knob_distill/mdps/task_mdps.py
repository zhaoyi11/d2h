"""Observation and objective terms for knob policy distillation."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul

from src.policy.knob_interface import build_aria_frame

from .commands import wrap_to_pi, yaw_from_quat


def student_frame(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    action_name: str = "hand_action",
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Current 35D hardware-observable frame; IsaacLab supplies its three-frame history."""
    robot = env.scene[asset_cfg.name]
    knob = env.scene[object_cfg.name]
    action = env.action_manager.get_term(action_name)
    joint_ids = action._joint_ids
    joint_pos = robot.data.joint_pos[:, joint_ids]
    limits = robot.data.soft_joint_pos_limits[:, joint_ids]
    applied_targets = action._prev_applied_actions
    angle = yaw_from_quat(knob.data.root_quat_w)
    velocity = knob.data.root_ang_vel_w[:, 2]
    command = env.command_manager.get_term(command_name)
    target_quat_w = quat_mul(robot.data.root_quat_w, command.pose_command_b[:, 3:])
    target = yaw_from_quat(target_quat_w)
    return build_aria_frame(joint_pos, applied_targets, angle, velocity, target, limits[..., 0], limits[..., 1])


def angle_error(env, command_name: str = "object_pose") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    angle = yaw_from_quat(command.object.data.root_quat_w)
    return wrap_to_pi(command.target_angle - angle).abs()


def yaw_tracking(env, command_name: str = "object_pose", std: float = 0.5) -> torch.Tensor:
    return 1.0 - torch.tanh(angle_error(env, command_name) / std)


def action_rate_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


__all__ = ["action_rate_l2", "angle_error", "student_frame", "yaw_tracking"]
