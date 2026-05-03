from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, subtract_frame_transforms

from src.tasks.common.mdps import *  # noqa: F401, F403

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _get_place_context(env: ManagerBasedRLEnv, command_name: str) -> PlaceInBoxCommand:
    return env.command_manager.get_term(command_name)


@configclass
class PlaceInBoxCommandCfg(CommandTermCfg):
    """Context command for the clean-table place-in-box task."""

    class_type: type = None

    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
    box_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object")
    table_cfg: SceneEntityCfg = SceneEntityCfg("table")

    table_half_height: float = 0.02
    lift_height: float = 0.08
    box_target_pos: tuple[float, float, float] = (0.0, 0.0, 0.12)
    box_min: tuple[float, float, float] = (-0.18, -0.26, 0.02)
    box_max: tuple[float, float, float] = (0.18, 0.26, 0.24)

    def __post_init__(self):
        self.class_type = PlaceInBoxCommand


class PlaceInBoxCommand(CommandTerm):
    """Command-like context term that tracks object placement progress.

    The task has no sampled pose command. This term centralizes placement
    metrics so rewards and logging read the same tensors.
    """

    cfg: PlaceInBoxCommandCfg

    def __init__(self, cfg: PlaceInBoxCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.robot_cfg.name]
        self.object: RigidObject = env.scene[cfg.object_cfg.name]
        self.box: RigidObject = env.scene[cfg.box_cfg.name]
        self.table: RigidObject = env.scene[cfg.table_cfg.name]

        self._command = torch.zeros(self.num_envs, 3, device=self.device)
        self.object_height_above_table = torch.zeros(self.num_envs, device=self.device)
        self.object_to_box_distance = torch.zeros(self.num_envs, device=self.device)
        self.object_pos_box = torch.zeros(self.num_envs, 3, device=self.device)
        self.lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.inside_box = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.inside_box_metric = torch.zeros(self.num_envs, device=self.device)
        self.success_metric = torch.zeros(self.num_envs, device=self.device)

        self.metrics["object_height_above_table"] = self.object_height_above_table
        self.metrics["object_to_box_distance"] = self.object_to_box_distance
        self.metrics["inside_box"] = self.inside_box_metric
        self.metrics["success"] = self.success_metric

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        table_top_z = self.table.data.root_pos_w[:, 2] + self.cfg.table_half_height
        self.object_height_above_table[:] = self.object.data.root_pos_w[:, 2] - table_top_z
        self.lifted[:] = self.object_height_above_table > self.cfg.lift_height

        object_pos_box, _ = subtract_frame_transforms(
            self.box.data.root_pos_w,
            self.box.data.root_quat_w,
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
        )
        self.object_pos_box[:] = object_pos_box

        target = torch.tensor(self.cfg.box_target_pos, device=self.device).unsqueeze(0)
        self.object_to_box_distance[:] = torch.norm(object_pos_box - target, dim=1)

        box_min = torch.tensor(self.cfg.box_min, device=self.device).unsqueeze(0)
        box_max = torch.tensor(self.cfg.box_max, device=self.device).unsqueeze(0)
        self.inside_box[:] = ((object_pos_box >= box_min) & (object_pos_box <= box_max)).all(dim=1)
        self.success[:] = self.inside_box
        self.inside_box_metric[:] = self.inside_box.float()
        self.success_metric[:] = self.success.float()

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


PlaceInBoxCommandCfg.class_type = PlaceInBoxCommand


def lift_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "task_command",
    target_height: float = 0.08,
) -> torch.Tensor:
    context = _get_place_context(env, command_name)
    return torch.clamp(context.object_height_above_table / target_height, 0.0, 1.0)


def transport_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "task_command",
    std: float = 0.35,
) -> torch.Tensor:
    context = _get_place_context(env, command_name)
    reward = 1.0 - torch.tanh(context.object_to_box_distance / std)
    return reward * context.lifted.float()


def inside_box_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "task_command",
) -> torch.Tensor:
    context = _get_place_context(env, command_name)
    return context.inside_box.float()


def place_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "task_command",
) -> torch.Tensor:
    context = _get_place_context(env, command_name)
    return context.success.float()


def good_finger_contact(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Reward contact where thumb and at least one other fingertip touch the object."""

    sensor_names = [
        "thumb_fingertip_object_s",
        "fingertip_object_s",
        "fingertip_2_object_s",
        "fingertip_3_object_s",
    ]
    contact_magnitudes = []
    for name in sensor_names:
        force_w = env.scene.sensors[name].data.force_matrix_w
        force_w = torch.nan_to_num(force_w, nan=0.0)
        force_w = force_w.reshape(env.num_envs, -1, 3).sum(dim=1)
        contact_magnitudes.append(torch.norm(force_w, dim=-1))

    thumb_contact = contact_magnitudes[0] > threshold
    other_contact = (
        (contact_magnitudes[1] > threshold)
        | (contact_magnitudes[2] > threshold)
        | (contact_magnitudes[3] > threshold)
    )
    return (thumb_contact & other_contact).float()


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
