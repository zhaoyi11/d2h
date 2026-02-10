# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclab.sensors import ContactSensor

from isaaclab.utils.math import quat_apply

from src.utils.point_cloud import sample_object_point_cloud

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bonus reward for successfully reaching the goal.

    The object is considered to have reached the goal when the object orientation is within the threshold.
    The reward is 1.0 if the object has reached the goal, otherwise 0.0.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(
        command_name
    )

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # obtain the threshold for the orientation error
    threshold = command_term.cfg.orientation_success_threshold
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)

    return dtheta <= threshold


def track_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward for tracking the object position using the L2 norm.

    The reward is the distance between the object position and the goal position.

    Args:
        env: The environment object.
        command_term: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(
        command_name
    )

    # obtain the goal position
    goal_pos_e = command_term.command[:, 0:3]
    # obtain the object position in the environment frame
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    return torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
) -> torch.Tensor:
    """Reward for tracking the object orientation using the inverse of the orientation error.

    The reward is the inverse of the orientation error between the object orientation and the goal orientation.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        rot_eps: The threshold for the orientation error. Default is 1e-3.
    """
    # extract useful elements

    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(
        command_name
    )

    # obtain the goal orientation
    goal_quat_w = command_term.command[:, 3:7]
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)
    return 1.0 / (dtheta + rot_eps)


def neg_fingertip_object_distance(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Calculate the average distance between the object and the robot's fingertips as a reward term."""

    # get fingertip positions
    fingertip_pos = env.scene.sensors["fingertip_transforms"].data.target_pos_w
    num_finger = fingertip_pos.shape[1]
    fingertip_pos -= env.scene.env_origins.repeat((1, num_finger)).reshape(
        env.num_envs, num_finger, 3
    )

    # get obj's pose
    obj: RigidObject = env.scene["object"]
    obj_pos = obj.data.root_pos_w - env.scene.env_origins

    # calculate the average distance between fingertips and obj
    object_pos_expanded = obj_pos.unsqueeze(1)
    dists = torch.norm(fingertip_pos - object_pos_expanded, p=2, dim=-1)

    return -torch.mean(dists, dim=-1)


# TODO: check the usage of contact force.
def fingertip_object_contacts(
    env: ManagerBasedRLEnv, contact_sensor_names: list[str], threshold: float = 1e-3
) -> torch.Tensor:
    """Counts fingertip contacts with object as binary contacts summed over listed sensors.

    Uses "max mode" over the sensor's filtered force matrix: a fingertip is considered in contact if
    any filtered body contact force magnitude exceeds the threshold.
    """
    contacts = []

    for name in contact_sensor_names:
        sensor: ContactSensor = env.scene.sensors[name]

        force_matrix_w = sensor.data.force_matrix_w

        if force_matrix_w is None:
            # No filter configured / no data available.
            max_mag = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        else:
            # force_matrix_w: (num_envs, num_bodies, num_filters, 3)
            mag = torch.linalg.norm(force_matrix_w, dim=-1)
            mag = torch.nan_to_num(mag, nan=0.0)
            max_mag = mag.amax(dim=(1, 2))

        contact = (max_mag > threshold).float()
        contacts.append(contact)
    counts = torch.stack(contacts, dim=1).sum(dim=1)

    # Optional debug: print per-sensor forces when reward debugging enabled
    if getattr(env.cfg, "print_reward_terms", False):
        try:
            norms = []
            for n in contact_sensor_names:
                fm = env.scene.sensors[n].data.force_matrix_w
                if fm is None:
                    norms.append(0.0)
                else:
                    mag0 = torch.linalg.norm(fm[0], dim=-1)
                    mag0 = torch.nan_to_num(mag0, nan=0.0)
                    norms.append(mag0.amax().item())
            step = getattr(env, "common_step_counter", 0)
            print(
                f"[ContactDebug][step {step}] norms {dict(zip(contact_sensor_names, norms))}"
            )
        except Exception:
            pass
    return counts
