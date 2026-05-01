from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .commands import InHandReOrientationCommand
from .contacts import contacts, good_contact_count


def success_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Bonus reward for successfully reaching the goal.

    The object is considered to have reached the goal when the object orientation and position are within
    the command thresholds.
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
    # obtain the commanded goal position in world frame
    goal_pos_w = command_term.command[:, 0:3] + env.scene.env_origins
    # obtain the thresholds for the pose error
    orientation_threshold = command_term.cfg.orientation_success_threshold
    position_threshold = command_term.cfg.position_success_threshold
    # calculate the orientation error
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)
    position_error = torch.norm(goal_pos_w - asset.data.root_pos_w, p=2, dim=-1)
    return dtheta <= orientation_threshold
    # return (dtheta <= orientation_threshold) & (position_error < position_threshold)


def track_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    max_pos_error: float|None = None, # maximum position error to enable the reward
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
    pos_error = torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)
    if max_pos_error is not None:
        pos_error = torch.clamp(pos_error, max=max_pos_error)
    return pos_error


def track_position(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    pos_scale: float = 2.0,
    pos_temp: float = 0.5,
    max_pos_error: float | None = None,
    need_contact: bool = False,
) -> torch.Tensor:
    """Reward for tracking the object position with an exponential (Gaussian) kernel.

    reward = exp(-||active_pos||_2^2 * pos_scale / pos_temp)

    where active_pos = goal_pos_e - object_pos_e is the position error in the env frame.
    If `max_pos_error` is provided, the reward is zeroed for envs whose position error
    exceeds the threshold.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    goal_pos_e = command_term.command[:, 0:3]
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    active_pos = goal_pos_e - object_pos_e

    pos_error = torch.norm(active_pos, p=2, dim=-1)
    reward = torch.exp(-(pos_error ** 2) * pos_scale / pos_temp)

    if max_pos_error is not None:
        reward = torch.where(pos_error <= max_pos_error, reward, torch.zeros_like(reward))

    if need_contact:
        reward = reward * contacts(env, 0.3, mode="any")

    return reward


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
    need_contact: bool = False,
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
    value = 1.0 / (dtheta + rot_eps)
    if need_contact:
        value = value * contacts(env, 0.3, mode="any")
    return value


def track_orientation(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_scale: float = 5.0,
    rot_temp: float = 1.0,
    need_contact: bool = False,
) -> torch.Tensor:
    """Reward for tracking the object orientation with an exponential (Gaussian) kernel.

    reward = exp(-||active_quat_vec||_2^2 * rot_scale / rot_temp)

    where active_quat = goal_quat * conj(object_quat) is the relative rotation and
    active_quat_vec is its vector (imaginary) part. For a rotation of angle theta,
    ||active_quat_vec||_2 = |sin(theta/2)|, so the reward is 1 at alignment and decays
    smoothly with orientation error.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    goal_quat_w = command_term.command[:, 3:7]
    object_quat_w = asset.data.root_quat_w

    active_quat = math_utils.quat_mul(goal_quat_w, math_utils.quat_conjugate(object_quat_w))
    # isaaclab quaternions are wxyz; the vector part is components [1:4]
    active_quat_vec = active_quat[..., 1:4]

    reward = torch.exp(-torch.norm(active_quat_vec, p=2, dim=-1) ** 2 * rot_scale / rot_temp)

    if need_contact:
        reward = reward * contacts(env, 0.3, mode="any")

    return reward


def track_orientation_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_scale: float = 1.0,
    rot_temp: float = 1.0,
    need_contact: bool = False,
) -> torch.Tensor:
    """Reward for tracking object orientation using exp kernel on geodesic angle.

    reward = exp(-dtheta^2 * rot_scale / rot_temp)

    where dtheta is the geodesic angle error in radians from quat_error_magnitude.
    Unlike track_orientation which uses sin^2(theta/2), this uses theta^2 directly,
    giving stronger gradients for large misalignments.
    """
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    goal_quat_w = command_term.command[:, 3:7]
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)
    reward = torch.exp(-dtheta * rot_scale / rot_temp)

    if need_contact:
        reward = reward * contacts(env, 0.3, mode="any")

    return reward


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


def joint_pos_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared deviation from the robot's default joint pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos - asset.data.default_joint_pos
    return torch.sum(torch.square(joint_error), dim=1)


def joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize instantaneous mechanical power (sum of |torque * velocity|) to encourage energy-saving behavior."""
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(robot.data.applied_torque * robot.data.joint_vel), dim=1)


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


###############################
#### Good-Contact Metrics  ####
###############################

def good_contact_reward(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
    contact_pose_range_deg: float = 50.0,
    contact_scale: float = 1.0,
    contact_temp: float = 1.0,
) -> torch.Tensor:
    """tanh-normalised good-contact reward: tanh(n_good / n_tips * scale / temp)."""
    n_tips = float(len(contact_sensor_names))
    n_good = good_contact_count(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, contact_pose_range_deg
    )
    return torch.tanh(n_good / n_tips * contact_scale / contact_temp)
