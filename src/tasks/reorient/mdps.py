from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform


import torch
import numpy as np
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

###########################
###### Command Term #######
###########################


# THIS IS DIFFERENT FROM THE OFFICIAL ONE AS THE ORIENTATION IS RESAMPLED AROUND THE CURRENT OBJECT POSE, NOT THE DEFAULT POSE.
class InHandReOrientationCommand(CommandTerm):
    """Command term that generates 3D pose commands for in-hand manipulation task.

    This command term generates 3D orientation commands for the object. The orientation commands
    are sampled uniformly from the 3D orientation space. The position commands are the default
    root state of the object.

    The constant position commands is to encourage that the object does not move during the task.
    For instance, the object should not fall off the robot's palm.

    Unlike typical command terms, where the goals are resampled based on time, this command term
    does not resample the goals based on time. Instead, the goals are resampled when the object
    reaches the goal orientation. The goal orientation is considered to be reached when the
    orientation error is below a certain threshold.
    """

    cfg: InHandReOrientationCommandCfg
    """Configuration for the command term."""

    def __init__(self, cfg: InHandReOrientationCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command term class.

        Args:
            cfg: The configuration parameters for the command term.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # object
        self.object: RigidObject = env.scene[cfg.asset_name]

        # create buffers to store the command
        # -- command: (x, y, z)
        self.init_pos_offset = torch.tensor(
            cfg.init_pos_offset, dtype=torch.float, device=self.device
        )
        self.pos_command_w = self.object.data.root_pos_w.clone() + self.init_pos_offset
        self.pos_command_e = self.pos_command_w - self._env.scene.env_origins

        # -- orientation: (w, x, y, z)
        self.quat_command_w = self.object.data.root_quat_w.clone()

        # -- unit vectors
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )

        # -- metrics
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self._goal_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._new_success_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.random_range = cfg.random_range

    def __str__(self) -> str:
        msg = "InHandManipulationCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # -- compute the orientation error
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(
            self.object.data.root_pos_w - self.pos_command_w, dim=1
        )
        successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        if self.cfg.use_position_success:
            successes = successes & (self.metrics["position_error"] < self.cfg.position_success_threshold)
        # only count success once per goal command
        new_success = successes & ~self._goal_succeeded
        self._new_success_this_step = new_success  # saved before _goal_succeeded update for _update_command
        self.metrics["consecutive_success"] += new_success.float()
        # used for consecutive_success, only record one success per goal command.
        self._goal_succeeded |= successes

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        idx: slice | Sequence[int] = slice(None) if env_ids is None else env_ids
        self.pos_command_w[idx] = self.object.data.root_pos_w[idx] + self.init_pos_offset
        self.pos_command_e[idx] = self.pos_command_w[idx] - self._env.scene.env_origins[idx]
        self.quat_command_w[idx] = self.object.data.root_quat_w[idx]
        self._hold_counter[idx] = 0
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        self._goal_succeeded[env_ids] = False
        self._hold_counter[env_ids] = 0
        n = len(env_ids)
        min_rad, max_rad = self.random_range
        # approximaly uniformally sample from SO3 around the current command pose
        # Haar-correct: p(θ) ∝ sin²(θ/2); CDF = θ/2 - sin(θ)/2
        lo = 0.5 * (min_rad - _math.sin(min_rad))
        hi = 0.5 * (max_rad - _math.sin(max_rad))
        u = torch.rand((n,), device=self.device) * (hi - lo) + lo
        theta = (12 * u).clamp(min=1e-8).pow(1 / 3)  # cubic-root init
        f = 0.5 * (theta - torch.sin(theta)) - u
        df = 0.5 * (1 - torch.cos(theta))
        angle = (theta - f / df.clamp(min=1e-6)).clamp(min_rad, max_rad)
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        quat_delta = math_utils.quat_from_angle_axis(angle, axis)

        # apply delta to the current object orientation so new goals stay within
        # random_range of the actual object state (not the previous command, which
        # may be unachieved in time-based resampling).
        init_quat = self.object.data.root_quat_w[env_ids]
        quat = math_utils.quat_mul(init_quat, quat_delta)

        self.quat_command_w[env_ids] = (
            math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat
        )

    def _update_command(self):
        if self.cfg.resample_on == "success":
            successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
            if self.cfg.use_position_success:
                successes = successes & (
                    self.metrics["position_error"] < self.cfg.position_success_threshold
                )

            hold_steps = self.cfg.hold_steps_on_success
            prev_holding = self._hold_counter > 0

            # Decrement hold counters for envs currently in hold phase
            self._hold_counter[prev_holding] -= 1

            # Envs whose hold just expired → resample now
            hold_done = prev_holding & (self._hold_counter == 0)

            if hold_steps > 0:
                # _new_success_this_step was captured in _update_metrics() before _goal_succeeded
                # was updated — this is the only reliable way to detect the first-success edge,
                # since _goal_succeeded is already True by the time _update_command() runs.
                new_first_success = self._new_success_this_step & ~prev_holding
                self._hold_counter[new_first_success] = hold_steps
                # all other successes (post-reset stale case, or goal lost/regained) → immediate resample
                immediate_resample = successes & ~prev_holding & ~new_first_success
            else:
                # hold_steps == 0: original behavior — immediate resample on any success
                immediate_resample = successes & ~prev_holding

            resample_ids = (hold_done | immediate_resample).nonzero(as_tuple=False).squeeze(-1)
            if len(resample_ids) > 0:
                self._resample(resample_ids)
        # "time" mode: base-class timer in compute() calls _resample() automatically

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
            if not hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
            # set visibility
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Goal pose visualization
        # add an offset to the marker position to visualize the goal
        marker_pos = self.pos_command_w + torch.tensor(
            self.cfg.marker_pos_offset, device=self.device
        )
        marker_quat = self.quat_command_w
        # visualize the goal marker
        self.goal_pose_visualizer.visualize(
            translations=marker_pos, orientations=marker_quat
        )

        # Current object pose visualization
        # visualize at actual object position
        current_pos = self.object.data.root_pos_w
        current_quat = self.object.data.root_quat_w
        self.current_pose_visualizer.visualize(
            translations=current_pos, orientations=current_quat
        )


@configclass
class InHandReOrientationCommandCfg(CommandTermCfg):
    """Configuration for the uniform 3D orientation command term.

    Please refer to the :class:`InHandReOrientationCommand` class for more details.
    """

    random_range: tuple[float, float] = (0.0, 1.0)
    """Min/max geodesic angle in radians for goal orientation sampling."""

    class_type: type = InHandReOrientationCommand
    resampling_time_range: tuple[float, float] = (
        1e6,
        1e6,
    )  # no resampling based on time

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the asset from its default position.

    This is used to account for the offset typically present in the object's default position
    so that the object is spawned at a height above the robot's palm. When the position command
    is generated, the object's default position is used as the reference and the offset specified
    is added to it to get the desired position of the object.
    """

    make_quat_unique: bool = MISSING
    """Whether to make the quaternion unique or not.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    orientation_success_threshold: float = MISSING
    """Threshold for the orientation error to consider the goal orientation to be reached."""

    use_position_success: bool = False
    """Whether to also require position error below threshold for success. Defaults to False."""

    position_success_threshold: float = 0.05
    """Threshold for the position error (m) when use_position_success is True."""

    hold_steps_on_success: int = 0
    """Number of timesteps to hold at the goal after first success before resampling.
    0 (default) = immediate resample on success (original behavior)."""

    resample_on: Literal["success", "time"] = "success"
    """When to resample the goal command.

    - ``"success"``: resample when the object reaches the goal (orientation error, and optionally
      position error, fall below their respective thresholds).
    - ``"time"``: resample on a timer; set :attr:`resampling_time_range` to the desired interval.
      The base-class timer handles resampling automatically.
    """

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset of the marker from the object's desired position.

    This is useful to position the marker at a height above the object's desired position.
    Otherwise, the marker may occlude the object in the visualization.
    """

    # Goal pose visualization - XYZ frame axes (5cm scale)
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),  # ~5cm axes
            ),
        },
    )
    """The configuration for the goal pose visualization marker. Defaults to XYZ axes (5cm)."""

    # Current object pose visualization - XYZ frame axes (5cm scale)
    current_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_marker",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),  # ~5cm axes
            ),
        },
    )
    """The configuration for the current object pose visualization marker."""

###########################
###### Reward Term #######
###########################

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

from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from src.tasks.common.obj_point_cloud import sample_object_point_cloud
# from src.tasks.reorient.curriculum import CurriculumCfg


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

import math as _math

_FINGERTIP_SENSOR_NAMES = [
    "thumb_tip_object_s",
    "index_tip_object_s",
    "middle_tip_object_s",
    "ring_tip_object_s",
]


def _compute_contact_metrics(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str,
    force_threshold: float,
    contact_pose_range_deg: float,
) -> dict[str, torch.Tensor]:
    """Shared layer: per-fingertip forces rotated into fingertip frame, theta/phi, contact masks.

    Returns dict with keys:
      in_contact   (num_envs, n_fingers) bool
      force_mag    (num_envs, n_fingers) float
      contact_pose (num_envs, n_fingers, 2) float  — [theta, phi] clamped to pose_limit
    """
    from isaaclab.utils.math import quat_apply_inverse

    pose_limit = _math.radians(contact_pose_range_deg)
    # (num_envs, n_fingers, 4) world-frame fingertip orientations as (w, x, y, z)
    target_quat_w = env.scene.sensors[fingertip_transforms_name].data.target_quat_w

    in_contact_list, force_mag_list, pose_list = [], [], []

    for i, name in enumerate(contact_sensor_names):
        sensor = env.scene.sensors[name]
        fm = sensor.data.force_matrix_w

        if fm is None or fm.numel() == 0:
            zeros = torch.zeros(env.num_envs, device=env.device)
            in_contact_list.append(zeros.bool())
            force_mag_list.append(zeros)
            pose_list.append(torch.zeros(env.num_envs, 2, device=env.device))
            continue

        F_world = torch.nan_to_num(fm, nan=0.0).sum(dim=(1, 2))        # (N, 3)
        F_mag = torch.linalg.norm(F_world, dim=-1)                      # (N,)
        in_contact = F_mag > force_threshold

        # Rotate reaction force into fingertip frame.
        # Frame convention: red (X) = object-to-finger (reaction direction),
        # so theta=0, phi=0 at centered contact without negation.
        F_tip = quat_apply_inverse(target_quat_w[:, i, :], F_world)  # (N, 3)
        F_tip_mag = F_mag  # rotation preserves norm

        # Spherical coordinates in fingertip frame: theta=0, phi=0 when
        # reaction is along +X (red axis = fingertip outward normal).
        theta = torch.where(
            F_tip_mag < force_threshold,
            torch.zeros_like(F_tip_mag),
            torch.atan2(F_tip[:, 1], F_tip[:, 0]),
        )
        phi = torch.where(
            F_tip_mag < force_threshold,
            torch.zeros_like(F_tip_mag),
            torch.acos(torch.clamp(F_tip[:, 2] / (F_tip_mag + 1e-8), -1.0, 1.0)) - torch.pi / 2,
        )

        pose = torch.clamp(torch.stack([theta, phi], dim=-1), -pose_limit, pose_limit)

        in_contact_list.append(in_contact)
        force_mag_list.append(F_mag)
        pose_list.append(pose)

    return {
        "in_contact":   torch.stack(in_contact_list, dim=1),   # (N, n)
        "force_mag":    torch.stack(force_mag_list, dim=1),     # (N, n)
        "contact_pose": torch.stack(pose_list, dim=1),          # (N, n, 2)
    }


def good_contact_count(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
    contact_pose_range_deg: float = 50.0,
) -> torch.Tensor:
    """Count fingertips whose contact force is above threshold AND within the angular pose limit."""
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, contact_pose_range_deg
    )
    good_pose = (torch.abs(m["contact_pose"]) < _math.radians(contact_pose_range_deg)).all(dim=-1)
    return (m["in_contact"] & good_pose).float().sum(dim=-1)    # (N,)


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


###############################
###### Termination Term #######
###############################

def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int, command_name: str) -> torch.Tensor:
    """Check if the task has been completed consecutively for a certain number of times.

    Args:
        env: The environment object.
        num_success: Threshold for the number of consecutive successes required.
        command_name: The command term to be used for extracting the goal.
    """
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    return command_term.metrics["consecutive_success"] >= num_success


def object_away_from_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the goal.

    The object is considered to be out-of-reach if the distance between the goal and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    # extract useful elements
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)
    asset = env.scene[object_cfg.name]

    # object pos
    asset_pos_e = asset.data.root_pos_w - env.scene.env_origins
    goal_pos_e = command_term.command[:, :3]

    return torch.norm(asset_pos_e - goal_pos_e, p=2, dim=1) > threshold


def object_away_from_robot(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Check if object has gone far from the robot.

    The object is considered to be out-of-reach if the distance between the robot and the object is greater
    than the threshold.

    Args:
        env: The environment object.
        threshold: The threshold for the distance between the robot and the object.
        asset_cfg: The configuration for the robot entity. Default is "robot".
        object_cfg: The configuration for the object entity. Default is "object".
    """
    # extract useful elements
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]

    # compute distance
    dist = torch.norm(robot.data.root_pos_w - object.data.root_pos_w, dim=1)

    return dist > threshold


###########################
####### Event Term ########
###########################

class reset_joints_within_limits_range(ManagerTermBase):
    """Reset an articulation's joints to a random position in the given limit ranges.

    This function samples random values for the joint position and velocities from the given limit ranges.
    The values are then set into the physics simulation.

    The parameters to the function are:

    * :attr:`position_range` - a dictionary of position ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each joint. The keys of the dictionary are the
      joint names (or regular expressions) of the asset.
    * :attr:`use_default_offset` - a boolean flag to indicate if the ranges are offset by the default joint state.
      Defaults to False.
    * :attr:`asset_cfg` - the configuration of the asset to reset. Defaults to the entity named "robot" in the scene.
    * :attr:`operation` - whether the ranges are scaled values of the joint limits, or absolute limits.
       Defaults to "abs".

    The dictionary values are a tuple of the form ``(a, b)``. Based on the operation, these values are
    interpreted differently:

    * If the operation is "abs", the values are the absolute minimum and maximum values for the joint, i.e.
      the joint range becomes ``[a, b]``.
    * If the operation is "scale", the values are the scaling factors for the joint limits, i.e. the joint range
      becomes ``[a * min_joint_limit, b * max_joint_limit]``.

    If the ``a`` or the ``b`` value is ``None``, the joint limits are used instead.

    Note:
        If the dictionary does not contain a key, the joint position or joint velocity is set to the default value for
        that joint.

    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # check if the cfg has the required parameters
        if "position_range" not in cfg.params or "velocity_range" not in cfg.params:
            raise ValueError(
                "The term 'reset_joints_within_range' requires parameters: 'position_range' and 'velocity_range'."
                f" Received: {list(cfg.params.keys())}."
            )

        # parse the parameters
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        use_default_offset = cfg.params.get("use_default_offset", False)
        operation = cfg.params.get("operation", "abs")
        # check if the operation is valid
        if operation not in ["abs", "scale"]:
            raise ValueError(
                f"For event 'reset_joints_within_limits_range', unknown operation: '{operation}'."
                " Please use 'abs' or 'scale'."
            )

        # extract the used quantities (to enable type-hinting)
        self._asset: Articulation = env.scene[asset_cfg.name]
        default_joint_pos = self._asset.data.default_joint_pos[0]
        default_joint_vel = self._asset.data.default_joint_vel[0]

        # create buffers to store the joint position range
        self._pos_ranges = self._asset.data.soft_joint_pos_limits[0].clone()
        # parse joint position ranges
        pos_joint_ids = []
        for joint_name, joint_range in cfg.params["position_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            pos_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] *= joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] *= joint_range[1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint position ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._pos_ranges[joint_ids] += default_joint_pos[joint_ids].unsqueeze(1)

        # store the joint pos ids (used later to sample the joint positions)
        self._pos_joint_ids = torch.tensor(
            pos_joint_ids, device=self._pos_ranges.device
        )
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

        # create buffers to store the joint velocity range
        self._vel_ranges = torch.stack(
            [
                -self._asset.data.soft_joint_vel_limits[0],
                self._asset.data.soft_joint_vel_limits[0],
            ],
            dim=1,
        )
        # parse joint velocity ranges
        vel_joint_ids = []
        for joint_name, joint_range in cfg.params["velocity_range"].items():
            # find the joint ids
            joint_ids = self._asset.find_joints(joint_name)[0]
            vel_joint_ids.extend(joint_ids)

            # set the joint position ranges based on the given values
            if operation == "abs":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = (
                        joint_range[0] * self._vel_ranges[joint_ids, 0]
                    )
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = (
                        joint_range[1] * self._vel_ranges[joint_ids, 1]
                    )
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint velocity ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)

        # store the joint vel ids (used later to sample the joint positions)
        self._vel_joint_ids = torch.tensor(
            vel_joint_ids, device=self._vel_ranges.device
        )
        self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict[str, tuple[float | None, float | None]],
        velocity_range: dict[str, tuple[float | None, float | None]],
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        operation: Literal["abs", "scale"] = "abs",
    ):
        # get default joint state
        joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
        joint_vel = self._asset.data.default_joint_vel[env_ids].clone()

        # sample random joint positions for each joint
        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (len(env_ids), len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = sample_uniform(
                self._pos_ranges[:, 0],
                self._pos_ranges[:, 1],
                joint_pos_shape,
                device=joint_pos.device,
            )
            # clip the joint positions to the joint limits
            joint_pos_limits = self._asset.data.soft_joint_pos_limits[
                0, self._pos_joint_ids
            ]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        # sample random joint velocities for each joint
        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = sample_uniform(
                self._vel_ranges[:, 0],
                self._vel_ranges[:, 1],
                joint_vel_shape,
                device=joint_vel.device,
            )
            # clip the joint velocities to the joint limits
            joint_vel_limits = self._asset.data.soft_joint_vel_limits[
                0, self._vel_joint_ids
            ]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class randomize_hand_object_default_pose(ManagerTermBase):
    """Randomize the robot root orientation and rotate the object's default pose with it."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        base_asset_cfg: SceneEntityCfg = cfg.params.get(
            "base_asset_cfg", SceneEntityCfg("robot")
        )
        object_asset_cfg: SceneEntityCfg = cfg.params.get(
            "object_asset_cfg", SceneEntityCfg("object")
        )
        self._robot: Articulation = env.scene[base_asset_cfg.name]
        self._object: RigidObject = env.scene[object_asset_cfg.name]

        if base_asset_cfg.body_names is not None:
            body_ids, _ = self._robot.find_bodies(base_asset_cfg.body_names)
            self._base_body_id: int = body_ids[0]
        else:
            self._base_body_id = 0 # default to the root body

        self.roll_range = cfg.params.get("roll_range", (-torch.pi, torch.pi))
        self.pitch_range = cfg.params.get("pitch_range", (-torch.pi, torch.pi))
        self.yaw_range = cfg.params.get("yaw_range", (-torch.pi, torch.pi))

        # Store original unrotated default states so resets don't accumulate rotations.
        self._orig_robot_default = self._robot.data.default_root_state.clone()
        self._orig_object_default = self._object.data.default_root_state.clone()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        base_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base*"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        roll_range: tuple[float, float] | None = None,
        pitch_range: tuple[float, float] | None = None,
        yaw_range: tuple[float, float] | None = None,
    ):
        if env_ids is None or env_ids == slice(None):
            env_ids = torch.arange(env.num_envs, device=env.device)

        roll_range = self.roll_range if roll_range is None else roll_range
        pitch_range = self.pitch_range if pitch_range is None else pitch_range
        yaw_range = self.yaw_range if yaw_range is None else yaw_range

        roll = torch.empty(len(env_ids), device=env.device).uniform_(*roll_range)
        pitch = torch.empty(len(env_ids), device=env.device).uniform_(*pitch_range)
        yaw = torch.empty(len(env_ids), device=env.device).uniform_(*yaw_range)
        quat_delta = quat_from_euler_xyz(roll, pitch, yaw)

        env_origins = env.scene.env_origins[env_ids]

        # Always rotate relative to the original unrotated defaults to avoid accumulation.
        robot_states = self._orig_robot_default[env_ids].clone()
        object_states = self._orig_object_default[env_ids].clone()

        # Rigid rotation around the robot root (env frame).
        # body_pos_w is unreliable at startup (not yet initialized), so we use
        # default_root_state which is always valid from config.
        pivot = robot_states[:, :3].clone()
        robot_states[:, 3:7] = math_utils.quat_mul(quat_delta, robot_states[:, 3:7])
        object_states[:, :3] = pivot + quat_apply(quat_delta, object_states[:, :3] - pivot)
        object_states[:, 3:7] = math_utils.quat_mul(quat_delta, object_states[:, 3:7])

        self._robot.data.default_root_state[env_ids] = robot_states
        self._object.data.default_root_state[env_ids] = object_states

        robot_root_states_w = robot_states.clone()
        object_root_states_w = object_states.clone()
        robot_root_states_w[:, :3] += env_origins
        object_root_states_w[:, :3] += env_origins

        self._robot.write_root_state_to_sim(robot_root_states_w, env_ids=env_ids)
        self._object.write_root_state_to_sim(object_root_states_w, env_ids=env_ids)


def _normalize_env_ids(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | Sequence[int] | slice | None
) -> torch.Tensor:
    """Normalize environment ids to a tensor on the env device."""
    if env_ids is None or env_ids == slice(None):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        start = 0 if env_ids.start is None else env_ids.start
        stop = env.num_envs if env_ids.stop is None else env_ids.stop
        step = 1 if env_ids.step is None else env_ids.step
        return torch.arange(start, stop, step, device=env.device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def contacts(env: ManagerBasedRLEnv, threshold: float, mode: Literal["opposite", "any"] = "opposite") -> torch.Tensor:
    """Check if the fingertip contacts with the object is above a threshold."""
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_tip_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_tip_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_tip_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_tip_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    if mode == "opposite":
        contact_cond = (thumb_contact_mag > threshold) & (
            (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
        )
    if mode == "any":
        finger_contacts = torch.stack([
            thumb_contact_mag > threshold,
            index_contact_mag > threshold,
            middle_contact_mag > threshold,
            ring_contact_mag > threshold,
        ], dim=-1)  # (num_envs, 4)
        contact_cond = finger_contacts.sum(dim=-1) >= 2
    return contact_cond


class apply_gravity_compensation_assist(ManagerTermBase):
    """Gravity-compensation assist that decays per step when good contact is detected.

    _assist_scale starts at 1.0 each episode and is multiplied by decay_ratio for each env
    where good_contact_count >= min_good_contacts. It is decay-only (never increases mid-episode).

    Good contact requires both force magnitude > contact_threshold AND force direction within
    contact_pose_range_deg of the fingertip normal (force-direction quality check).
    """

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._assist_scale = torch.ones(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self._assist_scale[:] = 1.0
        else:
            self._assist_scale[env_ids] = 1.0

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        contact_threshold: float = 1.0,
        contact_pose_range_deg: float = 50.0,
        min_good_contacts: int = 2,
        decay_ratio: float = 0.95,  # 0.95^120 (1s) ~ 0, so decays to 0 in ~1s
    ) -> None:
        env_ids_t = _normalize_env_ids(env, env_ids)
        obj: RigidObject = env.scene[asset_cfg.name]

        # Current physics gravity — live from PhysX after variable_gravity applied.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        grav_raw = physics_sim_view.get_gravity()

        gravity_vec = torch.tensor(
            [grav_raw[0], grav_raw[1], grav_raw[2]], device=env.device, dtype=torch.float32
        )
        # Actual per-env mass; get_masses() returns CPU tensor.
        masses = obj.root_physx_view.get_masses().to(env.device)  # (num_envs, 1)
        mass = masses[env_ids_t, 0]                               # (n,)

        # Decay wherever force-direction-quality good contact count >= min_good_contacts.
        # contact = (
        #     good_contact_count(env, _FINGERTIP_SENSOR_NAMES,
        #                        force_threshold=contact_threshold,
        #                        contact_pose_range_deg=contact_pose_range_deg)
        #     >= min_good_contacts
        # )                                                          # (num_envs,) bool
        contact = contacts(env, threshold=contact_threshold, mode='any')
        self._assist_scale[env_ids_t] = torch.where(
            contact[env_ids_t],
            (self._assist_scale[env_ids_t] * decay_ratio).round(decimals=4),
            self._assist_scale[env_ids_t],
        )

        # Apply force; shape (n, 1, 3) required by set_external_force_and_torque.
        scale = self._assist_scale[env_ids_t]   
        force_vec = -gravity_vec.unsqueeze(0) * (mass * scale).unsqueeze(-1)
        assist_force = force_vec.unsqueeze(1)
        torques = torch.zeros_like(assist_force)

        obj.set_external_force_and_torque(
            assist_force, torques, env_ids=env_ids_t, is_global=True
        )


class reset_root_state_from_pose(ManagerTermBase):
    """Reset an asset root state to a fixed pose and velocity."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset = env.scene[asset_cfg.name]

        pose = cfg.params.get("pose")
        if pose is None or len(pose) != 7:
            raise ValueError(
                "reset_root_state_from_pose requires 'pose' as (x, y, z, w, x, y, z)."
            )

        velocity = cfg.params.get("velocity", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        if len(velocity) != 6:
            raise ValueError(
                "reset_root_state_from_pose requires 'velocity' as (vx, vy, vz, wx, wy, wz)."
            )

        self._pose = torch.tensor(pose, dtype=torch.float32, device=env.device)
        self._velocity = torch.tensor(velocity, dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        pose: tuple[float, float, float, float, float, float, float] | None = None,
        velocity: tuple[float, float, float, float, float, float] | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        del asset_cfg
        env_ids_t = _normalize_env_ids(env, env_ids)
        pose_tensor = self._pose if pose is None else torch.tensor(pose, dtype=torch.float32, device=env.device)
        vel_tensor = (
            self._velocity if velocity is None else torch.tensor(velocity, dtype=torch.float32, device=env.device)
        )

        root_states = self._asset.data.default_root_state[env_ids_t].clone()
        root_states[:, 0:3] = pose_tensor[0:3] + env.scene.env_origins[env_ids_t]
        root_states[:, 3:7] = pose_tensor[3:7]
        root_states[:, 7:10] = vel_tensor[0:3]
        root_states[:, 10:13] = vel_tensor[3:6]
        self._asset.write_root_state_to_sim(root_states, env_ids=env_ids_t)


class reset_joints_around_default(ManagerTermBase):
    """Sample candidate joint positions as default plus uniform per-joint deltas."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._joint_reset_delta = float(cfg.params.get("joint_reset_delta", 0.25))

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        joint_reset_delta: float | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        del asset_cfg
        env_ids_t = _normalize_env_ids(env, env_ids)
        delta = self._joint_reset_delta if joint_reset_delta is None else float(joint_reset_delta)

        joint_pos = self._asset.data.default_joint_pos[env_ids_t].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos += delta * (2.0 * torch.rand_like(joint_pos) - 1.0)

        joint_limits = self._asset.data.soft_joint_pos_limits[env_ids_t]
        joint_pos = torch.clamp(joint_pos, joint_limits[..., 0], joint_limits[..., 1])
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)