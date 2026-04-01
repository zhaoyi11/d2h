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
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

###########################
###### Command Term #######
###########################

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
        init_pos_offset = torch.tensor(
            cfg.init_pos_offset, dtype=torch.float, device=self.device
        )
        self.pos_command_e = (
            self.object.data.default_root_state[:, :3] + init_pos_offset
        )
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins

        # -- orientation: (w, x, y, z)
        # self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        # self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0
        self.quat_command_w = self.object.data.default_root_state[:, 3:7]

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
        # -- compute the number of consecutive successes
        successes = (
            self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        )
        self.metrics["consecutive_success"] += successes.float()

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new orientation targets
        # range in pi (i.e. 0.25 * pi = 45 deg)
        # TODO: set up a curriculum for the range (e.g., 0.1->0.25->0.5->1.0)
        r_range = self.random_range
        # if r_range > 0.1:
        #     print("r_range", r_range)

        rand_floats = 2.0 * torch.rand((len(env_ids), 3), device=self.device) - 1.0
        # rotate randomly about x-axis, y-axis, and z-axis with small angles
        quat_delta = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(
                rand_floats[:, 0] * r_range * torch.pi, self._X_UNIT_VEC[env_ids]
            ),
            math_utils.quat_mul(
                math_utils.quat_from_angle_axis(
                    rand_floats[:, 1] * r_range * torch.pi, self._Y_UNIT_VEC[env_ids]
                ),
                math_utils.quat_from_angle_axis(
                    rand_floats[:, 2] * r_range * torch.pi, self._Z_UNIT_VEC[env_ids]
                ),
            ),
        )

        # apply delta to the default orientation
        init_quat = self.object.data.default_root_state[env_ids, 3:7]

        quat = math_utils.quat_mul(init_quat, quat_delta)

        # make sure the quaternion real-part is always positive
        self.quat_command_w[env_ids] = (
            math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat
        )

    def _update_command(self):
        # update the command if goal is reached
        if self.cfg.update_goal_on_success:
            # compute the goal resets
            goal_resets = (
                self.metrics["orientation_error"]
                < self.cfg.orientation_success_threshold
            )
            goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            # resample the goals
            self._resample(goal_reset_ids)

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

    random_range: float = 1.0  # TODO: tune this one, and set up a curriculum for the range (e.g., 0.1->0.25->0.5->1.0)
    """Range for the random orientation."""

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

    update_goal_on_success: bool = MISSING
    """Whether to update the goal orientation when the goal orientation is reached."""

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
from src.tasks.reorient.curriculum import CurriculumCfg


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

def gravity_enabled(
    env: ManagerBasedRLEnv, 
    gravity_eps: float = 1e-6) -> torch.Tensor:
    """Check if gravity is enabled.
    """
    gravity = torch.tensor(
        sim_utils.SimulationContext.instance().physics_sim_view.get_gravity(),
        device=env.device,
        dtype=torch.float32,
    )
    return torch.linalg.vector_norm(gravity) > gravity_eps

def track_pos_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    gravity_eps: float = 1e-6,
    max_pos_error: float|None = None, # maximum position error to enable the reward
    use_gravity_gate: bool = True,
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
    if use_gravity_gate and not gravity_enabled(env, gravity_eps):
        return torch.zeros(env.num_envs, device=env.device, dtype=asset.data.root_quat_w.dtype)

    # obtain the goal position
    goal_pos_e = command_term.command[:, 0:3]
    # obtain the object position in the environment frame
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    pos_error = torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)
    if max_pos_error is not None:
        pos_error = torch.clamp(pos_error, max=max_pos_error)
    return pos_error


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    rot_eps: float = 1e-3,
    gravity_eps: float = 1e-6,
    use_gravity_gate: bool = True,
) -> torch.Tensor:
    """Reward for tracking the object orientation using the inverse of the orientation error.

    The reward is the inverse of the orientation error between the object orientation and the goal orientation.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        object_cfg: The configuration for the scene entity. Default is "object".
        rot_eps: The threshold for the orientation error. Default is 1e-3.
        gravity_eps: Minimum gravity magnitude required to enable the reward.
    """
    # extract useful elements
    asset: RigidObject = env.scene[object_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(
        command_name
    )
    
    if use_gravity_gate and not gravity_enabled(env, gravity_eps):
        return torch.zeros(env.num_envs, device=env.device, dtype=asset.data.root_quat_w.dtype)
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


def joint_pos_default_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize squared deviation from the robot's default joint pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_error = asset.data.joint_pos - asset.data.default_joint_pos
    return torch.sum(torch.square(joint_error), dim=1)


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

        self._roll_range = cfg.params.get("roll_range", (-torch.pi, torch.pi))
        self._pitch_range = cfg.params.get("pitch_range", (-torch.pi, torch.pi))
        self._yaw_range = cfg.params.get("yaw_range", (-torch.pi, torch.pi))

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

        roll_range = self._roll_range if roll_range is None else roll_range
        pitch_range = self._pitch_range if pitch_range is None else pitch_range
        yaw_range = self._yaw_range if yaw_range is None else yaw_range

        roll = torch.empty(len(env_ids), device=env.device).uniform_(*roll_range)
        pitch = torch.empty(len(env_ids), device=env.device).uniform_(*pitch_range)
        yaw = torch.empty(len(env_ids), device=env.device).uniform_(*yaw_range)
        quat_delta = quat_from_euler_xyz(roll, pitch, yaw)

        env_origins = env.scene.env_origins[env_ids]

        robot_states = self._robot.data.default_root_state[env_ids].clone()
        object_states = self._object.data.default_root_state[env_ids].clone()

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


def _contact_sensor_max_magnitude(sensor, num_envs: int, device: torch.device) -> torch.Tensor:
    """Return the max contact force magnitude for a contact sensor."""
    force_matrix_w = getattr(sensor.data, "force_matrix_w", None)
    if force_matrix_w is not None:
        magnitudes = torch.linalg.norm(force_matrix_w, dim=-1)
    else:
        net_forces_w = getattr(sensor.data, "net_forces_w", None)
        if net_forces_w is None:
            return torch.zeros(num_envs, device=device, dtype=torch.float32)
        magnitudes = torch.linalg.norm(net_forces_w, dim=-1)

    magnitudes = torch.nan_to_num(magnitudes, nan=0.0)
    if magnitudes.ndim == 1:
        return magnitudes
    reduce_dims = tuple(range(1, magnitudes.ndim))
    return magnitudes.amax(dim=reduce_dims)


def stable_grasp_invalid(
    env: ManagerBasedRLEnv,
    tip_contact_sensor_names: list[str],
    required_tip_contact_sensor_names: list[str],
    non_tip_contact_sensor_names: list[str] | None = None,
    tip_contact_force_threshold: float = 0.25,
    min_tip_contacts: int = 2,
    fingertip_frame_sensor_name: str = "fingertip_transforms",
    fingertip_distance_sum_limit: float = 0.25,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return a boolean mask for environments that fail the stable-grasp filter.

    This mirrors the effective ``smg_gym`` stable-grasp rejection logic:
    keep only states with enough fingertip contacts, require thumb and middle contacts,
    reject any non-tip contact, and reject fingertip layouts that are too far from the object.
    """
    tip_contact_flags = []
    for sensor_name in tip_contact_sensor_names:
        sensor = env.scene.sensors[sensor_name]
        max_mag = _contact_sensor_max_magnitude(sensor, env.num_envs, env.device)
        tip_contact_flags.append(max_mag > tip_contact_force_threshold)
    tip_contact_flags = torch.stack(tip_contact_flags, dim=1)

    insufficient_tip_contacts = tip_contact_flags.sum(dim=1) < min_tip_contacts

    if required_tip_contact_sensor_names:
        required_flags = []
        for sensor_name in required_tip_contact_sensor_names:
            sensor = env.scene.sensors[sensor_name]
            max_mag = _contact_sensor_max_magnitude(sensor, env.num_envs, env.device)
            required_flags.append(max_mag > tip_contact_force_threshold)
        missing_required_contacts = ~torch.stack(required_flags, dim=1).all(dim=1)
    else:
        missing_required_contacts = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    if non_tip_contact_sensor_names:
        non_tip_flags = []
        for sensor_name in non_tip_contact_sensor_names:
            sensor = env.scene.sensors[sensor_name]
            max_mag = _contact_sensor_max_magnitude(sensor, env.num_envs, env.device)
            non_tip_flags.append(max_mag > tip_contact_force_threshold)
        has_non_tip_contact = torch.stack(non_tip_flags, dim=1).any(dim=1)
    else:
        has_non_tip_contact = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    fingertip_sensor = env.scene.sensors[fingertip_frame_sensor_name]
    fingertip_pos_w = fingertip_sensor.data.target_pos_w
    object_pos_w = env.scene[object_cfg.name].data.root_pos_w
    fingertip_distance_sum = torch.linalg.norm(
        fingertip_pos_w - object_pos_w.unsqueeze(1), dim=-1
    ).sum(dim=1)
    excessive_fingertip_distance = fingertip_distance_sum > fingertip_distance_sum_limit

    return (
        insufficient_tip_contacts
        | missing_required_contacts
        | has_non_tip_contact
        | excessive_fingertip_distance
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


class reset_object_pose_in_robot_root_frame(ManagerTermBase):
    """Sample object pose deltas around the default pose in the robot root frame."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        object_asset_cfg: SceneEntityCfg = cfg.params.get("object_asset_cfg", SceneEntityCfg("object"))

        self._robot: Articulation = env.scene[robot_asset_cfg.name]
        self._object: RigidObject = env.scene[object_asset_cfg.name]

        # self._position_range = cfg.params.get(
        #     "position_range",
        #     {
        #         "x": (-0.025, 0.025),
        #         "y": (-0.08, -0.02),
        #         "z": (0.08, 0.16),
        #     },
        # )
        # self._euler_range = cfg.params.get(
        #     "euler_range",
        #     {
        #         "roll": (-torch.pi, torch.pi),
        #         "pitch": (-torch.pi, torch.pi),
        #         "yaw": (-torch.pi, torch.pi),
        #     },
        # )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        position_range: dict[str, tuple[float, float]] | None = None,
        euler_range: dict[str, tuple[float, float]] | None = None,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        del robot_asset_cfg, object_asset_cfg
        env_ids_t = _normalize_env_ids(env, env_ids)
        pos_range = self._position_range if position_range is None else position_range
        rot_range = self._euler_range if euler_range is None else euler_range

        robot_root_state = self._robot.data.default_root_state[env_ids_t]
        robot_pos_e = robot_root_state[:, 0:3]
        robot_quat = robot_root_state[:, 3:7]
        robot_quat_inv = robot_quat.clone()
        robot_quat_inv[:, 1:] *= -1.0

        default_object_state = self._object.data.default_root_state[env_ids_t]
        default_object_pos_e = default_object_state[:, 0:3]
        default_object_quat = default_object_state[:, 3:7]
        default_rel_pos = math_utils.quat_apply(
            robot_quat_inv, default_object_pos_e - robot_pos_e
        )
        default_rel_quat = math_utils.quat_mul(robot_quat_inv, default_object_quat)

        pos_delta = torch.stack(
            (
                torch.empty(len(env_ids_t), device=env.device).uniform_(*pos_range["x"]),
                torch.empty(len(env_ids_t), device=env.device).uniform_(*pos_range["y"]),
                torch.empty(len(env_ids_t), device=env.device).uniform_(*pos_range["z"]),
            ),
            dim=1,
        )
        roll = torch.empty(len(env_ids_t), device=env.device).uniform_(*rot_range["roll"])
        pitch = torch.empty(len(env_ids_t), device=env.device).uniform_(*rot_range["pitch"])
        yaw = torch.empty(len(env_ids_t), device=env.device).uniform_(*rot_range["yaw"])
        quat_delta = quat_from_euler_xyz(roll, pitch, yaw)

        rel_pos = default_rel_pos + pos_delta
        rel_quat = math_utils.quat_mul(quat_delta, default_rel_quat)

        root_state = self._object.data.default_root_state[env_ids_t].clone()
        root_state[:, 0:3] = robot_pos_e + math_utils.quat_apply(robot_quat, rel_pos)
        root_state[:, 3:7] = math_utils.quat_mul(robot_quat, rel_quat)
        root_state[:, 7:13] = 0.0
        root_state[:, 0:3] += env.scene.env_origins[env_ids_t]
        self._object.write_root_state_to_sim(root_state, env_ids=env_ids_t)


def _sanitize_cache_metadata(params: dict) -> dict[str, object]:
    """Convert config params into a saveable metadata snapshot."""
    snapshot = {}
    for key, value in params.items():
        if key in {"robot_asset_cfg", "object_asset_cfg"}:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot[key] = value
        elif isinstance(value, dict):
            snapshot[key] = {
                sub_key: list(sub_value) if isinstance(sub_value, tuple) else sub_value
                for sub_key, sub_value in value.items()
            }
        elif isinstance(value, (tuple, list)):
            snapshot[key] = list(value)
        else:
            snapshot[key] = repr(value)
    return snapshot


def _load_stable_grasp_cache(cache_path: str) -> dict[str, np.ndarray]:
    """Load and validate the saved stable-grasp cache."""
    payload = np.load(Path(cache_path), allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    if not isinstance(data, dict):
        raise ValueError("Stable grasp cache must contain a dictionary payload.")

    required = ("robot_root_state", "object_root_state", "robot_joint_pos")
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required stable-grasp cache key: '{key}'.")

    normalized = {}
    for key in required + ("robot_joint_vel",):
        if key not in data or data[key] is None:
            continue
        array = np.asarray(data[key], dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"Stable grasp cache key '{key}' must be a 1D or 2D array.")
        normalized[key] = array

    row_count = normalized["robot_root_state"].shape[0]
    for key in ("object_root_state", "robot_joint_pos"):
        if normalized[key].shape[0] != row_count:
            raise ValueError(
                "Stable grasp cache row counts must match across robot/object root states and joint positions."
            )
    if "robot_joint_vel" in normalized and normalized["robot_joint_vel"].shape[0] != row_count:
        raise ValueError("Stable grasp cache row count for 'robot_joint_vel' must match the other arrays.")

    if "robot_joint_vel" not in normalized:
        normalized["robot_joint_vel"] = np.zeros_like(normalized["robot_joint_pos"], dtype=np.float32)

    normalized["joint_names"] = list(data.get("joint_names", []))
    normalized["object_asset_path"] = data.get("object_asset_path")
    raw_paths = data.get("object_asset_paths")
    if raw_paths is not None:
        normalized["object_asset_paths"] = list(np.asarray(raw_paths).flat)
    normalized["generator_version"] = data.get("generator_version")
    normalized["config_snapshot"] = data.get("config_snapshot")
    return normalized


def _get_success_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the success mask for the environment.

    The success mask is a boolean tensor of shape (num_envs,) where each element is True if the distance between the object and the fingertip is less than a certain threshold, and the number of contacts are greater than 2.
    The threshold for the distance is 0.05 m.
    The threshold for the number of contacts is 2.
    """
    CONTACT_SENSOR_NAMES = [
        "thumb_tip_object_s",
        "index_tip_object_s",
        "middle_tip_object_s",
        "ring_tip_object_s",
    ]
    DISTANCE_THRESHOLD = 0.10
    CONTACT_FORCE_THRESHOLD = 1e-3
    MIN_CONTACTS = 2

    # distance check: mean fingertip–object distance
    fingertip_pos_w = env.scene.sensors["fingertip_transforms"].data.target_pos_w
    obj_pos_w = env.scene["object"].data.root_pos_w
    dists = torch.linalg.norm(fingertip_pos_w - obj_pos_w.unsqueeze(1), dim=-1)  # (N, F)
    close_enough = dists.mean(dim=1) < DISTANCE_THRESHOLD  # (N,)
    print(f"close_enough: {close_enough.sum()}, total: {close_enough.numel()}")
    # contact check: number of fingertips in contact with object
    contact_flags = []
    for name in CONTACT_SENSOR_NAMES:
        mag = _contact_sensor_max_magnitude(env.scene.sensors[name], env.num_envs, env.device)
        contact_flags.append(mag > CONTACT_FORCE_THRESHOLD)
    enough_contacts = torch.stack(contact_flags, dim=1).sum(dim=1) > MIN_CONTACTS  # (N,)
    print(f"enough_contacts: {enough_contacts.sum()}, total: {enough_contacts.numel()}")
    return close_enough & enough_contacts



def _resolve_per_env_object_asset_paths(
    env: ManagerBasedRLEnv,
    object_scene_key: str,
    fallback_path: str | None,
) -> list[str | None]:
    """Build a per-environment list of object USD asset paths.

    When ``MultiUsdFileCfg(random_choice=True)`` is used, each environment's Object
    prim holds a USD reference to its source file.  This function inspects those
    references to produce a mapping ``env_idx -> asset_path``.

    Falls back to the single ``fallback_path`` (or the spawner's ``usd_path``) when
    per-env references cannot be determined (e.g. single-asset scenes).
    """
    num_envs = env.num_envs
    scene_object = env.scene[object_scene_key]
    spawn_cfg = scene_object.cfg.spawn

    usd_path_list: list[str] | None = None
    if hasattr(spawn_cfg, "usd_path"):
        raw = spawn_cfg.usd_path
        usd_path_list = raw if isinstance(raw, list) else [raw]

    if usd_path_list is not None and len(usd_path_list) == 1:
        return [usd_path_list[0]] * num_envs

    if fallback_path and (usd_path_list is None or len(usd_path_list) <= 1):
        return [fallback_path] * num_envs

    try:
        from pxr import Sdf
        import isaaclab.sim as _sim_utils

        stage = _sim_utils.get_current_stage()
        env_prim_paths = env.scene.env_prim_paths
        object_prim_name = scene_object.cfg.prim_path.rsplit("/", 1)[-1]

        per_env: list[str | None] = []
        for env_path in env_prim_paths:
            obj_prim_path = f"{env_path}/{object_prim_name}"
            prim_spec = stage.GetRootLayer().GetPrimAtPath(Sdf.Path(obj_prim_path))
            if prim_spec is None:
                per_env.append(fallback_path)
                continue
            refs = prim_spec.referenceList.GetAddedOrExplicitItems()
            if refs:
                per_env.append(str(refs[0].assetPath))
            else:
                per_env.append(fallback_path)
        return per_env
    except Exception as e:
        print(f"[collect_stable_grasp_states] Could not resolve per-env asset paths: {e}")
        single = fallback_path
        if single is None and usd_path_list:
            single = usd_path_list[0]
        return [single] * num_envs


class collect_stable_grasp_states(ManagerTermBase):
    """Collect stable-grasp states on reset and persist them once the cache is full."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        object_asset_cfg: SceneEntityCfg = cfg.params.get("object_asset_cfg", SceneEntityCfg("object"))
        self._robot: Articulation = env.scene[robot_asset_cfg.name]
        self._object: RigidObject = env.scene[object_asset_cfg.name]

        self._cache_path = Path(cfg.params.get("cache_path", "stable_grasps.npy"))
        self._max_cached_grasp_size = int(cfg.params.get("max_cached_grasp_size", 1024))
        self._object_asset_path = cfg.params.get("object_asset_path")
        self._config_snapshot = _sanitize_cache_metadata(cfg.params)

        self._per_env_asset_paths: list[str | None] = _resolve_per_env_object_asset_paths(
            env, object_asset_cfg.name, self._object_asset_path # TODO: use relative path later.
        )
        self._cache = {
            "robot_root_state": [],
            "object_root_state": [],
            "robot_joint_pos": [],
            "robot_joint_vel": [],
            "object_asset_paths": [],
        }
        self._num_cached_grasps = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        cache_path: str | None = None,
        max_cached_grasp_size: int | None = None,
        object_asset_path: str | None = None,
    ):
        del robot_asset_cfg, object_asset_cfg, cache_path, max_cached_grasp_size, object_asset_path

        if env.extras.get("stable_grasp_generation_complete", False):
            return

        env_ids_t = _normalize_env_ids(env, env_ids)
        success_mask = _get_success_mask(env).index_select(0, env_ids_t)
        if not torch.any(success_mask):
            return

        success_env_ids = env_ids_t[success_mask]
        env_origins = env.scene.env_origins.index_select(0, success_env_ids)

        robot_root_state = self._robot.data.root_state_w.index_select(0, success_env_ids).clone()
        object_root_state = self._object.data.root_state_w.index_select(0, success_env_ids).clone()
        robot_root_state[:, 0:3] -= env_origins
        object_root_state[:, 0:3] -= env_origins

        self._cache["robot_root_state"].append(robot_root_state.detach().cpu().numpy().astype(np.float32))
        self._cache["object_root_state"].append(object_root_state.detach().cpu().numpy().astype(np.float32))
        self._cache["robot_joint_pos"].append(
            self._robot.data.joint_pos.index_select(0, success_env_ids).detach().cpu().numpy().astype(np.float32)
        )
        self._cache["robot_joint_vel"].append(
            self._robot.data.joint_vel.index_select(0, success_env_ids).detach().cpu().numpy().astype(np.float32)
        )
        import ipdb; ipdb.set_trace()
        row_asset_paths = [self._per_env_asset_paths[i] for i in success_env_ids.cpu().tolist()]
        self._cache["object_asset_paths"].extend(row_asset_paths)

        self._num_cached_grasps += int(success_env_ids.numel())
        env.extras["stable_grasp_num_cached"] = self._num_cached_grasps
        print(f"Number of cached grasps: {self._num_cached_grasps}, max cached grasp size: {self._max_cached_grasp_size}")
        if self._num_cached_grasps < self._max_cached_grasp_size:
            return

        payload = {}
        for key, chunks in self._cache.items():
            if key == "object_asset_paths":
                continue
            payload[key] = np.concatenate(chunks, axis=0)[: self._max_cached_grasp_size]
        payload["joint_names"] = list(self._robot.joint_names)
        payload["object_asset_paths"] = np.array(
            self._cache["object_asset_paths"][: self._max_cached_grasp_size], dtype=object
        )
        payload["object_asset_path"] = self._object_asset_path
        payload["generator_version"] = "isaaclab_manager_based_v1"
        payload["config_snapshot"] = self._config_snapshot

        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self._cache_path, payload, allow_pickle=True)
        env.extras["stable_grasp_generation_complete"] = True
        env.extras["stable_grasp_cache_path"] = str(self._cache_path)


class sample_saved_stable_grasps(ManagerTermBase):
    """Sample arbitrary rows from a saved stable-grasp cache and apply them on reset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        object_asset_cfg: SceneEntityCfg = cfg.params.get("object_asset_cfg", SceneEntityCfg("object"))
        self._robot: Articulation = env.scene[robot_asset_cfg.name]
        self._object: RigidObject = env.scene[object_asset_cfg.name]

        cache_path = cfg.params.get("cache_path")
        if cache_path is None:
            raise ValueError("sample_saved_stable_grasps requires 'cache_path'.")

        state = _load_stable_grasp_cache(cache_path)
        self._num_rows = state["robot_root_state"].shape[0]
        self._robot_root_state = torch.as_tensor(state["robot_root_state"], dtype=torch.float32, device=env.device)
        self._object_root_state = torch.as_tensor(state["object_root_state"], dtype=torch.float32, device=env.device)
        self._robot_joint_pos = torch.as_tensor(state["robot_joint_pos"], dtype=torch.float32, device=env.device)
        self._robot_joint_vel = torch.as_tensor(state["robot_joint_vel"], dtype=torch.float32, device=env.device)

        joint_names = state.get("joint_names", [])
        if joint_names and len(joint_names) != len(self._robot.joint_names):
            raise ValueError("Stable grasp cache joint count does not match the current robot articulation.")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | Sequence[int] | slice | None,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        cache_path: str | None = None,
    ):
        del robot_asset_cfg, object_asset_cfg, cache_path
        env_ids_t = _normalize_env_ids(env, env_ids)
        sample_ids = torch.randint(0, self._num_rows, (len(env_ids_t),), device=env.device)
        env_origins = env.scene.env_origins.index_select(0, env_ids_t)

        robot_root_state = self._robot_root_state.index_select(0, sample_ids).clone()
        object_root_state = self._object_root_state.index_select(0, sample_ids).clone()
        robot_root_state[:, 0:3] += env_origins
        object_root_state[:, 0:3] += env_origins

        robot_joint_pos = self._robot_joint_pos.index_select(0, sample_ids).clone()
        robot_joint_vel = self._robot_joint_vel.index_select(0, sample_ids).clone()

        self._robot.write_root_state_to_sim(robot_root_state, env_ids=env_ids_t)
        self._object.write_root_state_to_sim(object_root_state, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(robot_joint_pos, robot_joint_vel, env_ids=env_ids_t)
        env.extras["stable_grasp_sample_ids"] = sample_ids.detach().cpu()
