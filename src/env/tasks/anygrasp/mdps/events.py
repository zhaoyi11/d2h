# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""


from __future__ import annotations

import torch
import re
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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
        self._pos_joint_ids = torch.tensor(pos_joint_ids, device=self._pos_ranges.device)
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]

        # create buffers to store the joint velocity range
        self._vel_ranges = torch.stack(
            [-self._asset.data.soft_joint_vel_limits[0], self._asset.data.soft_joint_vel_limits[0]], dim=1
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
                    self._vel_ranges[joint_ids, 0] = joint_range[0] * self._vel_ranges[joint_ids, 0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1] * self._vel_ranges[joint_ids, 1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint velocity ranges. Please use 'abs' or 'scale'."
                )
            # add the default offset
            if use_default_offset:
                self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)

        # store the joint vel ids (used later to sample the joint positions)
        self._vel_joint_ids = torch.tensor(vel_joint_ids, device=self._vel_ranges.device)
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
                self._pos_ranges[:, 0], self._pos_ranges[:, 1], joint_pos_shape, device=joint_pos.device
            )
            # clip the joint positions to the joint limits
            joint_pos_limits = self._asset.data.soft_joint_pos_limits[0, self._pos_joint_ids]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])

        # sample random joint velocities for each joint
        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = sample_uniform(
                self._vel_ranges[:, 0], self._vel_ranges[:, 1], joint_vel_shape, device=joint_vel.device
            )
            # clip the joint velocities to the joint limits
            joint_vel_limits = self._asset.data.soft_joint_vel_limits[0, self._vel_joint_ids]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)

        # set into the physics simulation
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


class reset_root_state_from_pose(ManagerTermBase):
    """Reset an asset root state to a fixed pose and velocity."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._asset = env.scene[asset_cfg.name]

        pose = cfg.params.get("pose", None)
        if pose is None or len(pose) != 7:
            raise ValueError("reset_root_state_from_pose requires 'pose' as (x,y,z,w,x,y,z).")

        velocity = cfg.params.get("velocity", None)
        if velocity is None:
            velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        if len(velocity) != 6:
            raise ValueError("reset_root_state_from_pose requires 'velocity' as (vx,vy,vz,wx,wy,wz).")

        self._pose = torch.tensor(pose, dtype=torch.float32, device=env.device)
        self._velocity = torch.tensor(velocity, dtype=torch.float32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose: tuple[float, float, float, float, float, float, float] | None = None,
        velocity: tuple[float, float, float, float, float, float] | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ):
        pose_tensor = self._pose if pose is None else torch.tensor(pose, dtype=torch.float32, device=env.device)
        vel_tensor = (
            self._velocity if velocity is None else torch.tensor(velocity, dtype=torch.float32, device=env.device)
        )

        root_states = self._asset.data.default_root_state[env_ids].clone()
        root_states[:, 0:3] = pose_tensor[0:3] + env.scene.env_origins[env_ids]
        root_states[:, 3:7] = pose_tensor[3:7]
        root_states[:, 7:10] = vel_tensor[0:3]
        root_states[:, 10:13] = vel_tensor[3:6]

        self._asset.write_root_state_to_sim(root_states, env_ids=env_ids)


class reset_joints_to_fixed(ManagerTermBase):
    """Reset an articulation's joints to a fixed pose (e.g., from grasp data)."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._asset: Articulation = env.scene[asset_cfg.name]

        joint_pos = cfg.params.get("joint_pos", None)
        if joint_pos is None:
            raise ValueError("reset_joints_to_fixed requires 'joint_pos' in MuJoCo/cuRobo order.")

        self._joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=env.device)
        self._mapping = self._build_mujoco_to_isaaclab_mapping(self._asset.joint_names)

        self._joint_pos_isaac = self._joint_pos[self._mapping].clone()

    def _build_mujoco_to_isaaclab_mapping(self, isaaclab_joint_names: list[str]) -> torch.Tensor:
        curobo_joint_order = [
            "j1",
            "j0",
            "j2",
            "j3",
            "j5",
            "j4",
            "j6",
            "j7",
            "j9",
            "j8",
            "j10",
            "j11",
            "j12",
            "j13",
            "j14",
            "j15",
        ]

        mapping = []
        for isaac_name in isaaclab_joint_names:
            match = re.search(r"(\d+)", isaac_name)
            if match:
                joint_num = int(match.group(1))
                target_joint = f"j{joint_num}"
                if target_joint in curobo_joint_order:
                    mapping.append(curobo_joint_order.index(target_joint))
                else:
                    mapping.append(len(mapping))
            else:
                mapping.append(len(mapping))

        return torch.tensor(mapping, dtype=torch.long, device=self._joint_pos.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        joint_pos: list[float] | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        joint_pos_tensor = (
            self._joint_pos_isaac
            if joint_pos is None
            else torch.tensor(joint_pos, dtype=torch.float32, device=env.device)[self._mapping]
        )

        joint_pos_out = self._asset.data.default_joint_pos[env_ids].clone()
        joint_pos_out[:] = joint_pos_tensor
        joint_vel_out = torch.zeros_like(joint_pos_out)

        self._asset.write_joint_state_to_sim(joint_pos_out, joint_vel_out, env_ids=env_ids)
