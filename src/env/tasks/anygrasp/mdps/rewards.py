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


def fingertip_object_distance(
    env: ManagerBasedRLEnv,
    fingertip_body_idx: list[int],
    fingertip_offset_pos: list[list[float]] | None = None,
    fingertip_offset_rot: list[list[float]] | None = None,
    threshold: float = 1e-3,
) -> torch.Tensor:
    """Calculate the average distance between the object and the robot's fingertips as a reward term."""

    fingertip_pos = env.scene.sensors["fingertip_transforms"].data.target_pos_w
    fingertip_pos -= env.scene.env_origins.repeat((1, len(fingertip_body_idx))).reshape(
        env.num_envs, len(fingertip_body_idx), 3
    )

    # extract obj's pose
    obj: RigidObject = env.scene["object"]
    # obtain the object position in the environment frame
    obj_pos = obj.data.root_pos_w - env.scene.env_origins

    # calculate the average distance between fingertips and obj
    object_pos_expanded = obj_pos.unsqueeze(1)
    dists = torch.norm(fingertip_pos - object_pos_expanded, p=2, dim=-1)

    return torch.mean(dists, dim=-1)


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
        # net_matrix_w = sensor.data.net_forces_w

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


class FingertipObjectProximityReward(ManagerTermBase):
    """Geometric fingertip-to-object proximity reward using sampled surface points.

    This term samples:
    - ``num_object_points`` points on each environment's object surface (in the object's root frame)
    - ``num_tip_points`` points on each fingertip link surface (in the fingertip link frame)

    At runtime, it transforms both sets into world frame and computes, for each fingertip, the minimal
    distance between the fingertip point set and the object point set.

    Note:
        Object point sampling is per-environment.
        Fingertip point sampling is done from env_0 and reused across environments.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot_cfg: SceneEntityCfg = cfg.params.get(
            "robot_cfg", SceneEntityCfg("robot")
        )
        self.object_cfg: SceneEntityCfg = cfg.params.get(
            "object_cfg", SceneEntityCfg("object")
        )
        self.fingertip_prim_paths: list[str] = cfg.params.get(
            "fingertip_prim_paths", []
        )

        self.num_tip_points: int = int(cfg.params.get("num_tip_points", 12))
        self.num_object_points: int = int(cfg.params.get("num_object_points", 64))
        self.object_chunk_size: int = int(cfg.params.get("object_chunk_size", 16))

        self._robot: Articulation = env.scene[self.robot_cfg.name]
        self._object: RigidObject = env.scene[self.object_cfg.name]

        if self.robot_cfg.body_ids is None:
            raise ValueError(
                "FingertipObjectProximityReward requires robot_cfg with resolved body_ids (provide body_names)."
            )
        self._tip_body_ids = torch.as_tensor(
            self.robot_cfg.body_ids, device=env.device, dtype=torch.long
        )

        if len(self.fingertip_prim_paths) != int(self._tip_body_ids.numel()):
            raise ValueError(
                "fingertip_prim_paths length must match robot_cfg.body_ids length. "
                f"Got {len(self.fingertip_prim_paths)} vs {int(self._tip_body_ids.numel())}."
            )

        # Per-env object surface points in object root frame: (N, P_obj, 3)
        self._object_points_local = sample_object_point_cloud(
            env.num_envs,
            self.num_object_points,
            self._object.cfg.prim_path,
            device=env.device,
        )

        # Fingertip surface points in fingertip link frame, sampled from env_0 only.
        tip_pts_local = []
        for prim_path in self.fingertip_prim_paths:
            pts = sample_object_point_cloud(
                1, self.num_tip_points, prim_path, device=env.device
            )[0]
            tip_pts_local.append(pts)
        self._tip_points_local = torch.stack(tip_pts_local, dim=0)  # (N_tip, P_tip, 3)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        fingertip_prim_paths: list[str] | None = None,
        num_tip_points: int = 12,
        num_object_points: int = 64,
        object_chunk_size: int = 16,
        distance_threshold: float = 0.01,
        mode: str = "neg",
        sigma: float = 0.01,
    ) -> torch.Tensor:
        # object points in world: (N, P_obj, 3)
        obj_pos_w = self._object.data.root_pos_w
        obj_quat_w = self._object.data.root_quat_w
        obj_quat_w_rep = obj_quat_w.unsqueeze(1).repeat(1, self.num_object_points, 1)
        obj_pts_w = quat_apply(
            obj_quat_w_rep, self._object_points_local
        ) + obj_pos_w.unsqueeze(1)

        # fingertip points in world: (N, N_tip, P_tip, 3)
        tip_pos_w = self._robot.data.body_pos_w[:, self._tip_body_ids]
        tip_quat_w = self._robot.data.body_quat_w[:, self._tip_body_ids]
        tip_pts_local = self._tip_points_local.unsqueeze(0).expand(
            env.num_envs, -1, -1, -1
        )
        tip_quat_w_rep = tip_quat_w.unsqueeze(2).repeat(1, 1, self.num_tip_points, 1)
        tip_pts_w = quat_apply(tip_quat_w_rep, tip_pts_local) + tip_pos_w.unsqueeze(2)

        # Flatten fingertip points and compute per-fingertip min distance to object points.
        tip_flat = tip_pts_w.reshape(env.num_envs, -1, 3)  # (N, Q, 3)
        a2 = (tip_flat * tip_flat).sum(dim=-1, keepdim=True)  # (N, Q, 1)

        q = tip_flat.shape[1]
        min_d2_flat = torch.full(
            (env.num_envs, q), float("inf"), device=env.device, dtype=tip_flat.dtype
        )

        # allow overriding chunk size at call-time (config passes it as a param)
        chunk = max(1, int(object_chunk_size))
        for start in range(0, self.num_object_points, chunk):
            end = min(self.num_object_points, start + chunk)
            b = obj_pts_w[:, start:end, :]  # (N, K, 3)
            b2 = (b * b).sum(dim=-1).unsqueeze(1)  # (N, 1, K)
            ab = torch.bmm(tip_flat, b.transpose(1, 2))  # (N, Q, K)
            d2 = (a2 + b2 - 2.0 * ab).clamp_min(0.0)
            min_d2_flat = torch.minimum(min_d2_flat, d2.amin(dim=-1))

        # reshape Q -> (N_tip, P_tip) and reduce
        min_d2 = min_d2_flat.view(env.num_envs, -1, self.num_tip_points).amin(dim=-1)
        min_d = torch.sqrt(min_d2.clamp_min(1e-12))

        if mode == "count":
            return (min_d < distance_threshold).to(dtype=torch.float32).sum(dim=1)
        if mode == "neg":
            return -min_d.sum(dim=1)
        # default: smooth proximity
        return torch.exp(-min_d / max(sigma, 1e-6)).sum(dim=1)


def object_stay_close(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    eps: float = 1e-6,
) -> torch.Tensor:
    """Rewards keeping object near its reset position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    init_pos = env.extras.get("object_init_pos", asset.data.root_pos_w)
    dist = torch.norm(asset.data.root_pos_w - init_pos, dim=-1)
    return -dist
