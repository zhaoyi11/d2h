# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from src.utils.point_cloud import sample_object_point_cloud

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def fingertip_pos_source(
    env: ManagerBasedRLEnv,
    sensor_name: str = "fingertip_transforms",
    flatten: bool = True,
) -> torch.Tensor:
    """Fingertip positions relative to the source frame (robot base).

    Uses the FrameTransformer sensor's target_pos_source data.

    Args:
        env: The environment.
        sensor_name: Name of the FrameTransformer sensor. Defaults to "fingertip_transforms".
        flatten: If True, returns flattened tensor of shape (num_envs, num_fingers * 3).
                 If False, returns tensor of shape (num_envs, num_fingers, 3).

    Returns:
        Tensor of fingertip positions relative to the robot base frame.
    """
    fingertip_pos = env.scene.sensors[sensor_name].data.target_pos_source
    if flatten:
        return fingertip_pos.view(env.num_envs, -1)
    return fingertip_pos


def fingertip_quat_source(
    env: ManagerBasedRLEnv,
    sensor_name: str = "fingertip_transforms",
    flatten: bool = True,
) -> torch.Tensor:
    """Fingertip orientations (quaternions) relative to the source frame (robot base).

    Uses the FrameTransformer sensor's target_quat_source data.

    Args:
        env: The environment.
        sensor_name: Name of the FrameTransformer sensor. Defaults to "fingertip_transforms".
        flatten: If True, returns flattened tensor of shape (num_envs, num_fingers * 4).
                 If False, returns tensor of shape (num_envs, num_fingers, 4).

    Returns:
        Tensor of fingertip orientations (w, x, y, z) relative to the robot base frame.
    """
    fingertip_quat = env.scene.sensors[sensor_name].data.target_quat_source
    if flatten:
        return fingertip_quat.view(env.num_envs, -1)
    return fingertip_quat


def object_pos_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Object position in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 3)``: object position [x, y, z] expressed in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(
        robot.data.root_quat_w, object.data.root_pos_w - robot.data.root_pos_w
    )


def object_quat_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation in the robot's root frame.

    Args:
        env: The environment.
        robot_cfg: Scene entity for the robot (reference frame). Defaults to ``SceneEntityCfg("robot")``.
        object_cfg: Scene entity for the object. Defaults to ``SceneEntityCfg("object")``.

    Returns:
        Tensor of shape ``(num_envs, 4)``: object quaternion ``(w, x, y, z)`` in the robot root frame.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def body_state_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state (pos, quat, lin vel, ang vel) in the base asset's root frame.

    The state for each body is stacked horizontally as
    ``[position(3), quaternion(4)(wxyz), linvel(3), angvel(3)]`` and then concatenated over bodies.

    Args:
        env: The environment.
        body_asset_cfg: Scene entity for the articulated body whose links are observed.
        base_asset_cfg: Scene entity providing the reference (root) frame.

    Returns:
        Tensor of shape ``(num_envs, num_bodies * 13)`` with per-body states expressed in the base root frame.
    """
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(
        -1, 3
    )
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(
        -1, 3
    )
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = (
        base_asset.data.root_link_pos_w.unsqueeze(1)
        .repeat_interleave(num_bodies, dim=1)
        .view(-1, 3)
    )
    root_quat_w = (
        base_asset.data.root_link_quat_w.unsqueeze(1)
        .repeat_interleave(num_bodies, dim=1)
        .view(-1, 4)
    )
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, body_pos_w, body_quat_w
    )
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    # concate and return
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


class object_point_cloud_b(ManagerTermBase):
    """Object surface point cloud expressed in a reference asset's root frame.

    Points are pre-sampled on the object's surface in its local frame and transformed to world,
    then into the reference (e.g., robot) root frame. Optionally visualizes the points.

    Args (from ``cfg.params``):
        object_cfg: Scene entity for the object to sample. Defaults to ``SceneEntityCfg("object")``.
        ref_asset_cfg: Scene entity providing the reference frame. Defaults to ``SceneEntityCfg("robot")``.
        num_points: Number of points to sample on the object surface. Defaults to ``10``.
        visualize: Whether to draw markers for the points. Defaults to ``True``.
        static: If ``True``, cache world-space points on reset and reuse them (no per-step resampling).

    Returns (from ``__call__``):
        If ``flatten=False``: tensor of shape ``(num_envs, num_points, 3)``.
        If ``flatten=True``: tensor of shape ``(num_envs, 3 * num_points)``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object_cfg: SceneEntityCfg = cfg.params.get(
            "object_cfg", SceneEntityCfg("object")
        )
        self.ref_asset_cfg: SceneEntityCfg = cfg.params.get(
            "ref_asset_cfg", SceneEntityCfg("robot")
        )
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        # lazy initialize visualizer and point cloud
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG

            ray_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path="/Visuals/ObservationPointCloud"
            )
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ref_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        num_points: int = 10,
        flatten: bool = False,
        visualize: bool = True,
    ):
        """Compute the object point cloud in the reference asset's root frame.

        Note:
            Points are pre-sampled at initialization using ``self.num_points``; the ``num_points`` argument is
            kept for API symmetry and does not change the sampled set at runtime.

        Args:
            env: The environment.
            ref_asset_cfg: Reference frame provider (root). Defaults to ``SceneEntityCfg("robot")``.
            object_cfg: Object to sample. Defaults to ``SceneEntityCfg("object")``.
            num_points: Unused at runtime; see note above.
            flatten: If ``True``, return a flattened tensor ``(num_envs, 3 * num_points)``.
            visualize: If ``True``, draw markers for the points.

        Returns:
            Tensor of shape ``(num_envs, num_points, 3)`` or flattened if requested.
        """
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(
            1, num_points, 1
        )

        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(
            1, num_points, 1
        )
        # apply rotation + translation
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(
            ref_pos_w, ref_quat_w, self.points_w, None
        )

        return (
            object_point_cloud_pos_b.view(env.num_envs, -1)
            if flatten
            else object_point_cloud_pos_b
        )


def fingers_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.

    Returns:
        Tensor of shape ``(num_envs, num_sensors, 3)`` with forces stacked along dimension 1 as
        ``[fx, fy, fz]`` per sensor.
    """
    # We want one 3D force vector per sensor per environment in the robot base frame.
    # Prefer filtered contact forces (force_matrix_w) when a filter is configured; otherwise fall back
    # to net_forces_w (unfiltered sum over all contacts on the sensor bodies).
    forces_w = []
    for name in contact_sensor_names:
        sensor = env.scene.sensors[name]
        # Filtered matrix: (num_envs, num_bodies, num_filters, 3)
        fm = getattr(sensor.data, "net_force_w", None)
        if fm is not None and fm.numel() > 0:
            f_w = torch.nan_to_num(fm, nan=0.0).sum(dim=(1, 2))
        else:
            # Net forces: (num_envs, num_bodies, 3)
            net = sensor.data.net_forces_w
            f_w = torch.nan_to_num(net, nan=0.0).sum(dim=1)
        forces_w.append(f_w)
    force_w = torch.stack(forces_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(
        robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w
    )

    return forces_b.view(env.num_envs, -1)


def goal_quat_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    make_quat_unique: bool,
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(
        command_name
    )

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat
