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
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from src.tasks.common.obj_point_cloud import sample_object_point_cloud
from src.tasks.common.mdps.contacts import _compute_contact_metrics

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


def object_pose_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose in the robot's root frame as ``[position, quaternion]``."""
    return torch.cat(
        (
            object_pos_b(env, robot_cfg=robot_cfg, object_cfg=object_cfg),
            object_quat_b(env, robot_cfg=robot_cfg, object_cfg=object_cfg),
        ),
        dim=1,
    )


def object_pos_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object position expressed in a selected articulated body frame."""
    object_pos_b, _ = _object_pose_body_b(env, body_asset_cfg, object_cfg)
    return object_pos_b


def object_quat_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object orientation expressed in a selected articulated body frame."""
    _, object_quat_b = _object_pose_body_b(env, body_asset_cfg, object_cfg)
    return object_quat_b


def object_pose_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object pose in a selected articulated body frame as ``[position, quaternion]``."""
    object_pos_b, object_quat_b = _object_pose_body_b(env, body_asset_cfg, object_cfg)
    return torch.cat((object_pos_b, object_quat_b), dim=1)


def object_lin_vel_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object linear velocity relative to a selected body, expressed in that body frame."""
    object: RigidObject = env.scene[object_cfg.name]
    _, body_quat_w, body_lin_vel_w, _ = _single_body_state_w(env, body_asset_cfg)
    rel_vel_w = object.data.root_lin_vel_w - body_lin_vel_w
    return quat_apply_inverse(body_quat_w, rel_vel_w)


def object_ang_vel_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object angular velocity relative to a selected body, expressed in that body frame."""
    object: RigidObject = env.scene[object_cfg.name]
    _, body_quat_w, _, body_ang_vel_w = _single_body_state_w(env, body_asset_cfg)
    rel_ang_vel_w = object.data.root_ang_vel_w - body_ang_vel_w
    return quat_apply_inverse(body_quat_w, rel_ang_vel_w)


def command_object_pose_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target object pose slice from a command that may contain additional target frames."""
    return env.command_manager.get_command(command_name)[:, :7]


def command_hand_base_pose_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Target robot hand-base pose from an object-and-hand-base command."""
    return env.command_manager.get_command(command_name)[:, 7:14]


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
    body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b = _body_state_components_b(
        env, body_asset_cfg, base_asset_cfg
    )
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=-1)
    return out.reshape(env.num_envs, -1)


def body_state_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_body_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body state relative to one selected body frame."""
    body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b = _body_state_components_body_b(
        env, body_asset_cfg, base_body_asset_cfg
    )
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=-1)
    return out.reshape(env.num_envs, -1)


def body_pos_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body positions in the base asset's root frame."""
    body_pos_b, _, _, _ = _body_state_components_b(env, body_asset_cfg, base_asset_cfg)
    return body_pos_b.reshape(env.num_envs, -1)


def body_quat_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body orientations in the base asset's root frame.

    Quaternions are returned in ``(w, x, y, z)`` order.
    """
    _, body_quat_b, _, _ = _body_state_components_b(env, body_asset_cfg, base_asset_cfg)
    return body_quat_b.reshape(env.num_envs, -1)


def body_lin_vel_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body linear velocities in the base asset's root frame."""
    _, _, body_lin_vel_b, _ = _body_state_components_b(
        env, body_asset_cfg, base_asset_cfg
    )
    return body_lin_vel_b.reshape(env.num_envs, -1)


def body_ang_vel_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body angular velocities in the base asset's root frame."""
    _, _, _, body_ang_vel_b = _body_state_components_b(
        env, body_asset_cfg, base_asset_cfg
    )
    return body_ang_vel_b.reshape(env.num_envs, -1)


def _body_state_components_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base-frame body state components shaped ``(num_envs, num_bodies, dim)``."""
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    # get world pose of bodies
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].reshape(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].reshape(
        -1, 4
    )
    body_lin_vel_w = body_asset.data.body_lin_vel_w[
        :, body_asset_cfg.body_ids
    ].reshape(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[
        :, body_asset_cfg.body_ids
    ].reshape(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    # get world pose of base frame
    root_pos_w = (
        base_asset.data.root_link_pos_w.unsqueeze(1)
        .repeat_interleave(num_bodies, dim=1)
        .reshape(-1, 3)
    )
    root_quat_w = (
        base_asset.data.root_link_quat_w.unsqueeze(1)
        .repeat_interleave(num_bodies, dim=1)
        .reshape(-1, 4)
    )
    # transform from world body pose to local body pose
    body_pos_b, body_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, body_pos_w, body_quat_w
    )
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    return (
        body_pos_b.reshape(env.num_envs, num_bodies, 3),
        body_quat_b.reshape(env.num_envs, num_bodies, 4),
        body_lin_vel_b.reshape(env.num_envs, num_bodies, 3),
        body_ang_vel_b.reshape(env.num_envs, num_bodies, 3),
    )


def _body_state_components_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_body_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return body state components shaped ``(num_envs, num_bodies, dim)`` in one body frame."""
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    body_pos_w = body_asset.data.body_pos_w[:, _resolve_body_ids(body_asset, body_asset_cfg)]
    body_quat_w = body_asset.data.body_quat_w[:, _resolve_body_ids(body_asset, body_asset_cfg)]
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, _resolve_body_ids(body_asset, body_asset_cfg)]
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, _resolve_body_ids(body_asset, body_asset_cfg)]
    if body_pos_w.ndim == 2:
        body_pos_w = body_pos_w.unsqueeze(1)
        body_quat_w = body_quat_w.unsqueeze(1)
        body_lin_vel_w = body_lin_vel_w.unsqueeze(1)
        body_ang_vel_w = body_ang_vel_w.unsqueeze(1)

    num_bodies = body_pos_w.shape[1]
    base_pos_w, base_quat_w, base_lin_vel_w, base_ang_vel_w = _single_body_state_w(env, base_body_asset_cfg)
    base_pos_w = base_pos_w.unsqueeze(1).repeat(1, num_bodies, 1)
    base_quat_w = base_quat_w.unsqueeze(1).repeat(1, num_bodies, 1)
    base_lin_vel_w = base_lin_vel_w.unsqueeze(1).repeat(1, num_bodies, 1)
    base_ang_vel_w = base_ang_vel_w.unsqueeze(1).repeat(1, num_bodies, 1)

    body_pos_b, body_quat_b = subtract_frame_transforms(base_pos_w, base_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(base_quat_w, body_lin_vel_w - base_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(base_quat_w, body_ang_vel_w - base_ang_vel_w)
    return body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b


def _object_pose_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    object: RigidObject = env.scene[object_cfg.name]
    body_pos_w, body_quat_w, _, _ = _single_body_state_w(env, body_asset_cfg)
    return subtract_frame_transforms(
        body_pos_w,
        body_quat_w,
        object.data.root_pos_w,
        object.data.root_quat_w,
    )


def _single_body_state_w(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    body_ids = _resolve_body_ids(body_asset, body_asset_cfg)
    body_pos_w = body_asset.data.body_pos_w[:, body_ids]
    body_quat_w = body_asset.data.body_quat_w[:, body_ids]
    body_lin_vel_data = getattr(body_asset.data, "body_lin_vel_w", None)
    body_ang_vel_data = getattr(body_asset.data, "body_ang_vel_w", None)
    body_lin_vel_w = body_lin_vel_data[:, body_ids] if body_lin_vel_data is not None else torch.zeros_like(body_pos_w)
    body_ang_vel_w = body_ang_vel_data[:, body_ids] if body_ang_vel_data is not None else torch.zeros_like(body_pos_w)
    if body_pos_w.ndim == 2:
        body_pos_w = body_pos_w.unsqueeze(1)
        body_quat_w = body_quat_w.unsqueeze(1)
        body_lin_vel_w = body_lin_vel_w.unsqueeze(1)
        body_ang_vel_w = body_ang_vel_w.unsqueeze(1)
    if body_pos_w.shape[1] != 1:
        raise ValueError(f"Expected one body frame for {body_asset_cfg.name}, found {body_pos_w.shape[1]}.")
    return body_pos_w[:, 0], body_quat_w[:, 0], body_lin_vel_w[:, 0], body_ang_vel_w[:, 0]


def _resolve_body_ids(body_asset: Articulation, body_asset_cfg: SceneEntityCfg):
    body_ids = getattr(body_asset_cfg, "body_ids", None)
    if body_ids is None:
        body_names = getattr(body_asset_cfg, "body_names", None)
        body_ids, _ = body_asset.find_bodies(body_names)
    return body_ids


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
    filter_indices: list[int] | None = None,
) -> torch.Tensor:
    """base-frame contact forces from listed sensors, concatenated per env.

    Args:
        env: The environment.
        contact_sensor_names: Names of contact sensors in ``env.scene.sensors`` to read.
        filter_indices: Which contact-filter indices to aggregate. ``None`` sums all
            filters (default). Pass a subset (e.g. only the object's filter index) to
            restrict the force to specific contact targets when a sensor has multiple
            filters.

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
        fm = getattr(sensor.data, "force_matrix_w", None)
        if fm is not None and fm.numel() > 0:
            fm = torch.nan_to_num(fm, nan=0.0)
            if filter_indices is not None:
                valid = [k for k in filter_indices if 0 <= k < fm.shape[2]]
                fm = fm[:, :, valid, :] if valid else fm[:, :, :0, :]
            f_w = fm.sum(dim=(1, 2))
        else:
            # Net forces: (num_envs, num_bodies, 3)
            net = sensor.data.force_matrix_w
            f_w = torch.nan_to_num(net, nan=0.0).sum(dim=1)
        forces_w.append(f_w)
    force_w = torch.stack(forces_w, dim=1)

    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(
        robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w
    )
    return forces_b.view(env.num_envs, -1)


def fingers_contact_force_body_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    base_body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    filter_indices: list[int] | None = None,
) -> torch.Tensor:
    """Contact forces from listed sensors, expressed in one selected body frame.

    ``filter_indices`` selects which contact-filter columns of ``force_matrix_w`` to
    aggregate (``None`` sums all). Pass the object's filter index to keep the per-finger
    object force channel clean when a sensor filters against multiple targets.
    """
    forces_w = []
    for name in contact_sensor_names:
        sensor = env.scene.sensors[name]
        fm = getattr(sensor.data, "force_matrix_w", None)
        if fm is not None and fm.numel() > 0:
            fm = torch.nan_to_num(fm, nan=0.0)
            if filter_indices is not None:
                valid = [k for k in filter_indices if 0 <= k < fm.shape[2]]
                fm = fm[:, :, valid, :] if valid else fm[:, :, :0, :]
            f_w = fm.sum(dim=(1, 2))
        else:
            net = getattr(sensor.data, "net_forces_w", None)
            if net is None:
                net = sensor.data.force_matrix_w
            f_w = torch.nan_to_num(net, nan=0.0).sum(dim=1)
        forces_w.append(f_w)
    force_w = torch.stack(forces_w, dim=1)

    _, base_quat_w, _, _ = _single_body_state_w(env, base_body_asset_cfg)
    forces_b = quat_apply_inverse(base_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b.view(env.num_envs, -1)


def gravity_dir_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    """Unit gravity vector expressed in one selected body frame."""
    _, body_quat_w, _, _ = _single_body_state_w(env, body_asset_cfg)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=body_quat_w.dtype, device=body_quat_w.device)
    return quat_apply_inverse(body_quat_w, gravity_w.expand(env.num_envs, -1))


def goal_pos_diff_body_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    command_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position from the goal to the asset, expressed in one selected body frame."""
    asset_pos_b, _ = _object_pose_body_b(env, body_asset_cfg, asset_cfg)
    goal_pos_b, _ = _command_object_pose_body_b(env, command_name, body_asset_cfg, command_asset_cfg)
    return asset_pos_b - goal_pos_b


def goal_quat_diff_body_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    make_quat_unique: bool,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    command_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Rotation from the goal orientation to the asset orientation in one selected body frame."""
    _, asset_quat_b = _object_pose_body_b(env, body_asset_cfg, asset_cfg)
    _, goal_quat_b = _command_object_pose_body_b(env, command_name, body_asset_cfg, command_asset_cfg)
    quat = math_utils.quat_mul(asset_quat_b, math_utils.quat_conjugate(goal_quat_b))
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def _command_object_pose_body_b(
    env: ManagerBasedRLEnv,
    command_name: str,
    body_asset_cfg: SceneEntityCfg,
    command_asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_command(command_name)
    if command.shape[-1] >= 14:
        # For object-and-hand-base commands, command[:7] is the desired in-hand
        # object pose. The arm IK owns command[7:14], so the low-level hand
        # policy should not chase the target hand-base transit error.
        return command[:, :3], command[:, 3:7]

    command_asset: Articulation = env.scene[command_asset_cfg.name]
    goal_pos_w, goal_quat_w = combine_frame_transforms(
        command_asset.data.root_pos_w,
        command_asset.data.root_quat_w,
        command[:, :3],
        command[:, 3:7],
    )
    body_pos_w, body_quat_w, _, _ = _single_body_state_w(env, body_asset_cfg)
    return subtract_frame_transforms(body_pos_w, body_quat_w, goal_pos_w, goal_quat_w)


def goal_quat_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    make_quat_unique: bool,
    robot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Rotation from the goal orientation to the asset's current orientation.

    When ``robot_cfg`` is provided both quaternions are first transformed into the
    robot root frame before computing the difference, making the result invariant to
    rigid scene rotations (e.g. startup wrist-pose randomisation).  When ``robot_cfg``
    is ``None`` (default) the computation is done in world frame for backward
    compatibility.

    The quaternion is represented as (w, x, y, z).
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(
        command_name
    )

    # obtain world-frame orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    if robot_cfg is not None:
        # transform both into robot root frame so the result is invariant to
        # rigid scene rotations applied at startup
        robot: Articulation = env.scene[robot_cfg.name]
        robot_quat_inv = quat_inv(robot.data.root_quat_w)
        asset_quat = quat_mul(robot_quat_inv, asset_quat_w)
        goal_quat = quat_mul(robot_quat_inv, goal_quat_w)
    else:
        asset_quat = asset_quat_w
        goal_quat = goal_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat, math_utils.quat_conjugate(goal_quat))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def goal_pos_diff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    robot_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Position from the goal to the asset's current position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(
        command_name
    )

    goal_pos_w = command_term.command[:, 0:3] + env.scene.env_origins
    asset_pos_w = asset.data.root_pos_w

    if robot_cfg is not None:
        robot: Articulation = env.scene[robot_cfg.name]
        asset_pos = quat_apply_inverse(
            robot.data.root_quat_w, asset_pos_w - robot.data.root_pos_w
        )
        goal_pos = quat_apply_inverse(
            robot.data.root_quat_w, goal_pos_w - robot.data.root_pos_w
        )
    else:
        asset_pos = asset_pos_w
        goal_pos = goal_pos_w

    return asset_pos - goal_pos


def tip_contact_mask_obs(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
    filter_indices: list[int] | None = None,
) -> torch.Tensor:
    """Per-fingertip binary contact mask, shape (num_envs, n_fingers).

    ``filter_indices`` selects which contact targets to aggregate (``None`` -> object only);
    pass ``external_indices()`` for the merged external-contact channel.
    """
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, 50.0, filter_indices
    )
    return m["in_contact"].float()                               # (N, n)


def tip_contact_force_mag_obs(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
    filter_indices: list[int] | None = None,
) -> torch.Tensor:
    """Per-fingertip contact force magnitudes, shape (num_envs, n_fingers).

    ``filter_indices`` selects which contact targets to aggregate (``None`` -> object only).
    """
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, 50.0, filter_indices
    )
    return m["force_mag"]                                        # (N, n)


def tip_contact_pose_flat(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
    contact_pose_range_deg: float = 50.0,
    filter_indices: list[int] | None = None,
) -> torch.Tensor:
    """Per-fingertip (theta, phi) flattened; zeroed for non-contacting fingers. Shape (N, n*2).

    ``filter_indices`` selects which contact targets to aggregate (``None`` -> object only).
    """
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold,
        contact_pose_range_deg, filter_indices
    )
    pose = m["contact_pose"] * m["in_contact"].unsqueeze(-1).float()
    return pose.flatten(start_dim=1)                             # (N, n*2)
