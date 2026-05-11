from __future__ import annotations

import math as _math
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_inv, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class HandFrameReOrientationCommand(CommandTerm):
    """Reorientation goal command expressed in an articulation body frame."""

    cfg: "HandFrameReOrientationCommandCfg"

    def __init__(self, cfg: "HandFrameReOrientationCommandCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object: RigidObject = env.scene[cfg.asset_name]
        self.body_asset_cfg = cfg.body_asset_cfg
        self.body_asset_cfg.resolve(env.scene)

        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)
        self._goal_succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._new_success_this_step = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._set_command_from_current_object(torch.arange(self.num_envs, device=self.device))

    def __str__(self) -> str:
        msg = "HandFrameReOrientationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """Desired object pose in the configured body frame. Shape is ``(num_envs, 7)``."""
        return self.pose_command_b

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        env_ids_tensor = self._env_ids_tensor(env_ids)
        self._set_command_from_current_object(env_ids_tensor)
        self._hold_counter[env_ids_tensor] = 0
        self._goal_succeeded[env_ids_tensor] = False
        self._new_success_this_step[env_ids_tensor] = False
        return super().reset(env_ids_tensor)

    def _update_metrics(self):
        self._update_pose_command_w()
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.pose_command_w[:, 3:7]
        )
        self.metrics["position_error"] = torch.norm(
            self.object.data.root_pos_w - self.pose_command_w[:, :3], dim=1
        )

        successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        if self.cfg.use_position_success:
            successes = successes & (self.metrics["position_error"] < self.cfg.position_success_threshold)

        new_success = successes & ~self._goal_succeeded
        self._new_success_this_step = new_success
        self.metrics["consecutive_success"] += new_success.float()
        self._goal_succeeded |= successes

    def _resample_command(self, env_ids: Sequence[int]):
        self._goal_succeeded[env_ids] = False
        self._hold_counter[env_ids] = 0

        n = len(env_ids)
        min_rad, max_rad = self.cfg.random_range
        lo = 0.5 * (min_rad - _math.sin(min_rad))
        hi = 0.5 * (max_rad - _math.sin(max_rad))
        u = torch.rand((n,), device=self.device) * (hi - lo) + lo
        theta = (12 * u).clamp(min=1e-8).pow(1 / 3)
        f = 0.5 * (theta - torch.sin(theta)) - u
        df = 0.5 * (1 - torch.cos(theta))
        angle = (theta - f / df.clamp(min=1e-6)).clamp(min_rad, max_rad)
        axis = torch.randn((n, 3), device=self.device)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        quat_delta = math_utils.quat_from_angle_axis(angle, axis)

        _, body_quat_w, _, _ = _body_frame_state_w(self._env, self.body_asset_cfg, env_ids)
        object_quat_b = quat_mul(quat_inv(body_quat_w), self.object.data.root_quat_w[env_ids])
        quat = quat_mul(object_quat_b, quat_delta)
        self.pose_command_b[env_ids, 3:7] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat
        self._update_pose_command_w(env_ids)

    def _update_command(self):
        if self.cfg.resample_on != "success":
            return

        successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        if self.cfg.use_position_success:
            successes = successes & (self.metrics["position_error"] < self.cfg.position_success_threshold)

        hold_steps = self.cfg.hold_steps_on_success
        prev_holding = self._hold_counter > 0
        self._hold_counter[prev_holding] -= 1
        hold_done = prev_holding & (self._hold_counter == 0)

        if hold_steps > 0:
            new_first_success = self._new_success_this_step & ~prev_holding
            self._hold_counter[new_first_success] = hold_steps
            immediate_resample = successes & ~prev_holding & ~new_first_success
        else:
            immediate_resample = successes & ~prev_holding

        resample_ids = (hold_done | immediate_resample).nonzero(as_tuple=False).squeeze(-1)
        if len(resample_ids) > 0:
            self._resample(resample_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            if not hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self._update_pose_command_w()
        marker_pos = self.pose_command_w[:, :3] + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=self.pose_command_w[:, 3:7])
        self.current_pose_visualizer.visualize(
            translations=self.object.data.root_pos_w, orientations=self.object.data.root_quat_w
        )

    def _set_command_from_current_object(self, env_ids: torch.Tensor) -> None:
        body_pos_w, body_quat_w, _, _ = _body_frame_state_w(self._env, self.body_asset_cfg, env_ids)
        self.pose_command_b[env_ids, :3] = quat_apply_inverse(
            body_quat_w, self.object.data.root_pos_w[env_ids] - body_pos_w
        )
        self.pose_command_b[env_ids, 3:7] = quat_mul(quat_inv(body_quat_w), self.object.data.root_quat_w[env_ids])
        self._update_pose_command_w(env_ids)

    def _update_pose_command_w(self, env_ids: torch.Tensor | None = None) -> None:
        body_pos_w, body_quat_w, _, _ = _body_frame_state_w(self._env, self.body_asset_cfg, env_ids)
        idx = slice(None) if env_ids is None else env_ids
        self.pose_command_w[idx, :3] = body_pos_w + quat_apply(body_quat_w, self.pose_command_b[idx, :3])
        self.pose_command_w[idx, 3:7] = quat_mul(body_quat_w, self.pose_command_b[idx, 3:7])

    def _env_ids_tensor(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class HandFrameReOrientationCommandCfg(CommandTermCfg):
    """Configuration for hand-frame object reorientation commands."""

    class_type: type = HandFrameReOrientationCommand

    random_range: tuple[float, float] = (0.0, 1.0)
    """Min/max geodesic angle in radians for goal orientation sampling."""

    resampling_time_range: tuple[float, float] = (1e6, 1e6)
    """Large default so success-driven resampling controls the goal."""

    asset_name: str = MISSING
    """Name of the object asset for which commands are generated."""

    body_asset_cfg: SceneEntityCfg = MISSING
    """Articulation body frame in which the command is expressed."""

    make_quat_unique: bool = MISSING
    """Whether to enforce a positive real quaternion component."""

    orientation_success_threshold: float = MISSING
    """Orientation error threshold for goal success."""

    use_position_success: bool = False
    """Whether success also requires position error below threshold."""

    position_success_threshold: float = 0.05
    """Position error threshold when position success is enabled."""

    hold_steps_on_success: int = 0
    """Number of control steps to hold a newly reached goal before resampling."""

    resample_on: Literal["success", "time"] = "success"
    """When to resample the goal command."""

    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-frame debug marker offset applied to the goal marker."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )
    """Goal pose visualization marker."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/current_marker",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            ),
        },
    )
    """Current object pose visualization marker."""


def reset_object_pose_relative_to_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    local_pos: tuple[float, float, float] = (0.0, -0.08, 0.12),
    pose_range: dict[str, tuple[float, float]] | None = None,
    local_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    """Reset an object pose from a local pose expressed in an articulation body frame."""

    if env_ids is None or env_ids == slice(None):
        env_ids = torch.arange(env.num_envs, device=env.device)

    body_pos_w, body_quat_w, _, _ = _body_frame_state_w(env, body_asset_cfg, env_ids)
    object_asset: RigidObject = env.scene[object_cfg.name]

    pose_range = pose_range or {}
    local_pos_tensor = torch.tensor(local_pos, device=env.device, dtype=torch.float32).repeat(len(env_ids), 1)
    local_pos_tensor[:, 0] += _sample_range(env, env_ids, pose_range.get("x", (0.0, 0.0)))
    local_pos_tensor[:, 1] += _sample_range(env, env_ids, pose_range.get("y", (0.0, 0.0)))
    local_pos_tensor[:, 2] += _sample_range(env, env_ids, pose_range.get("z", (0.0, 0.0)))

    roll = _sample_range(env, env_ids, pose_range.get("roll", (0.0, 0.0)))
    pitch = _sample_range(env, env_ids, pose_range.get("pitch", (0.0, 0.0)))
    yaw = _sample_range(env, env_ids, pose_range.get("yaw", (0.0, 0.0)))
    local_rot_tensor = torch.tensor(local_rot, device=env.device, dtype=torch.float32).repeat(len(env_ids), 1)
    local_quat = quat_mul(local_rot_tensor, quat_from_euler_xyz(roll, pitch, yaw))

    object_pos_w = body_pos_w + quat_apply(body_quat_w, local_pos_tensor)
    object_quat_w = quat_mul(body_quat_w, local_quat)
    object_asset.write_root_pose_to_sim(torch.cat((object_pos_w, object_quat_w), dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=env.device), env_ids=env_ids)


def body_state_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    base_body_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Body states expressed in another body frame of the same or another articulation."""

    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_pos_w, base_quat_w, base_lin_vel_w, base_ang_vel_w = _body_frame_state_w(env, base_body_asset_cfg)
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].reshape(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].reshape(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].reshape(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].reshape(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)

    base_pos_w = base_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 3)
    base_quat_w = base_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 4)
    base_lin_vel_w = base_lin_vel_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 3)
    base_ang_vel_w = base_ang_vel_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 3)

    pos_b, quat_b = math_utils.subtract_frame_transforms(base_pos_w, base_quat_w, body_pos_w, body_quat_w)
    lin_vel_b = quat_apply_inverse(base_quat_w, body_lin_vel_w - base_lin_vel_w)
    ang_vel_b = quat_apply_inverse(base_quat_w, body_ang_vel_w - base_ang_vel_w)
    out = torch.cat(
        (
            pos_b.reshape(env.num_envs, num_bodies, 3),
            quat_b.reshape(env.num_envs, num_bodies, 4),
            lin_vel_b.reshape(env.num_envs, num_bodies, 3),
            ang_vel_b.reshape(env.num_envs, num_bodies, 3),
        ),
        dim=-1,
    )
    return out.reshape(env.num_envs, -1)


def object_pos_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    base_pos_w, base_quat_w, _, _ = _body_frame_state_w(env, body_asset_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(base_quat_w, obj.data.root_pos_w - base_pos_w)


def object_quat_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    _, base_quat_w, _, _ = _body_frame_state_w(env, body_asset_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(base_quat_w), obj.data.root_quat_w)


def object_lin_vel_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    _, base_quat_w, base_lin_vel_w, _ = _body_frame_state_w(env, body_asset_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(base_quat_w, obj.data.root_lin_vel_w - base_lin_vel_w)


def object_ang_vel_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    _, base_quat_w, _, base_ang_vel_w = _body_frame_state_w(env, body_asset_cfg)
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(base_quat_w, obj.data.root_ang_vel_w - base_ang_vel_w)


def gravity_dir_body_b(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    _, base_quat_w, _, _ = _body_frame_state_w(env, body_asset_cfg)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, -1)
    return quat_apply_inverse(base_quat_w, gravity_w)


def fingers_contact_force_body_b(
    env: ManagerBasedRLEnv,
    contact_sensor_names: list[str],
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    forces_w = []
    for name in contact_sensor_names:
        sensor = env.scene.sensors[name]
        force_matrix_w = getattr(sensor.data, "force_matrix_w", None)
        if force_matrix_w is not None and force_matrix_w.numel() > 0:
            force_w = torch.nan_to_num(force_matrix_w, nan=0.0).sum(dim=(1, 2))
        else:
            force_w = torch.nan_to_num(sensor.data.net_forces_w, nan=0.0).sum(dim=1)
        forces_w.append(force_w)
    force_w = torch.stack(forces_w, dim=1)
    _, base_quat_w, _, _ = _body_frame_state_w(env, body_asset_cfg)
    forces_b = quat_apply_inverse(base_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b.reshape(env.num_envs, -1)


def goal_pos_diff_body_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    goal_pos_w, _ = _command_pose_w(env, command_name)
    base_pos_w, base_quat_w, _, _ = _body_frame_state_w(env, body_asset_cfg)
    asset_pos_b = quat_apply_inverse(base_quat_w, asset.data.root_pos_w - base_pos_w)
    goal_pos_b = quat_apply_inverse(base_quat_w, goal_pos_w - base_pos_w)
    return asset_pos_b - goal_pos_b


def goal_quat_diff_body_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    make_quat_unique: bool,
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    _, goal_quat_w = _command_pose_w(env, command_name)
    _, base_quat_w, _, _ = _body_frame_state_w(env, body_asset_cfg)
    base_quat_inv = quat_inv(base_quat_w)
    asset_quat_b = quat_mul(base_quat_inv, asset.data.root_quat_w)
    goal_quat_b = quat_mul(base_quat_inv, goal_quat_w)
    quat = quat_mul(asset_quat_b, math_utils.quat_conjugate(goal_quat_b))
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def _sample_range(env: ManagerBasedRLEnv, env_ids: torch.Tensor, value_range: tuple[float, float]) -> torch.Tensor:
    return torch.empty(len(env_ids), device=env.device).uniform_(*value_range)


def _body_frame_state_w(
    env: ManagerBasedRLEnv,
    body_asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[body_asset_cfg.name]
    idx = slice(None) if env_ids is None else env_ids
    if getattr(body_asset_cfg, "body_ids", None) is not None:
        body_id = body_asset_cfg.body_ids[0]
        body_lin_vel_w = getattr(asset.data, "body_lin_vel_w", torch.zeros_like(asset.data.body_pos_w))
        body_ang_vel_w = getattr(asset.data, "body_ang_vel_w", torch.zeros_like(asset.data.body_pos_w))
        return (
            asset.data.body_pos_w[idx, body_id],
            asset.data.body_quat_w[idx, body_id],
            body_lin_vel_w[idx, body_id],
            body_ang_vel_w[idx, body_id],
        )
    root_lin_vel_w = getattr(asset.data, "root_lin_vel_w", torch.zeros_like(asset.data.root_pos_w))
    root_ang_vel_w = getattr(asset.data, "root_ang_vel_w", torch.zeros_like(asset.data.root_pos_w))
    return (
        asset.data.root_pos_w[idx],
        asset.data.root_quat_w[idx],
        root_lin_vel_w[idx],
        root_ang_vel_w[idx],
    )


def _command_pose_w(env: ManagerBasedRLEnv, command_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    command_term = env.command_manager.get_term(command_name)
    command = command_term.command
    if hasattr(command_term, "pose_command_w"):
        return command_term.pose_command_w[:, :3], command_term.pose_command_w[:, 3:]
    if hasattr(command_term, "pose_command_b") and hasattr(command_term, "robot"):
        return math_utils.combine_frame_transforms(
            command_term.robot.data.root_pos_w,
            command_term.robot.data.root_quat_w,
            command[:, :3],
            command[:, 3:7],
        )
    return command[:, :3] + env.scene.env_origins, command[:, 3:7]


def object_lifted_above_table(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    height: float = 0.06,
    table_half_height: float = 0.02,
) -> torch.Tensor:
    """Reward object lift height above the table top, clamped to [0, 1]."""

    object: RigidObject = env.scene[object_cfg.name]
    table: RigidObject = env.scene[table_cfg.name]
    table_top_z = table.data.root_pos_w[:, 2] + table_half_height
    height_above_table = object.data.root_pos_w[:, 2] - table_top_z
    return torch.clamp(height_above_table / height, 0.0, 1.0)


def object_to_hole_xy_tanh(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    std: float = 0.08,
    contact_threshold: float = 1.0,
    lift_height: float = 0.06,
    lift_gate: float = 0.5,
) -> torch.Tensor:
    """Reward XY alignment between the object and hole, gated by grasp and lift."""

    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    xy_dist = torch.norm(object.data.root_pos_w[:, :2] - hole.data.root_pos_w[:, :2], dim=1)
    reward = 1.0 - torch.tanh(xy_dist / std)
    return reward * _pick_insert_gate(env, table_cfg, contact_threshold, lift_height, lift_gate)


def peg_hole_axis_alignment(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    std: float = 0.35,
) -> torch.Tensor:
    """Reward alignment of the peg and hole local Z axes."""

    axis_dot = _peg_hole_axis_dot(env, object_cfg, hole_cfg)
    axis_error = 1.0 - axis_dot
    return 1.0 - torch.tanh(axis_error / std)


def peg_insertion_depth(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    table_cfg: SceneEntityCfg = SceneEntityCfg("table"),
    target_depth: float = 0.015,
    approach_height: float = 0.08,
    xy_tolerance: float = 0.04,
    axis_tolerance: float = 0.25,
    contact_threshold: float = 1.0,
    lift_height: float = 0.06,
    lift_gate: float = 0.5,
) -> torch.Tensor:
    """Reward downward insertion progress once grasped, lifted, centered, and aligned."""

    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    target_z = hole.data.root_pos_w[:, 2] + target_depth
    approach_z = target_z + approach_height
    progress = torch.clamp((approach_z - object.data.root_pos_w[:, 2]) / approach_height, 0.0, 1.0)

    xy_dist = torch.norm(object.data.root_pos_w[:, :2] - hole.data.root_pos_w[:, :2], dim=1)
    axis_error = 1.0 - _peg_hole_axis_dot(env, object_cfg, hole_cfg)
    insertion_gate = ((xy_dist < xy_tolerance) & (axis_error < axis_tolerance)).float()
    return progress * insertion_gate * _pick_insert_gate(env, table_cfg, contact_threshold, lift_height, lift_gate)


def peg_inserted_success(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    hole_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
    pos_tol: float = 0.015,
    axis_tol: float = 0.15,
    depth: float = 0.015,
) -> torch.Tensor:
    """Sparse success for a peg centered in the hole at the inserted height."""

    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    target_pos = hole.data.root_pos_w.clone()
    target_pos[:, 2] = target_pos[:, 2] + depth
    pos_dist = torch.norm(object.data.root_pos_w - target_pos, dim=1)
    axis_error = 1.0 - _peg_hole_axis_dot(env, object_cfg, hole_cfg)
    return ((pos_dist < pos_tol) & (axis_error < axis_tol)).float()


def _pick_insert_gate(
    env: ManagerBasedRLEnv,
    table_cfg: SceneEntityCfg,
    contact_threshold: float,
    lift_height: float,
    lift_gate: float,
) -> torch.Tensor:
    contact = _good_finger_contact(env, contact_threshold).float()
    lifted = object_lifted_above_table(env, table_cfg=table_cfg, height=lift_height)
    return contact * (lifted >= lift_gate).float()


def _good_finger_contact(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    thumb_contact = _contact_magnitude(env.scene.sensors["thumb_fingertip_object_s"])
    index_contact = _contact_magnitude(env.scene.sensors["fingertip_object_s"])
    middle_contact = _contact_magnitude(env.scene.sensors["fingertip_2_object_s"])
    ring_contact = _contact_magnitude(env.scene.sensors["fingertip_3_object_s"])
    return (thumb_contact > threshold) & (
        (index_contact > threshold)
        | (middle_contact > threshold)
        | (ring_contact > threshold)
    )


def _contact_magnitude(sensor: ContactSensor) -> torch.Tensor:
    force_w = torch.nan_to_num(sensor.data.force_matrix_w, nan=0.0).reshape(sensor.data.force_matrix_w.shape[0], -1, 3)
    return torch.norm(force_w.sum(dim=1), dim=-1)


def _peg_hole_axis_dot(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg,
    hole_cfg: SceneEntityCfg,
) -> torch.Tensor:
    object: RigidObject = env.scene[object_cfg.name]
    hole: RigidObject = env.scene[hole_cfg.name]
    local_z = torch.zeros(env.num_envs, 3, device=object.data.root_quat_w.device)
    local_z[:, 2] = 1.0
    object_axis = quat_apply(object.data.root_quat_w, local_z)
    hole_axis = quat_apply(hole.data.root_quat_w, local_z)
    return torch.sum(object_axis * hole_axis, dim=1).abs().clamp(0.0, 1.0)
