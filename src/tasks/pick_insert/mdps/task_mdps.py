from __future__ import annotations

from dataclasses import MISSING
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_inv,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


class CommandHandBaseIKAction(ActionTerm):
    """Automatic IK action that tracks the hand-base pose stored in a command."""

    cfg: CommandHandBaseIKActionCfg
    _asset: Articulation

    def __init__(self, cfg: CommandHandBaseIKActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._ik_controller.action_dim, device=self.device)
        self._target_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._target_pose_b[:, 3] = 1.0
        self._step_limit = torch.tensor(self.cfg.scale, device=self.device).repeat(self.num_envs, 1)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def target_pose_b(self) -> torch.Tensor:
        return self._target_pose_b

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot_matrix = matrix_from_quat(quat_inv(self._asset.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != 0:
            raise ValueError(f"Expected zero external arm action dims, got {actions.shape[-1]}.")

        command = self._env.command_manager.get_command(self.cfg.command_name)
        command_end = self.cfg.command_start + 7
        if command.shape[-1] < command_end:
            raise ValueError(
                f"Command {self.cfg.command_name!r} must contain hand-base pose slice "
                f"{self.cfg.command_start}:{command_end}, got shape {tuple(command.shape)}."
            )
        self._target_pose_b[:] = command[:, self.cfg.command_start : command_end]

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        pos_error, axis_angle_error = compute_pose_error(
            ee_pos_curr,
            ee_quat_curr,
            self._target_pose_b[:, :3],
            self._target_pose_b[:, 3:7],
            rot_error_type="axis_angle",
        )
        relative_command = torch.cat((pos_error, axis_angle_error), dim=1)
        self._processed_actions[:] = torch.clamp(relative_command, -self._step_limit, self._step_limit)
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self.jacobian_b
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._ik_controller.reset(env_ids=env_ids)

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        return subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )


@configclass
class CommandHandBaseIKActionCfg(ActionTermCfg):
    """Configuration for command-driven hand-base IK tracking."""

    class_type: type[ActionTerm] = CommandHandBaseIKAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions controlled by IK."""

    body_name: str = MISSING
    """Body name to track with IK."""

    command_name: str = "object_pose"
    """Command term containing the target hand-base pose."""

    command_start: int = 7
    """Start index of the target hand-base pose in the command tensor."""

    scale: tuple[float, float, float, float, float, float] = (0.05, 0.05, 0.05, 0.25, 0.25, 0.25)
    """Maximum relative IK command per environment step."""

    controller: DifferentialIKControllerCfg = MISSING
    """Differential IK controller configuration."""


class CommandHandBaseOSCAction(ActionTerm):
    """Variable-impedance OSC action that holds the hand base at the command anchor pose.

    The arm is driven by an :class:`OperationalSpaceController` whose equilibrium is the
    target hand-base pose carried in the command (``command[:, command_start:command_start+7]``,
    expressed in the robot root frame). The task-space stiffness behaves as the *cost* of
    deviating from that anchor: it is high in free motion (precise anchor tracking) and is
    automatically reduced when the wrist meets resistance, so the arm compliantly yields to
    let contact (e.g. a peg entering a hole) guide the motion. Rotational stiffness is kept
    higher than translational, encoding the requirement that rotating away from the anchor
    costs more than translating.

    The resistance signal is the wrist force/torque read from the articulation's built-in
    joint-reaction wrench at the controlled body (``body_incoming_joint_wrench_b``). A slow
    EMA of the free-motion wrench is used as a baseline so only the *external/contact*
    component drives the softening; a deadband plus stiffness EMA keep the loop stable.

    Like :class:`CommandHandBaseIKAction`, this term consumes zero external action dims; the
    high-level behavior is fully determined by the command and the internal stiffness law.
    """

    cfg: CommandHandBaseOSCActionCfg
    _asset: Articulation

    def __init__(self, cfg: CommandHandBaseOSCActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_dof = len(self._joint_ids)
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]
        # Keep an explicit joint-id list for indexing dynamic quantities; only collapse the
        # control index to a slice when this term owns every joint of the articulation.
        self._dyn_joint_ids = self._joint_ids
        if self._num_dof == self._asset.num_joints:
            self._joint_ids = slice(None)

        self._osc = OperationalSpaceController(cfg=self.cfg.controller_cfg, num_envs=self.num_envs, device=self.device)

        # buffers
        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._target_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._target_pose_b[:, 3] = 1.0
        self._jacobian_b = torch.zeros(self.num_envs, 6, self._num_dof, device=self.device)
        self._mass_matrix = torch.zeros(self.num_envs, self._num_dof, self._num_dof, device=self.device)
        self._gravity = torch.zeros(self.num_envs, self._num_dof, device=self.device)
        self._ee_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_vel_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, self._num_dof, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, self._num_dof, device=self.device)
        self._joint_efforts = torch.zeros(self.num_envs, self._num_dof, device=self.device)

        # variable-impedance state: stiffness (per task axis) and slow wrench baseline.
        self._stiffness = torch.zeros(self.num_envs, 6, device=self.device)
        self._wrench_baseline = torch.zeros(self.num_envs, 6, device=self.device)
        # Logging-only buffers: last raw wrist wrench and its external (contact) component.
        self._wrench_raw = torch.zeros(self.num_envs, 6, device=self.device)
        self._wrench_ext = torch.zeros(self.num_envs, 6, device=self.device)
        self._reset_stiffness(slice(None))

        if self.cfg.controller_cfg.nullspace_control == "position":
            self._nullspace_joint_pos_target = torch.mean(
                self._asset.data.soft_joint_pos_limits[:, self._dyn_joint_ids, :], dim=-1
            )
        else:
            self._nullspace_joint_pos_target = None

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def stiffness(self) -> torch.Tensor:
        return self._stiffness

    @property
    def external_wrench(self) -> torch.Tensor:
        """Last external (contact) wrist wrench driving the softening: force ``[:, :3]``,
        torque ``[:, 3:]`` (raw wrench minus the free-motion baseline)."""
        return self._wrench_ext

    @property
    def raw_wrench(self) -> torch.Tensor:
        """Last raw wrist reaction wrench at the controlled body: force ``[:, :3]``,
        torque ``[:, 3:]``."""
        return self._wrench_raw

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot_matrix = matrix_from_quat(quat_inv(self._asset.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != 0:
            raise ValueError(f"Expected zero external arm action dims, got {actions.shape[-1]}.")

        command = self._env.command_manager.get_command(self.cfg.command_name)
        command_end = self.cfg.command_start + 7
        if command.shape[-1] < command_end:
            raise ValueError(
                f"Command {self.cfg.command_name!r} must contain hand-base pose slice "
                f"{self.cfg.command_start}:{command_end}, got shape {tuple(command.shape)}."
            )
        self._target_pose_b[:] = command[:, self.cfg.command_start : command_end]

        self._compute_ee_pose()
        self._update_variable_stiffness()

        osc_command = torch.cat((self._target_pose_b, self._stiffness), dim=-1)
        self._osc.set_command(command=osc_command, current_ee_pose_b=self._ee_pose_b, current_task_frame_pose_b=None)

    def apply_actions(self):
        self._compute_dynamic_quantities()
        self._jacobian_b[:] = self.jacobian_b
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._joint_pos[:] = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_vel[:] = self._asset.data.joint_vel[:, self._joint_ids]
        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=None,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=self._nullspace_joint_pos_target,
        )
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._osc.reset()
        self._wrench_baseline[env_ids] = 0.0
        self._reset_stiffness(env_ids)

    def _reset_stiffness(self, env_ids) -> None:
        self._stiffness[env_ids, :3] = self.cfg.stiffness_max_trans
        self._stiffness[env_ids, 3:] = self.cfg.stiffness_max_rot

    def _update_variable_stiffness(self) -> None:
        """Lower task-space stiffness on axes where the wrist meets sustained resistance."""
        wrench = self._asset.data.body_incoming_joint_wrench_b[:, self._body_idx]
        wrench_ext = wrench - self._wrench_baseline
        # Stash for diagnostics (logging-only; does not affect the control law).
        self._wrench_raw[:] = wrench
        self._wrench_ext[:] = wrench_ext

        force_mag = torch.norm(wrench_ext[:, :3], dim=-1, keepdim=True)
        torque_mag = torch.norm(wrench_ext[:, 3:], dim=-1, keepdim=True)
        force_excess = torch.clamp(force_mag - self.cfg.force_deadband, min=0.0)
        torque_excess = torch.clamp(torque_mag - self.cfg.torque_deadband, min=0.0)

        # soften in [0, 1]: 0 keeps full stiffness, 1 collapses to the floor.
        soften_trans = torch.clamp(force_excess / self.cfg.force_scale, max=1.0)
        soften_rot = torch.clamp(torque_excess / self.cfg.torque_scale, max=1.0)
        k_trans = self.cfg.stiffness_max_trans - (self.cfg.stiffness_max_trans - self.cfg.stiffness_min) * soften_trans
        k_rot = self.cfg.stiffness_max_rot - (self.cfg.stiffness_max_rot - self.cfg.stiffness_min) * soften_rot
        target = torch.cat((k_trans.expand(-1, 3), k_rot.expand(-1, 3)), dim=-1)

        beta = self.cfg.stiffness_ema
        self._stiffness[:] = (1.0 - beta) * self._stiffness + beta * target

        # Only learn the free-motion baseline while there is no detected contact; freeze it
        # during contact so the external estimate does not bleed away.
        in_free = (force_excess <= 0.0) & (torque_excess <= 0.0)
        update = in_free.to(wrench.dtype) * self.cfg.wrench_baseline_ema
        self._wrench_baseline[:] = (1.0 - update) * self._wrench_baseline + update * wrench

    def _compute_dynamic_quantities(self) -> None:
        self._mass_matrix[:] = self._asset.root_physx_view.get_generalized_mass_matrices()[
            :, self._dyn_joint_ids, :
        ][:, :, self._dyn_joint_ids]
        self._gravity[:] = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._dyn_joint_ids]

    def _compute_ee_pose(self) -> None:
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        self._ee_pose_b[:, :3], self._ee_pose_b[:, 3:] = subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )

    def _compute_ee_velocity(self) -> None:
        ee_vel_w = self._asset.data.body_vel_w[:, self._body_idx, :]
        relative_vel_w = ee_vel_w - self._asset.data.root_vel_w
        self._ee_vel_b[:, 0:3] = quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 0:3])
        self._ee_vel_b[:, 3:6] = quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 3:6])


@configclass
class CommandHandBaseOSCActionCfg(ActionTermCfg):
    """Configuration for command-driven hand-base variable-impedance OSC control."""

    class_type: type[ActionTerm] = CommandHandBaseOSCAction

    joint_names: list[str] = MISSING
    """List of arm joint names or regex expressions controlled by OSC (effort control)."""

    body_name: str = MISSING
    """Body name whose pose is regulated to the command anchor pose."""

    command_name: str = "object_pose"
    """Command term containing the target hand-base (anchor) pose."""

    command_start: int = 7
    """Start index of the target hand-base pose in the command tensor."""

    controller_cfg: OperationalSpaceControllerCfg = MISSING
    """Operational-space controller configuration. Use ``impedance_mode="variable_kp"`` and
    ``target_types=["pose_abs"]`` so the term can supply a per-step 6-axis stiffness."""

    stiffness_max_trans: float = 500.0
    """Free-motion translational task-space stiffness (precise anchor tracking)."""

    stiffness_max_rot: float = 500.0
    """Free-motion rotational task-space stiffness. Higher than translation so rotating away
    from the anchor costs more."""

    stiffness_min: float = 30.0
    """Stiffness floor reached under strong sustained resistance (maximally compliant)."""

    force_deadband: float = 3.0
    """External wrist force (N) below which no translational softening occurs."""

    force_scale: float = 20.0
    """External force (N) above the deadband that drives translational stiffness to the floor."""

    torque_deadband: float = 0.5
    """External wrist torque (Nm) below which no rotational softening occurs."""

    torque_scale: float = 3.0
    """External torque (Nm) above the deadband that drives rotational stiffness to the floor."""

    stiffness_ema: float = 0.05
    """EMA factor for the per-step stiffness update (smaller = slower, more hysteresis)."""

    wrench_baseline_ema: float = 0.02
    """EMA factor for learning the free-motion wrench baseline (frozen during contact)."""


def reset_object_pose_relative_to_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    body_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    local_pos: tuple[float, float, float] = (0.12, 0.0, 0.08),
    local_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    random_orientation: bool = False,
    velocity: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
):
    """Reset an object to a pose expressed relative to one articulated body."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_asset: Articulation = env.scene[body_asset_cfg.name]

    env_ids = _env_ids_tensor(env, env_ids, device=asset.device)
    body_pos_w, body_quat_w = _single_body_state_w(body_asset, body_asset_cfg, env_ids)[:2]

    local_pos_t = torch.tensor(local_pos, dtype=torch.float32, device=asset.device).repeat(len(env_ids), 1)
    if random_orientation:
        local_rot_t = math_utils.random_orientation(len(env_ids), device=asset.device)
    else:
        local_rot_t = torch.tensor(local_rot, dtype=torch.float32, device=asset.device).repeat(len(env_ids), 1)

    object_pos_w, object_quat_w = combine_frame_transforms(body_pos_w, body_quat_w, local_pos_t, local_rot_t)
    object_velocity = torch.tensor(velocity, dtype=torch.float32, device=asset.device).repeat(len(env_ids), 1)

    asset.write_root_pose_to_sim(torch.cat((object_pos_w, object_quat_w), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(object_velocity, env_ids=env_ids)


def _env_ids_tensor(env: ManagerBasedRLEnv, env_ids, device: str) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=device)


def _single_body_state_w(
    asset: Articulation,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    body_ids = getattr(asset_cfg, "body_ids", None)
    if body_ids is None:
        body_names = getattr(asset_cfg, "body_names", None)
        body_ids, body_names = asset.find_bodies(body_names)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body matching {body_names}, found {len(body_ids)}.")

    body_pos_w = asset.data.body_pos_w[env_ids][:, body_ids]
    body_quat_w = asset.data.body_quat_w[env_ids][:, body_ids]
    if body_pos_w.ndim == 2:
        body_pos_w = body_pos_w.unsqueeze(1)
        body_quat_w = body_quat_w.unsqueeze(1)
    if body_pos_w.shape[1] != 1:
        raise ValueError(f"Expected one body id for {asset_cfg.name}, found {body_pos_w.shape[1]}.")

    body_lin_vel_w = getattr(asset.data, "body_lin_vel_w", None)
    body_ang_vel_w = getattr(asset.data, "body_ang_vel_w", None)
    if body_lin_vel_w is not None:
        body_lin_vel_w = body_lin_vel_w[env_ids][:, body_ids]
        if body_lin_vel_w.ndim == 2:
            body_lin_vel_w = body_lin_vel_w.unsqueeze(1)
    if body_ang_vel_w is not None:
        body_ang_vel_w = body_ang_vel_w[env_ids][:, body_ids]
        if body_ang_vel_w.ndim == 2:
            body_ang_vel_w = body_ang_vel_w.unsqueeze(1)

    return body_pos_w[:, 0], body_quat_w[:, 0], body_lin_vel_w, body_ang_vel_w


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
