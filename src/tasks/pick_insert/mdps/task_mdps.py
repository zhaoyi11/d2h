from __future__ import annotations

import math
from dataclasses import MISSING, field
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


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


def move_dynamic_obstacle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("dynamic_obstacle"),
    center: tuple[float, float, float] = (0.4, 0.05, 0.45),
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    amplitude: float = 0.2,
    freq: float = 0.25,
    phase: float = 0.0,
) -> None:
    """Kinematically drive a scene prop along a sinusoid for use as a moving MPC obstacle.

    Per-env (identical across envs) world pose is
        ``pos = env_origin + center + axis * amplitude * sin(2*pi*freq*t + phase)``
    with ``t`` the GLOBAL sim time (``common_step_counter * step_dt``), so every env and the single
    shared cuRobo collision world agree on the obstacle pose. Orientation is identity. Intended as
    an every-step ``interval`` event (``interval_range_s=(0.0, 0.0)``). The matching cuRobo cuboid
    is updated separately by :class:`CommandHandBaseCuroboMpcAction`, which reads this prop's live
    pose each control step (see ``dynamic_obstacle_assets``).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids = _env_ids_tensor(env, env_ids, device=asset.device)

    t = float(env.common_step_counter) * env.step_dt
    offset = math.sin(2.0 * math.pi * freq * t + phase) * amplitude
    center_t = torch.tensor(center, dtype=torch.float32, device=asset.device)
    axis_t = torch.tensor(axis, dtype=torch.float32, device=asset.device)
    pos_local = center_t + axis_t * offset  # (3,) relative to each env origin

    pos_w = env.scene.env_origins[env_ids] + pos_local  # (n, 3)
    quat_w = torch.zeros(len(env_ids), 4, dtype=torch.float32, device=asset.device)
    quat_w[:, 0] = 1.0  # identity orientation (w, x, y, z)
    asset.write_root_pose_to_sim(torch.cat((pos_w, quat_w), dim=-1), env_ids=env_ids)


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


class CommandHandBaseCuroboMpcAction(ActionTerm):
    """Command-driven arm controller that drives the hand ``base`` to the command anchor with
    cuRobo reactive MPC, avoiding obstacles (e.g. the table) with the full Franka+LEAP model.

    Consumes zero external action dims: the goal is the
    hand-base anchor pose carried in the command (``command[:, command_start:command_start+7]``,
    in the robot root frame). Each control step it sets that pose as the MPC tool-pose goal,
    optimizes a short horizon (cuRobo 0.8 ``ModelPredictiveControl``), and applies the resulting
    arm joint-position targets. The LEAP hand joints are locked in cuRobo's model (a rigid
    collision envelope) and driven separately by ``hand_action``.

    cuRobo is imported lazily so importing this module does not pull cuRobo/Warp for other tasks.
    """

    cfg: "CommandHandBaseCuroboMpcActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "CommandHandBaseCuroboMpcActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # cuRobo 0.8 imports (after the Isaac app + CUDA are up)
        import yaml
        from curobo.model_predictive_control import (
            ModelPredictiveControl,
            ModelPredictiveControlCfg,
        )
        from curobo.types import GoalToolPose, JointState, Pose

        self._GoalToolPose = GoalToolPose
        self._JointState = JointState
        self._Pose = Pose

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        # Order the controlled-joint indices to match cuRobo's joint_names (panda_joint1..7).
        robot_dict = yaml.safe_load(open(self.cfg.robot_config_file))
        curobo_joint_names = robot_dict["robot_cfg"]["kinematics"]["cspace"]["joint_names"]
        name_to_id = {n: i for n, i in zip(self._joint_names, self._joint_ids)}
        self._ordered_joint_ids = [name_to_id[n] for n in curobo_joint_names]

        # Scene model: obstacle cuboids in the robot base frame (receptacle deliberately excluded).
        scene_model = {"cuboid": dict(self.cfg.obstacle_cuboids)}

        # Dynamic obstacles: scene assets whose live pose is mirrored into the cuRobo world every
        # control step. Each must have a pre-declared slot in obstacle_cuboids so a collision-world
        # buffer exists at build time (the pose is then rewritten in-place, CUDA-graph safe).
        for curobo_name in self.cfg.dynamic_obstacle_assets:
            if curobo_name not in self.cfg.obstacle_cuboids:
                raise ValueError(
                    f"dynamic_obstacle_assets cuboid '{curobo_name}' must also be declared in "
                    f"obstacle_cuboids (no collision-world slot exists otherwise)."
                )
        self._dyn_obstacles = {
            curobo_name: self._env.scene[asset_name]
            for curobo_name, asset_name in self.cfg.dynamic_obstacle_assets.items()
        }

        # Optimizer config: default is "mpc/lbfgs_mpc.yml". When a collision_weight override is set,
        # patch a COPY of that config (leaving the shared cuRobo yaml untouched). Lower weight ->
        # the arm prioritizes the goal over avoidance and deviates less near obstacles.
        # Load via cuRobo's resolve_config (NOT yaml.safe_load): the configs use scientific notation
        # like ``1e-3`` which the stock YAML resolver leaves as a *string*, crashing the CUDA kernel.
        optimizer_configs = ["mpc/lbfgs_mpc.yml"]
        if self.cfg.collision_weight is not None:
            from curobo._src.util.config_io import join_path, resolve_config
            from curobo.content import get_task_configs_path

            opt_dict = resolve_config(join_path(get_task_configs_path(), "mpc/lbfgs_mpc.yml"))
            opt_dict["rollout"]["constraint_cfg"]["scene_collision_cfg"]["weight"] = (
                self.cfg.collision_weight
            )
            optimizer_configs = [opt_dict]

        # MPC planning timestep. Kept deliberately LARGER than the env control dt: with
        # optimization_dt == env.step_dt (~0.0166s) `optimize_next_action` advances the plan too
        # slowly to ever reach a moving goal. ~0.05-0.1 makes each control step command meaningful
        # motion (validated headless).
        mpc_cfg = ModelPredictiveControlCfg.create(
            robot=robot_dict,
            scene_model=scene_model,
            optimizer_configs=optimizer_configs,
            optimization_dt=self.cfg.optimization_dt,
            interpolation_steps=self.cfg.interpolation_steps,
            use_cuda_graph=self.cfg.use_cuda_graph,
            self_collision_check=True,
            optimizer_collision_activation_distance=self.cfg.collision_activation_distance,
            max_batch_size=self.num_envs,
            multi_env=False,
        )
        self._mpc = ModelPredictiveControl(mpc_cfg)
        self._tool = self._mpc.tool_frames[0]

        # Setup MPC from the model's default config (real joint state arrives in process_actions).
        q0 = self._mpc.default_joint_position.clone().unsqueeze(0).repeat(self.num_envs, 1)
        state0 = JointState.from_position(q0, joint_names=self._mpc.joint_names)
        state0.velocity = torch.zeros_like(state0.position)
        state0.acceleration = torch.zeros_like(state0.position)
        self._mpc.setup(state0)

        # The MPC runs as a reference generator: it is stepped from its OWN advancing predicted
        # state (`_ref_js`), not the real (PD-lagging) arm state — feeding back the lagging real
        # state makes it re-plan from rest each step and crawl. The real arm tracks `_cmd_pos` via
        # its joint PD. `_ref_js` is re-synced to the real arm on reset.
        self._ref_js = state0.clone()

        self._raw_actions = torch.zeros(self.num_envs, 0, device=self.device)
        self._cmd_pos = q0.clone()
        self._cmd_vel = None
        self._target_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._target_pose_b[:, 3] = 1.0

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions

    def _arm_joint_state(self):
        # Feed the real joint position AND velocity so the MPC continues the trajectory with
        # momentum each control step (feeding position-only makes it re-plan from rest -> crawls).
        q = self._asset.data.joint_pos[:, self._ordered_joint_ids]
        qd = self._asset.data.joint_vel[:, self._ordered_joint_ids]
        js = self._JointState.from_position(q.contiguous(), joint_names=self._mpc.joint_names)
        js.velocity = qd.contiguous()
        js.acceleration = torch.zeros_like(js.velocity)
        return js

    def process_actions(self, actions: torch.Tensor):
        if actions.shape[-1] != 0:
            raise ValueError(f"Expected zero external arm action dims, got {actions.shape[-1]}.")

        command = self._env.command_manager.get_command(self.cfg.command_name)
        end = self.cfg.command_start + 7
        self._target_pose_b[:] = command[:, self.cfg.command_start : end]

        # Goal: hand-base anchor (robot root == panda_link0 base frame) as the tool-pose goal.
        goal_pose = self._Pose(
            position=self._target_pose_b[:, :3].contiguous(),
            quaternion=self._target_pose_b[:, 3:7].contiguous(),
        )
        goal = self._GoalToolPose.from_poses(
            {self._tool: goal_pose}, ordered_tool_frames=self._mpc.tool_frames, num_goalset=1
        )
        self._mpc.update_goal_tool_poses(goal, run_ik=False)

        # Mirror dynamic-obstacle scene props into the cuRobo world before optimizing. Pose is
        # converted to the robot base frame; update_obstacle_pose is an in-place tensor write, so it
        # is CUDA-graph safe. Single shared world (multi_env=False) -> env_idx=0 / first-env pose.
        for curobo_name, asset in self._dyn_obstacles.items():
            pos_b, quat_b = subtract_frame_transforms(
                self._asset.data.root_pos_w,
                self._asset.data.root_quat_w,
                asset.data.root_pos_w,
                asset.data.root_quat_w,
            )
            obstacle_pose = self._Pose(
                position=pos_b[:1].contiguous(), quaternion=quat_b[:1].contiguous()
            )
            self._mpc.scene_collision_checker.update_obstacle_pose(
                curobo_name, obstacle_pose, env_idx=0
            )

        # Step the MPC from its own advancing reference state and command the next action; the real
        # arm tracks `_cmd_pos`/`_cmd_vel` via its joint PD.
        result = self._mpc.optimize_next_action(self._ref_js)
        next_action = result.next_action
        if next_action is not None:
            self._ref_js = next_action.clone()
            if self._ref_js.acceleration is None:
                self._ref_js.acceleration = torch.zeros_like(self._ref_js.position)
            self._cmd_pos = next_action.position.clone()
            if next_action.velocity is not None:
                self._cmd_vel = next_action.velocity.clone()

    def apply_actions(self):
        self._asset.set_joint_position_target(self._cmd_pos, joint_ids=self._ordered_joint_ids)

    def reset(self, env_ids=None) -> None:
        # Put the arm joints in POSITION-drive mode for the MPC. The env zeros their stiffness/
        # damping at startup, which leaves set_joint_position_target with no drive to act on.
        # Restore position gains here (after startup gain-randomization).
        n = len(self._ordered_joint_ids)
        self._asset.write_joint_stiffness_to_sim(
            torch.full((self.num_envs, n), self.cfg.arm_stiffness, device=self.device),
            joint_ids=self._ordered_joint_ids,
        )
        self._asset.write_joint_damping_to_sim(
            torch.full((self.num_envs, n), self.cfg.arm_damping, device=self.device),
            joint_ids=self._ordered_joint_ids,
        )
        # Re-sync the MPC reference to the real arm state and drop the warm-start seed.
        self._ref_js = self._arm_joint_state()
        self._cmd_pos = self._ref_js.position.clone()
        self._cmd_vel = None
        self._mpc.reset_seed()


@configclass
class CommandHandBaseCuroboMpcActionCfg(ActionTermCfg):
    """Configuration for command-driven hand-base cuRobo reactive-MPC arm control."""

    class_type: type[ActionTerm] = CommandHandBaseCuroboMpcAction

    joint_names: list[str] = MISSING
    """Arm joint names/regex controlled by the MPC (the 7 panda joints)."""

    body_name: str = "base"
    """Body whose pose is regulated to the command anchor (the LEAP hand base)."""

    command_name: str = "object_pose"
    """Command term containing the target hand-base (anchor) pose in the robot root frame."""

    command_start: int = 7
    """Start index of the target hand-base pose in the command tensor."""

    robot_config_file: str = MISSING
    """Absolute path to the cuRobo Franka+LEAP robot config yml (franka_leap.yml)."""

    obstacle_cuboids: dict = MISSING
    """Obstacle cuboids in the robot base frame, e.g.
    ``{"table": {"dims": [0.8, 1.5, 0.04], "pose": [0.55, 0.0, 0.235, 1, 0, 0, 0]}}``.
    The receptacle is intentionally excluded so the peg can still reach the bore."""

    dynamic_obstacle_assets: dict = field(default_factory=dict)
    """Map of cuRobo-cuboid-name -> scene-asset-name. Each control step the action reads the named
    scene asset's world pose, converts it to the robot base frame, and rewrites that cuRobo cuboid's
    pose so the MPC avoids the moving prop (in-place update -> CUDA-graph safe). Every cuRobo name
    here MUST also appear in ``obstacle_cuboids`` so a collision-world slot exists. Empty (default)
    keeps a fully static world (current behavior). See :func:`move_dynamic_obstacle` for driving the
    prop's pose. Note: with ``multi_env=False`` the obstacle is synced from env 0 only."""

    arm_stiffness: float = 800.0
    """Position-drive stiffness written to the arm joints so set_joint_position_target has authority
    (the env zeros these actuator gains at startup)."""

    arm_damping: float = 40.0
    """Position-drive damping written to the arm joints."""

    optimization_dt: float = 0.05
    """MPC planning timestep (s). Keep larger than the env control dt (~0.0166) so each control
    step commands meaningful motion; ~0.05-0.1 tracks moving goals well. The default 0.01 / using
    env.step_dt makes ``optimize_next_action`` crawl and never reach the goal."""

    interpolation_steps: int = 4
    """MPC trajectory interpolation steps between optimization knots."""

    collision_activation_distance: float = 0.08
    """World-collision activation distance (m): how far from an obstacle the avoidance cost turns on.
    Larger -> the arm starts deviating earlier/from farther away (bigger detours); too small (e.g.
    the cuRobo default 0.01) lets the optimizer tunnel through thin obstacles. ~0.03-0.08 is sane."""

    collision_weight: float | None = None
    """World-collision cost weight in the MPC optimizer (``scene_collision_cfg.weight``). ``None``
    keeps the optimizer-config default (10000). Lower it (e.g. 3000) so the goal cost competes more
    with avoidance and the arm deviates less near obstacles (looser clearance); raise it for harder
    avoidance. Patched into a copy of ``mpc/lbfgs_mpc.yml`` at build (shared yaml untouched)."""

    use_cuda_graph: bool = True
    """Capture the MPC step in a CUDA graph (fixed batch size = num_envs)."""
