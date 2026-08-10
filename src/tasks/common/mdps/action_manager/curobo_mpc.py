from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


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
        self._last_joint_position_target = self._cmd_pos.clone()
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

    @property
    def ordered_joint_ids(self) -> tuple[int, ...]:
        """Robot joint indices in the order used by cuRobo's position target."""
        return tuple(self._ordered_joint_ids)

    @property
    def ordered_joint_names(self) -> tuple[str, ...]:
        """Robot joint names in the order used by cuRobo's position target."""
        return tuple(self._mpc.joint_names)

    @property
    def last_joint_position_target(self) -> torch.Tensor:
        """Most recent arm target applied to the robot, preserved across environment reset."""
        return self._last_joint_position_target

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
        self._last_joint_position_target[:] = self._cmd_pos

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
