"""Smoke-test command-driven hand-base IK for the pick-insert HRL env."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_TASK_DIR = REPO_ROOT / "src" / "tasks" / "pick_insert demo"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Smoke-test pick-insert HRL hand-base IK tracking.")
parser.add_argument("--task", type=str, default="Pick_Insert_HRL-v0", help="Registered Gym task to launch.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run.")
parser.add_argument("--steps", type=int, default=1200, help="Number of environment steps to simulate.")
parser.add_argument("--print_every", type=int, default=30, help="Print pose diagnostics every N steps.")
parser.add_argument("--demo_cfg", action="store_true", help="Use src/tasks/pick_insert demo/env_cfg.py.")
parser.add_argument(
    "--deterministic_reset",
    action="store_true",
    help="Disable reset randomization for debugging the initial hand-object pose.",
)
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default="/home/yizhao/yi/D2H/logs/rsl_rl/anyreorient/exported/policy.pt",
    help="RSL-RL actor checkpoint for hand control.",
)
parser.add_argument(
    "--zero_hand_action",
    action="store_true",
    help="Use zero hand actions instead of the low-level actor.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import isaaclab.sim as sim_utils  # noqa: E402
import src.tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    compute_pose_error,
    subtract_frame_transforms,
)
from src.policy.hl_policy import load_low_level_rsl_rl_policy  # noqa: E402


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _apply_deterministic_reset(env_cfg) -> None:
    reset_joints = getattr(env_cfg.events, "reset_robot_joints", None)
    if reset_joints is not None:
        reset_joints.params["position_range"] = [0.0, 0.0]

    reset_object = getattr(env_cfg.events, "reset_object_relative_to_hand", None)
    if reset_object is None:
        raise RuntimeError("--deterministic_reset requires reset_object_relative_to_hand on the env config.")
    reset_object.params["random_orientation"] = False
    reset_object.params["local_rot"] = (1.0, 0.0, 0.0, 0.0)
    print("[INFO]: Deterministic reset enabled: zero joint offset and identity object local rotation.", flush=True)


def _make_env_cfg():
    if args_cli.demo_cfg:
        demo_env_cfg = _load_module("pick_insert_demo_env_cfg_smoke", DEMO_TASK_DIR / "env_cfg.py")
        env_cfg = demo_env_cfg.DexsuiteFrankaLeapInsertHrlEnvCfg()
        env_cfg.sim.device = args_cli.device
        env_cfg.sim.use_fabric = not args_cli.disable_fabric
        env_cfg.scene.num_envs = args_cli.num_envs
        if args_cli.deterministic_reset:
            _apply_deterministic_reset(env_cfg)
        return env_cfg

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.deterministic_reset:
        _apply_deterministic_reset(env_cfg)
    return env_cfg


def _as_list(tensor: torch.Tensor) -> list[float]:
    return [round(float(value), 5) for value in tensor.detach().cpu().tolist()]


def _hand_base_pose_w(env) -> tuple[torch.Tensor, torch.Tensor, str]:
    robot = env.scene["robot"]
    body_ids, body_names = robot.find_bodies("base")
    if len(body_ids) != 1:
        raise RuntimeError(f"Expected one robot body matching 'base', found {len(body_ids)}: {body_names}.")
    body_idx = body_ids[0]
    return robot.data.body_pos_w[:, body_idx], robot.data.body_quat_w[:, body_idx], body_names[0]


def _hand_base_pose_b(env) -> tuple[torch.Tensor, str]:
    robot = env.scene["robot"]
    hand_pos_w, hand_quat_w, body_name = _hand_base_pose_w(env)
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        hand_pos_w,
        hand_quat_w,
    )
    return torch.cat((pos_b, quat_b), dim=1), body_name


def _object_pose_in_hand_base_b(env) -> torch.Tensor:
    object_asset = env.scene["object"]
    hand_pos_w, hand_quat_w, _ = _hand_base_pose_w(env)
    object_pos_b, object_quat_b = subtract_frame_transforms(
        hand_pos_w,
        hand_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    return torch.cat((object_pos_b, object_quat_b), dim=1)


def _target_object_pose_in_desired_hand_base_b(env) -> torch.Tensor:
    command = env.command_manager.get_command("object_pose")
    return command[:, :7]


def _low_level_obs(env) -> torch.Tensor:
    obs = env.observation_manager.compute_group("low_level")
    return obs.reshape(env.num_envs, -1)


def _low_level_actions(env, policy, action_dim: int) -> torch.Tensor:
    if policy is None:
        return torch.zeros((env.num_envs, action_dim), device=env.device)
    return policy.act(_low_level_obs(env))


def _make_frame_marker(prim_path: str) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.08, 0.08, 0.08),
            )
        },
    )
    marker = VisualizationMarkers(cfg)
    marker.set_visibility(True)
    return marker


def _make_target_object_marker(env) -> VisualizationMarkers:
    object_spawn = env.scene["object"].cfg.spawn
    if hasattr(object_spawn, "usd_path"):
        target_object_cfg = sim_utils.UsdFileCfg(
            usd_path=object_spawn.usd_path,
            scale=object_spawn.scale,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.55),
        )
    else:
        target_object_cfg = sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.08),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.55),
        )

    marker = VisualizationMarkers(
        VisualizationMarkersCfg(
            prim_path="/Visuals/PickInsertCommand/TargetObject",
            markers={"target_object": target_object_cfg},
        )
    )
    marker.set_visibility(True)
    return marker


def _command_poses_w(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    command = env.command_manager.get_command("object_pose")
    command_term = env.command_manager.get_term("object_pose")
    robot = env.scene["robot"]

    base_pos_w, base_quat_w = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        command[:, 7:10],
        command[:, 10:14],
    )
    object_pos_w, object_quat_w = combine_frame_transforms(
        base_pos_w,
        base_quat_w,
        command[:, :3],
        command[:, 3:7],
    )
    # Reconstruct the (correction-included) anchor from the hand-base target and the fixed
    # hand-base->anchor transform: anchor = hand_base_command ⊕ hand_base_to_anchor_pose.
    hand_base_to_anchor = torch.tensor(command_term.cfg.hand_base_to_anchor_pose, device=command.device).repeat(
        command.shape[0],
        1,
    )
    anchor_pos_b, anchor_quat_b = combine_frame_transforms(
        command[:, 7:10],
        command[:, 10:14],
        hand_base_to_anchor[:, :3],
        hand_base_to_anchor[:, 3:7],
    )
    anchor_pos_w, anchor_quat_w = combine_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        anchor_pos_b,
        anchor_quat_b,
    )
    return object_pos_w, object_quat_w, anchor_pos_w, anchor_quat_w, base_pos_w, base_quat_w


def _update_command_markers(
    env,
    target_object_marker: VisualizationMarkers,
    target_anchor_marker: VisualizationMarkers,
    target_base_marker: VisualizationMarkers,
) -> None:
    object_pos_w, object_quat_w, anchor_pos_w, anchor_quat_w, base_pos_w, base_quat_w = _command_poses_w(env)
    target_object_marker.visualize(object_pos_w, object_quat_w)
    target_anchor_marker.visualize(anchor_pos_w, anchor_quat_w)
    target_base_marker.visualize(base_pos_w, base_quat_w)


def _validate_managers(env) -> None:
    action_manager = env.action_manager
    command = env.command_manager.get_command("object_pose")

    if command.shape[-1] != 14:
        raise RuntimeError(f"Expected object_pose command shape (*, 14), got {tuple(command.shape)}.")

    arm_action = action_manager.get_term("arm_action")
    hand_action = action_manager.get_term("hand_action")
    arm_dim = int(arm_action.action_dim)
    hand_dim = int(hand_action.action_dim)
    total_dim = int(action_manager.total_action_dim)

    if arm_dim != 0:
        raise RuntimeError(f"Expected arm_action to consume 0 external dims, got {arm_dim}.")
    if total_dim != hand_dim:
        raise RuntimeError(f"Expected total action dim {hand_dim}, got {total_dim}.")

    print(f"[INFO]: Active action terms: {action_manager.active_terms}", flush=True)
    print(f"[INFO]: arm_action dim={arm_dim}, hand_action dim={hand_dim}, total_action_dim={total_dim}", flush=True)
    print(f"[INFO]: object_pose command shape={tuple(command.shape)}", flush=True)


def _object_goal_error(env) -> tuple[float, float] | None:
    """Object->goal error in robot root frame: (position norm [m], orientation norm [rad])."""
    command_term = env.command_manager.get_term("object_pose")
    goal_b = getattr(command_term, "pose_command_b", None)
    if goal_b is None:
        return None
    robot = env.scene["robot"]
    object_asset = env.scene["object"]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    pos_error, rot_error = compute_pose_error(
        object_pos_b, object_quat_b, goal_b[:, :3], goal_b[:, 3:7], rot_error_type="axis_angle"
    )
    return float(pos_error[0].norm()), float(rot_error[0].norm())


def _correction_norms(env) -> tuple[float, float] | None:
    """Norms of the applied object->anchor correction: (position [m], rotation [rad])."""
    command_term = env.command_manager.get_term("object_pose")
    correction = getattr(command_term, "objanchor_correction", None)
    if correction is None:
        return None
    return float(correction[0, :3].norm()), float(correction[0, 3:].norm())


def _print_snapshot(env, step: int) -> None:
    command = env.command_manager.get_command("object_pose")
    current_hand_base_b, body_name = _hand_base_pose_b(env)
    current_object_in_hand_b = _object_pose_in_hand_base_b(env)
    target_object_in_desired_hand_b = _target_object_pose_in_desired_hand_base_b(env)
    print(f"[STEP {step:04d}]: target object desired hand-local {_as_list(command[0, :7])}", flush=True)
    print(f"[STEP {step:04d}]: target hand base pose root {_as_list(command[0, 7:14])}", flush=True)
    print(f"[STEP {step:04d}]: current {body_name} pose b {_as_list(current_hand_base_b[0])}", flush=True)
    print(f"[STEP {step:04d}]: object pose in {body_name} b {_as_list(current_object_in_hand_b[0])}", flush=True)
    print(f"[STEP {step:04d}]: low-level object goal local {_as_list(target_object_in_desired_hand_b[0])}", flush=True)
    goal_error = _object_goal_error(env)
    if goal_error is not None:
        print(
            f"[STEP {step:04d}]: object->goal error root  pos={goal_error[0]:.5f} m  rot={goal_error[1]:.5f} rad",
            flush=True,
        )
    correction = _correction_norms(env)
    if correction is not None:
        print(
            f"[STEP {step:04d}]: objgoal correction norm   pos={correction[0]:.5f} m  rot={correction[1]:.5f} rad",
            flush=True,
        )
    _print_arm_impedance(env, step)


def _print_arm_impedance(env, step: int) -> None:
    """Log the OSC per-axis stiffness and the wrist wrench driving its softening."""
    try:
        arm_action = env.action_manager.get_term("arm_action")
    except (KeyError, ValueError):
        return
    stiffness = getattr(arm_action, "stiffness", None)
    if stiffness is None:
        return
    print(f"[STEP {step:04d}]: arm stiffness [kx,ky,kz,krx,kry,krz] {_as_list(stiffness[0])}", flush=True)
    ext = getattr(arm_action, "external_wrench", None)
    raw = getattr(arm_action, "raw_wrench", None)
    if ext is not None:
        ext_force, ext_torque = ext[0, :3], ext[0, 3:]
        raw_force_norm = float(raw[0, :3].norm()) if raw is not None else float("nan")
        raw_torque_norm = float(raw[0, 3:].norm()) if raw is not None else float("nan")
        print(
            f"[STEP {step:04d}]: ext force {_as_list(ext_force)} |F|={float(ext_force.norm()):.3f}N (raw|F|={raw_force_norm:.3f}N)",
            flush=True,
        )
        print(
            f"[STEP {step:04d}]: ext torque {_as_list(ext_torque)} |T|={float(ext_torque.norm()):.3f}Nm (raw|T|={raw_torque_norm:.3f}Nm)",
            flush=True,
        )


def main() -> None:
    env = None
    try:
        env_cfg = _make_env_cfg()
        env = gym.make(args_cli.task, cfg=env_cfg)
        env_unwrapped = env.unwrapped

        print(f"[INFO]: Gym observation space: {env.observation_space}", flush=True)
        print(f"[INFO]: Gym action space: {env.action_space}", flush=True)
        env.reset()
        _validate_managers(env_unwrapped)
        _print_snapshot(env_unwrapped, 0)

        target_object_marker = _make_target_object_marker(env_unwrapped)
        target_anchor_marker = _make_frame_marker("/Visuals/PickInsertCommand/TargetAnchorFrame")
        target_base_marker = _make_frame_marker("/Visuals/PickInsertCommand/TargetBaseFrame")
        _update_command_markers(env_unwrapped, target_object_marker, target_anchor_marker, target_base_marker)

        action_dim = int(env_unwrapped.action_manager.total_action_dim)
        low_level_policy = None
        if not args_cli.zero_hand_action:
            actual_obs = int(_low_level_obs(env_unwrapped).shape[-1])
            low_level_policy = load_low_level_rsl_rl_policy(
                args_cli.low_level_checkpoint,
                device=env_unwrapped.device,
                expected_obs_dim=actual_obs,
                expected_action_dim=action_dim,
            )
            expected_obs = int(low_level_policy.obs_dim)
            if actual_obs != expected_obs:
                raise RuntimeError(f"Expected low-level obs dim {expected_obs}, got {actual_obs}.")
            print(
                f"[INFO]: Loaded low-level RSL-RL actor from {args_cli.low_level_checkpoint} "
                f"(obs_dim={expected_obs}, action_dim={low_level_policy.action_dim}).",
                flush=True,
            )

        for step in range(1, args_cli.steps + 1):
            with torch.inference_mode():
                actions = _low_level_actions(env_unwrapped, low_level_policy, action_dim)
                env.step(actions)
            _update_command_markers(env_unwrapped, target_object_marker, target_anchor_marker, target_base_marker)
            if args_cli.print_every > 0 and (step == 1 or step % args_cli.print_every == 0 or step == args_cli.steps):
                _print_snapshot(env_unwrapped, step)

        print("[INFO]: Smoke test completed.", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
