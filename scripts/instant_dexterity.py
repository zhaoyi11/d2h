"""Instant dexterity from coarse demonstrations."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Import Warp BEFORE the Isaac app so the site-packages Warp (1.14, required by cuRobo 0.8) is
# cached in sys.modules and Isaac's bundled omni.warp.core (1.8.2) does not shadow it.
import warp  # noqa: F401

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Instant dexterity from coarse demonstrations.")
parser.add_argument("--task", type=str, default="Pick_Insert_HRL-v0", help="Registered Gym task to launch.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--steps", type=int, default=1200, help="Number of environment steps to simulate.")
parser.add_argument("--print_every", type=int, default=30, help="Print pose diagnostics every N steps.")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default="/home/yizhao/yi/dex_reorient/logs/rsl_rl/reorient/2026-06-25_15-18-57/model_14999.pt",
    help="RSL-RL actor checkpoint for hand control.",
)
parser.add_argument(
    "--gate_low_level",
    action="store_true",
    help="Gate the hand actions: apply them only when the hand-base is within --gate_dist of the "
    "live object's grasp anchor; otherwise hold the hand open (stretch).",
)
parser.add_argument(
    "--gate_dist",
    type=float,
    default=0.08,
    help="Max hand-base<->live-object distance (m) to enable the hand policy; above it => stretch.",
)
parser.add_argument("--record_data", action="store_true", help="Record BC observations, actions, and reset states.")
parser.add_argument("--record_dir", type=str, default=None, help="Output directory for recorded episode NPZ files.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import src.tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    compute_pose_error,
    subtract_frame_transforms,
)
from src.policy.high_level.gate import LowLevelGateCfg, LowLevelHandGate  # noqa: E402
from src.policy.instant_dexterity_recording import (  # noqa: E402
    InstantDexterityEpisodeRecorder,
    build_bc_action,
    build_bc_observation,
    flatten_scene_state,
)
from src.policy.low_level import load_low_level_rsl_rl_policy  # noqa: E402
from src.tasks.common.mdps.contacts import contacts as good_object_contact  # noqa: E402


def _as_list(tensor: torch.Tensor) -> list[float]:
    return [round(float(value), 5) for value in tensor.detach().cpu().tolist()]


_BASE_BODY: tuple[int, str] | None = None


def _hand_base_body(env) -> tuple[int, str]:
    """Resolve (and cache) the single robot body named 'base'."""
    global _BASE_BODY
    if _BASE_BODY is None:
        robot = env.scene["robot"]
        body_ids, body_names = robot.find_bodies("base")
        if len(body_ids) != 1:
            raise RuntimeError(f"Expected one robot body matching 'base', found {len(body_ids)}: {body_names}.")
        _BASE_BODY = (body_ids[0], body_names[0])
    return _BASE_BODY


def _hand_base_pose_w(env) -> tuple[torch.Tensor, torch.Tensor, str]:
    robot = env.scene["robot"]
    body_idx, body_name = _hand_base_body(env)
    return robot.data.body_pos_w[:, body_idx], robot.data.body_quat_w[:, body_idx], body_name


def _hand_base_pose_b(env, hand_pos_w: torch.Tensor, hand_quat_w: torch.Tensor) -> torch.Tensor:
    robot = env.scene["robot"]
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        hand_pos_w,
        hand_quat_w,
    )
    return torch.cat((pos_b, quat_b), dim=1)


def _object_pose_in_hand_base_b(env, hand_pos_w: torch.Tensor, hand_quat_w: torch.Tensor) -> torch.Tensor:
    object_asset = env.scene["object"]
    object_pos_b, object_quat_b = subtract_frame_transforms(
        hand_pos_w,
        hand_quat_w,
        object_asset.data.root_pos_w,
        object_asset.data.root_quat_w,
    )
    return torch.cat((object_pos_b, object_quat_b), dim=1)


def _low_level_obs(env, observations=None) -> torch.Tensor:
    obs = (
        env.observation_manager.compute_group("low_level")
        if observations is None
        else observations["low_level"]
    )
    return obs.reshape(env.num_envs, -1)


def _record_output_dir() -> Path:
    if args_cli.record_dir is not None:
        return Path(args_cli.record_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_slug = args_cli.task.replace("/", "_")
    return Path.cwd() / "datasets" / task_slug / f"instant_dexterity_{timestamp}"


def _recording_metadata(env, low_level_obs, arm_action, hand_action) -> dict:
    scene_fields = sorted(flatten_scene_state(env.scene.get_state(is_relative=True)))
    articulation_joint_names = {
        name: list(asset.joint_names) for name, asset in env.scene.articulations.items()
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "task": args_cli.task,
        "seed": args_cli.seed,
        "num_envs": env.num_envs,
        "step_dt": float(env.step_dt),
        "low_level_checkpoint": str(Path(args_cli.low_level_checkpoint).expanduser()),
        "observation": {
            "low_level_dim": int(low_level_obs.shape[1]),
            "bc_dim": int(low_level_obs.shape[1] + 21),
            "bc_order": ["low_level", "arm_joint_pos", "arm_joint_vel", "hand_base_command"],
        },
        "action": {
            "dim": 23,
            "order": ["arm_delta", "hand_joint_target"],
            "arm_delta": "target_minus_pre_step_position_radians",
            "hand_joint_target": "ema_processed_absolute_position_radians",
            "hand_policy_raw": "normalized_frozen_policy_output_before_gate",
            "hand_policy_executed": "normalized_action_after_gate_sent_to_teacher_env",
        },
        "joint_order": {
            "arm": list(arm_action.ordered_joint_names),
            "hand": list(hand_action._joint_names),
            "articulations": articulation_joint_names,
        },
        "scene_state": {
            "relative_to_env_origin": True,
            "fields": scene_fields,
        },
    }


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
    hand_pos_w, hand_quat_w, body_name = _hand_base_pose_w(env)
    current_hand_base_b = _hand_base_pose_b(env, hand_pos_w, hand_quat_w)
    current_object_in_hand_b = _object_pose_in_hand_base_b(env, hand_pos_w, hand_quat_w)
    print(f"[STEP {step:04d}]: target object desired hand-local {_as_list(command[0, :7])}", flush=True)
    print(f"[STEP {step:04d}]: target hand base pose root {_as_list(command[0, 7:14])}", flush=True)
    print(f"[STEP {step:04d}]: current {body_name} pose b {_as_list(current_hand_base_b[0])}", flush=True)
    print(f"[STEP {step:04d}]: object pose in {body_name} b {_as_list(current_object_in_hand_b[0])}", flush=True)
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
    command_term = env.command_manager.get_term("object_pose")
    gate_err = getattr(command_term, "metrics", {}).get("hand_base_object_error")
    if gate_err is not None:
        print(
            f"[STEP {step:04d}]: hand<->object gate error   {float(gate_err[0]):.5f} m",
            flush=True,
        )
    if "pouring_progress" in getattr(command_term, "metrics", {}):
        metrics = command_term.metrics
        print(
            f"[STEP {step:04d}]: pouring                   "
            f"frame={int(metrics['pouring_frame'][0])} "
            f"progress={float(metrics['pouring_progress'][0]):.3f} "
            f"gravity={bool(metrics['gravity_enabled'][0])} "
            f"contact={float(metrics['bottle_hand_contact_force'][0]):.3f}N "
            f"success={bool(metrics['pouring_success'][0])}",
            flush=True,
        )
    if "pick_and_place_progress" in getattr(command_term, "metrics", {}):
        metrics = command_term.metrics
        print(
            f"[STEP {step:04d}]: pick-and-place            "
            f"frame={int(metrics['pick_and_place_frame'][0])} "
            f"progress={float(metrics['pick_and_place_progress'][0]):.3f}",
            flush=True,
        )
    if "inside_box" in getattr(command_term, "metrics", {}):
        metrics = command_term.metrics
        stepper = command_term._stepper
        stage = stepper.step_to_stage[stepper.step]
        contact = good_object_contact(env, command_term.cfg.grasp_contact_force_threshold)
        print(
            f"[STEP {step:04d}]: clean-table grasp         "
            f"stage={int(stage[0])} "
            f"contact={bool(contact[0])} "
            f"streak={int(command_term._grasp_contact_streak[0])} "
            f"phase={int(command_term._grasp_phase_steps[0])} "
            f"keep_open={bool(metrics['keep_hand_open'][0])} "
            f"height={float(metrics['object_height_above_table'][0]):.5f} m "
            f"lifted={bool(command_term.lifted[0])}",
            flush=True,
        )
        print(
            f"[STEP {step:04d}]: clean-table placement     "
            f"inside={bool(metrics['inside_box'][0])} "
            f"released={bool(metrics['released'][0])} "
            f"hand_clear={bool(metrics['hand_clear'][0])} "
            f"success={bool(metrics['success'][0])}",
            flush=True,
        )
    stepper = getattr(command_term, "_stepper", None)
    clearance_streak = getattr(command_term, "_clearance_streak", None)
    if stepper is not None and clearance_streak is not None:
        stage = stepper.step_to_stage[stepper.step]
        metrics = command_term.metrics
        clearance = metrics.get("bolt_clearance")
        transitioned = metrics.get("clearance_transitioned")
        grasped = good_object_contact(env, command_term.cfg.grasp_contact_force_threshold)
        print(
            f"[STEP {step:04d}]: unscrew transition       stage={int(stage[0])} "
            f"clearance={float(clearance[0]):.5f} m grasped={bool(grasped[0])} "
            f"streak={int(clearance_streak[0])} transitioned={bool(transitioned[0])}",
            flush=True,
        )


def _print_scene_resets(
    env,
    step: int,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> None:
    reset_env_ids = (terminated | truncated).nonzero().flatten()
    if reset_env_ids.numel() == 0:
        return
    termination_manager = env.termination_manager
    for env_id in reset_env_ids.tolist():
        active_terms = [
            term_name
            for term_name in termination_manager.active_terms
            if bool(termination_manager.get_term(term_name)[env_id])
        ]
        print(
            f"[STEP {step:04d}]: scene reset env={env_id} terms={active_terms}",
            flush=True,
        )


def main() -> None:
    env = None
    recorder = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
        )
        env_cfg.commands.object_pose.debug_vis = True
        if args_cli.seed is not None:
            env_cfg.seed = args_cli.seed

        env = gym.make(args_cli.task, cfg=env_cfg)
        env_unwrapped = env.unwrapped

        print(f"[INFO]: Gym observation space: {env.observation_space}", flush=True)
        print(f"[INFO]: Gym action space: {env.action_space}", flush=True)
        observations, _ = env.reset()
        _validate_managers(env_unwrapped)
        _print_snapshot(env_unwrapped, 0)

        action_dim = int(env_unwrapped.action_manager.total_action_dim)
        low_level_obs = _low_level_obs(env_unwrapped, observations)
        actual_obs = int(low_level_obs.shape[-1])
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

        # low-level gate: apply the hand policy only when the hand-base is within --gate_dist of the
        # live object's grasp anchor; otherwise hold the hand open (stretch).
        gate = LowLevelHandGate(LowLevelGateCfg(hand_at_object_dist=args_cli.gate_dist))

        action_manager = env_unwrapped.action_manager
        arm_action = action_manager.get_term("arm_action")
        hand_action = action_manager.get_term("hand_action")
        if args_cli.record_data:
            output_dir = _record_output_dir()
            recorder = InstantDexterityEpisodeRecorder(
                output_dir=output_dir,
                num_envs=env_unwrapped.num_envs,
                metadata=_recording_metadata(env_unwrapped, low_level_obs, arm_action, hand_action),
            )
            print(f"[INFO]: Recording BC episodes to {output_dir}", flush=True)

        for step in range(1, args_cli.steps + 1):
            with torch.inference_mode():
                low_level_obs = _low_level_obs(env_unwrapped, observations)
                hand_policy_action = low_level_policy.act(low_level_obs)
                actions = hand_policy_action
                if gate is not None:
                    actions = gate.apply(actions, env_unwrapped)

                if recorder is not None:
                    robot = env_unwrapped.scene["robot"]
                    arm_joint_ids = arm_action.ordered_joint_ids
                    arm_joint_pos = robot.data.joint_pos[:, arm_joint_ids].clone()
                    arm_joint_vel = robot.data.joint_vel[:, arm_joint_ids].clone()
                    hand_base_command = env_unwrapped.command_manager.get_command("object_pose")[:, 7:14].clone()
                    bc_observation = build_bc_observation(
                        low_level_obs,
                        arm_joint_pos,
                        arm_joint_vel,
                        hand_base_command,
                    )
                    step_data = {
                        "observation.low_level": low_level_obs,
                        "observation.arm_joint_pos": arm_joint_pos,
                        "observation.arm_joint_vel": arm_joint_vel,
                        "observation.hand_base_command": hand_base_command,
                        "observation.bc": bc_observation,
                        "action.hand_policy_raw": hand_policy_action,
                        "action.hand_policy_executed": actions,
                        **flatten_scene_state(env_unwrapped.scene.get_state(is_relative=True)),
                    }

                observations, reward, terminated, truncated, _ = env.step(actions)

                if recorder is not None:
                    arm_joint_target = arm_action.last_joint_position_target.clone()
                    hand_joint_target = hand_action.processed_actions.clone()
                    bc_action, arm_delta = build_bc_action(
                        arm_joint_target,
                        arm_joint_pos,
                        hand_joint_target,
                    )
                    step_data.update(
                        {
                            "action": bc_action,
                            "action.arm_delta": arm_delta,
                            "action.arm_joint_target": arm_joint_target,
                            "action.hand_joint_target": hand_joint_target,
                        }
                    )
                    recorder.add_step(
                        step_data,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                    )
            _print_scene_resets(env_unwrapped, step, terminated, truncated)
            if args_cli.print_every > 0 and (step == 1 or step % args_cli.print_every == 0 or step == args_cli.steps):
                _print_snapshot(env_unwrapped, step)
                if gate is not None and gate.last_mask is not None:
                    print(
                        f"[STEP {step:04d}]: low-level active frac = {float(gate.last_mask.float().mean()):.3f}",
                        flush=True,
                    )

        print("[INFO]: Smoke test completed.", flush=True)
    finally:
        if recorder is not None:
            recorder.close()
            print(f"[INFO]: Saved {recorder.num_saved} recorded episode files.", flush=True)
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
