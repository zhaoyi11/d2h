"""Smoke-test the pick-screw HRL env: cuRobo-MPC arm + frozen low-level LEAP-hand policy.

Mirrors ``smoke_pick_insert_hrl_ik.py`` for the screw task. The 7-DOF Franka arm is driven by
cuRobo reactive MPC tracking the command hand-base anchor (0 external action dims); the 16-DOF
LEAP hand is driven by a frozen reorient RSL-RL policy reading the 167-dim ``low_level``
observation group (object + external fingertip-force sensing). There is NO high-level policy:
the only actions sent to the env are the low-level policy's 16 hand-joint targets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Import Warp BEFORE the Isaac app so the site-packages Warp (1.14, required by cuRobo 0.8) is
# cached in sys.modules and Isaac's bundled omni.warp.core (1.8.2) does not shadow it.
import warp  # noqa: F401

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Smoke-test pick-screw HRL cuRobo-MPC + low-level hand policy.")
parser.add_argument("--task", type=str, default="Pick_Screw_HRL-v0", help="Registered Gym task to launch.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run.")
parser.add_argument("--steps", type=int, default=1200, help="Number of environment steps to simulate.")
parser.add_argument("--print_every", type=int, default=30, help="Print diagnostics every N steps.")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default="/home/yizhao/yi/dex_reorient/logs/rsl_rl/reorient/2026-06-25_15-18-57/model_14999.pt",
    help="Frozen RSL-RL reorient actor checkpoint for hand control (167-dim obs, 16-dim action).",
)
parser.add_argument(
    "--low_level_obs_group",
    type=str,
    default="low_level",
    help="Observation group fed to the low-level RSL-RL policy.",
)
parser.add_argument(
    "--zero_hand_action",
    action="store_true",
    help="Use zero hand actions instead of the low-level actor (validate the MPC arm alone).",
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
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import src.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from src.policy.hl_policy import load_low_level_rsl_rl_policy  # noqa: E402


def _low_level_obs(env, group: str) -> torch.Tensor:
    obs = env.observation_manager.compute_group(group)
    return obs.reshape(env.num_envs, -1)


def _validate_managers(env) -> None:
    action_manager = env.action_manager
    command = env.command_manager.get_command("object_pose")
    if command.shape[-1] != 14:
        raise RuntimeError(f"Expected object_pose command shape (*, 14), got {tuple(command.shape)}.")

    arm_dim = int(action_manager.get_term("arm_action").action_dim)
    hand_dim = int(action_manager.get_term("hand_action").action_dim)
    total_dim = int(action_manager.total_action_dim)
    if arm_dim != 0:
        raise RuntimeError(f"Expected arm_action to consume 0 external dims, got {arm_dim}.")
    if total_dim != hand_dim:
        raise RuntimeError(f"Expected total action dim {hand_dim}, got {total_dim}.")

    low_obs_dim = int(_low_level_obs(env, args_cli.low_level_obs_group).shape[-1])
    print(f"[INFO]: active action terms: {action_manager.active_terms}", flush=True)
    print(f"[INFO]: arm_dim={arm_dim} hand_dim={hand_dim} total={total_dim}", flush=True)
    print(f"[INFO]: '{args_cli.low_level_obs_group}' obs dim = {low_obs_dim} (expect 167)", flush=True)
    if low_obs_dim != 167:
        raise RuntimeError(f"Expected low-level obs dim 167, got {low_obs_dim}.")


def main() -> None:
    env = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        env_unwrapped = env.unwrapped

        print(f"[INFO]: observation space: {env.observation_space}", flush=True)
        print(f"[INFO]: action space: {env.action_space}", flush=True)
        env.reset()
        _validate_managers(env_unwrapped)

        # Total env action dim == hand action dim (arm_action consumes 0 dims; MPC reads the goal
        # from the command). So the env step takes the low-level policy's 16-DOF hand action.
        action_dim = int(env_unwrapped.action_manager.total_action_dim)

        low_level_policy = None
        if not args_cli.zero_hand_action:
            actual_obs = int(_low_level_obs(env_unwrapped, args_cli.low_level_obs_group).shape[-1])
            low_level_policy = load_low_level_rsl_rl_policy(
                args_cli.low_level_checkpoint,
                device=env_unwrapped.device,
                expected_obs_dim=actual_obs,
                expected_action_dim=action_dim,
            )
            print(
                f"[INFO]: loaded low-level actor from {args_cli.low_level_checkpoint} "
                f"(obs_dim={low_level_policy.obs_dim}, action_dim={low_level_policy.action_dim}).",
                flush=True,
            )

        for step in range(1, args_cli.steps + 1):
            with torch.inference_mode():
                if low_level_policy is None:
                    actions = torch.zeros((env_unwrapped.num_envs, action_dim), device=env_unwrapped.device)
                else:
                    actions = low_level_policy.act(_low_level_obs(env_unwrapped, args_cli.low_level_obs_group))
                env.step(actions)
            if args_cli.print_every > 0 and (step == 1 or step % args_cli.print_every == 0 or step == args_cli.steps):
                print(f"[STEP {step:04d}]: stepping ok", flush=True)

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
