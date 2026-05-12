"""Roll out a direct BC low-level hand policy."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

from src.policy.hi_policy_direct import DEFAULT_DIRECT_BC_CHECKPOINT

parser = argparse.ArgumentParser(description="Play a direct BC low-level hand policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Pick_Insert_Debug_HRL-v0")
parser.add_argument("--low_level_checkpoint", type=str, default=str(DEFAULT_DIRECT_BC_CHECKPOINT))
parser.add_argument("--low_level_obs_group", type=str, default="low_level")
parser.add_argument("--video_length", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import src.tasks  # noqa: F401, E402
from src.policy.hi_policy_direct import load_direct_bc_policy  # noqa: E402


def _low_level_state(obs: dict[str, torch.Tensor], group: str, num_envs: int, device: torch.device) -> torch.Tensor:
    if group not in obs:
        raise KeyError(f"Observation group {group!r} is missing from environment observations.")
    return obs[group].to(device=device).reshape(num_envs, -1)


def _action_term_dim(env, term_name: str) -> int | None:
    try:
        return int(env.unwrapped.action_manager.get_term(term_name).action_dim)
    except (AttributeError, KeyError, ValueError):
        return None


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device
    num_envs = int(env.unwrapped.num_envs)
    total_action_dim = int(env.unwrapped.single_action_space.shape[-1])
    hand_action_dim = _action_term_dim(env, "hand_action") or total_action_dim
    arm_action_dim = _action_term_dim(env, "arm_action") or 0
    low_level_policy = load_direct_bc_policy(
        args_cli.low_level_checkpoint,
        device=device,
        expected_action_dim=hand_action_dim,
    )
    if arm_action_dim + low_level_policy.action_dim != total_action_dim:
        raise ValueError(
            "BC policy and environment action dimensions are incompatible: "
            f"arm={arm_action_dim}, hand={low_level_policy.action_dim}, env={total_action_dim}."
        )

    obs, _ = env.reset()
    arm_action = torch.zeros(num_envs, arm_action_dim, device=device) if arm_action_dim > 0 else None
    step_count = 0
    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            low_level_state = _low_level_state(obs, args_cli.low_level_obs_group, num_envs, device)
            hand_action = low_level_policy.act(low_level_state)
            env_action = (
                hand_action
                if arm_action is None
                else torch.cat((arm_action.to(dtype=hand_action.dtype), hand_action), dim=-1)
            )
            obs, _, _, _, _ = env.step(env_action)
            step_count += 1

        if args_cli.video_length is not None and step_count >= args_cli.video_length:
            break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
