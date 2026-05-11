"""Roll out a goal-conditioned low-level hand policy with the Franka arm held fixed."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play a goal-conditioned low-level hand policy.")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Pick_Insert_Debug_HRL-v0")
parser.add_argument("--low_level_checkpoint", type=str, required=True)
parser.add_argument("--low_level_obs_group", type=str, default="low_level")
parser.add_argument("--goal_dim", type=int, default=7)
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
from src.policy.hl_policy import load_low_level_goal_conditioned  # noqa: E402


def _low_level_state(obs: dict[str, torch.Tensor], group: str, num_envs: int, device: torch.device) -> torch.Tensor:
    if group not in obs:
        raise KeyError(f"Observation group {group!r} is missing from environment observations.")
    return obs[group].to(device=device).reshape(num_envs, -1)


def _split_state_goal(state: torch.Tensor, goal_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if state.shape[-1] <= goal_dim:
        raise ValueError(f"State dim {state.shape[-1]} must be greater than goal_dim {goal_dim}.")
    return state[:, :-goal_dim], state[:, -goal_dim:]


def _context(state_history: torch.Tensor, action_history: torch.Tensor) -> torch.Tensor:
    return torch.cat((state_history, action_history), dim=-1).reshape(state_history.shape[0], -1)


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
    hand_action_dim = int(env.unwrapped.action_manager.get_term("hand_action").action_dim)
    arm_action_dim = int(env.unwrapped.action_manager.get_term("arm_action").action_dim)
    low_level_policy = load_low_level_goal_conditioned(
        args_cli.low_level_checkpoint,
        device=device,
        expected_action_dim=hand_action_dim,
    )

    obs, _ = env.reset()
    raw_state = _low_level_state(obs, args_cli.low_level_obs_group, num_envs, device)
    state, goal = _split_state_goal(raw_state, args_cli.goal_dim)
    if state.shape[-1] != low_level_policy.state_dim:
        raise ValueError(f"Expected policy state dim {low_level_policy.state_dim}, got {state.shape[-1]}.")
    if goal.shape[-1] != low_level_policy.goal_dim:
        raise ValueError(f"Expected policy goal dim {low_level_policy.goal_dim}, got {goal.shape[-1]}.")

    state_history = state.unsqueeze(1).repeat(1, low_level_policy.config.past_length, 1)
    action_history = torch.zeros(
        num_envs,
        low_level_policy.config.past_length,
        low_level_policy.action_dim,
        device=device,
        dtype=state.dtype,
    )
    arm_action = torch.zeros(num_envs, arm_action_dim, device=device, dtype=state.dtype)
    chunk = low_level_policy.predict(_context(state_history, action_history), goal)
    chunk_idx = 0
    step_count = 0
    dt = env.unwrapped.step_dt

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            if chunk_idx >= low_level_policy.chunk_length:
                chunk = low_level_policy.predict(_context(state_history, action_history), goal)
                chunk_idx = 0
            hand_action = chunk[:, chunk_idx, :]
            env_action = torch.cat((arm_action, hand_action), dim=-1)
            obs, _, terminated, truncated, _ = env.step(env_action)
            raw_state = _low_level_state(obs, args_cli.low_level_obs_group, num_envs, device)
            state, goal = _split_state_goal(raw_state, args_cli.goal_dim)
            state_history = torch.cat((state_history[:, 1:, :], state.unsqueeze(1)), dim=1)
            action_history = torch.cat((action_history[:, 1:, :], hand_action.unsqueeze(1)), dim=1)
            chunk_idx += 1
            step_count += 1

            done = terminated.bool() | truncated.bool()
            if bool(done.any()):
                state_history[done] = state[done].unsqueeze(1).repeat(1, low_level_policy.config.past_length, 1)
                action_history[done] = 0.0

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
