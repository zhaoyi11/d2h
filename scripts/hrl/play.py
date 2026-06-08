"""Play a trained high-level HRL policy."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

_RSL_RL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.insert(0, str(_RSL_RL_SCRIPT_DIR))
import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description="Play a trained high-level HRL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Pick_Insert_HRL-v0", help="Name of the HRL task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    default="/home/yizhao/yi/D2H/logs/rsl_rl/anyreorient/model_14999.pt",
    help="Path to the frozen low-level RSL-RL checkpoint.",
)
parser.add_argument(
    "--low_level_obs_group",
    type=str,
    default="low_level",
    help="Observation group used as the low-level RSL-RL policy observation.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint is provided, use the last saved high-level model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, export_policy_as_jit, export_policy_as_onnx  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import src.tasks  # noqa: F401, E402
from src.policy.hl_policy import DirectLowLevelEnvWrapper, load_low_level_rsl_rl_policy  # noqa: E402
from src.policy.hl_policy.rsl_rl_wrapper import HrlRslRlVecEnvWrapper  # noqa: E402


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint is not None:
        return retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_last_checkpoint:
        run_dir = agent_cfg.load_run if agent_cfg.load_run else ".*"
        return get_checkpoint_path(log_root_path, run_dir, "model_.*.pt")
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _policy_module(runner: OnPolicyRunner) -> torch.nn.Module:
    try:
        return runner.alg.policy
    except AttributeError:
        return runner.alg.actor_critic


def _policy_normalizer(policy_nn: torch.nn.Module):
    if hasattr(policy_nn, "actor_obs_normalizer"):
        return policy_nn.actor_obs_normalizer
    if hasattr(policy_nn, "student_obs_normalizer"):
        return policy_nn.student_obs_normalizer
    return None


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with a high-level HRL policy."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(f"Unsupported runner class for HRL play: {agent_cfg.class_name}")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = _resolve_checkpoint(agent_cfg)
    print(f"[INFO]: Loading high-level model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    hand_action_dim = int(env.unwrapped.action_manager.get_term("hand_action").action_dim)
    wrist_action_dim = int(env.unwrapped.action_manager.total_action_dim) - hand_action_dim
    low_level_policy = load_low_level_rsl_rl_policy(
        args_cli.low_level_checkpoint,
        device=env.unwrapped.device,
        expected_action_dim=hand_action_dim,
    )
    env = DirectLowLevelEnvWrapper(
        env,
        low_level_policy=low_level_policy,
        low_level_obs_group=args_cli.low_level_obs_group,
        wrist_action_dim=wrist_action_dim,
    )
    env = HrlRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = _policy_module(runner)
    normalizer = _policy_normalizer(policy_nn)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="hrl_policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="hrl_policy.onnx")

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if hasattr(policy_nn, "reset"):
                policy_nn.reset(dones)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
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
