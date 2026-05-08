"""Train the high-level HRL policy for pick-insert."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

_RSL_RL_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "rsl_rl"
sys.path.insert(0, str(_RSL_RL_SCRIPT_DIR))
import cli_args  # isort: skip  # noqa: E402


parser = argparse.ArgumentParser(description="Train a high-level HRL agent with RSL-RL PPO.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Pick_Insert_HRL-v0", help="Name of the HRL task.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument("--init_at_random_ep_len", type=bool, default=False, help="Randomize initial episode lengths.")
parser.add_argument(
    "--low_level_checkpoint",
    type=str,
    required=True,
    help="Path to the frozen low-level VAE checkpoint.",
)
parser.add_argument(
    "--low_level_obs_group",
    type=str,
    default="low_level",
    help="Observation group used as the VAE low-level state.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import src.tasks  # noqa: F401, E402
from src.policy.hl_policy import HierarchicalChunkEnvWrapper, load_low_level_vae  # noqa: E402
from src.policy.hl_policy.rsl_rl_wrapper import HrlRslRlVecEnvWrapper  # noqa: E402
from src.utils import _patch_rsl_wandb_writer  # noqa: E402


logger = logging.getLogger(__name__)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _scheduled_entropy_coef(initial_entropy_coef, final_entropy_coef, iteration, decay_iterations):
    if decay_iterations <= 0:
        return final_entropy_coef
    progress = min(max(iteration / decay_iterations, 0.0), 1.0)
    return initial_entropy_coef + progress * (final_entropy_coef - initial_entropy_coef)


def _apply_entropy_coef_schedule(runner, agent_cfg):
    final_entropy_coef = getattr(agent_cfg, "entropy_coef_final", None)
    decay_fraction = getattr(agent_cfg, "entropy_coef_decay_fraction", None)
    if final_entropy_coef is None or decay_fraction is None:
        return

    initial_entropy_coef = runner.alg.entropy_coef
    decay_iterations = int(agent_cfg.max_iterations * decay_fraction)
    iteration = runner.current_learning_iteration
    update = runner.alg.update

    def update_with_entropy_schedule():
        nonlocal iteration
        runner.alg.entropy_coef = _scheduled_entropy_coef(
            initial_entropy_coef,
            final_entropy_coef,
            iteration,
            decay_iterations,
        )
        iteration += 1
        return update()

    runner.alg.update = update_with_entropy_schedule


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if getattr(agent_cfg, "logger", None) == "wandb":
        _patch_rsl_wandb_writer()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    hand_action_dim = int(env.unwrapped.action_manager.get_term("hand_action").action_dim)
    # import ipdb; ipdb.set_trace()
    low_level_policy = load_low_level_vae(
        args_cli.low_level_checkpoint,
        device=env.unwrapped.device,
        expected_action_dim=hand_action_dim,
    )
    env = HierarchicalChunkEnvWrapper(
        env,
        low_level_policy=low_level_policy,
        low_level_obs_group=args_cli.low_level_obs_group,
    )
    env = HrlRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    start_time = time.time()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        runner.load(resume_path)
    _apply_entropy_coef_schedule(runner, agent_cfg)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print_dict({"low_level_checkpoint": args_cli.low_level_checkpoint, "low_level_obs_group": args_cli.low_level_obs_group})

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=args_cli.init_at_random_ep_len,
    )
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
