"""Train a D2H task with FlashSAC (off-policy SAC, replaces RSL-RL PPO).

Mirrors the layout of ``scripts/rsl_rl/train.py``: AppLauncher is launched from
CLI args before any Isaac import, then we build the env via ``gym.make`` using a
Hydra-resolved ``env_cfg``, wrap it in ``FlashSACIsaacLabAdapter``, and drive
FlashSAC's training loop (replay buffer fill → SAC update → checkpoint). The
loop body is adapted from ``FlashSAC_dex/train.py``; we keep the same semantics
but source hyperparameters from each task's ``flash_sac_cfg_entry_point``.

Note on "evaluation frequency": FlashSAC's upstream loop calls ``evaluate()``
on a separate ``eval_env`` every ``runner.evaluation_per_interaction_step``.
Isaac Lab enforces a single ``SimulationApp`` per process, so we cannot stand
up an independent eval env mid-training. Instead, the adapter populates
``infos["episode_info"]`` with rolling train-time episodic return + length
(plus per-term ``Episode_Reward/*`` from IsaacLab's reward manager), and the
trainer flushes those to W&B every ``runner.logging_per_interaction_step``
steps. Use ``--logging_interval`` to tune that cadence.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a D2H task with FlashSAC.")
# --- Environment / task args (mirror the RSL-RL trainer) ---
parser.add_argument("--task", type=str, required=True, help="Registered gym task id (e.g. Reorient-v0).")
parser.add_argument("--num_envs", type=int, default=None, help="Override scene.num_envs.")
parser.add_argument("--seed", type=int, default=None, help="Override agent + env seed.")
parser.add_argument("--grasp_path", type=str, default=None, help="Path to grasp data file (.npy).")
parser.add_argument("--obj_urdf_path", type=str, default=None, help="Path to object URDF.")
parser.add_argument("--obj_scale", type=float, default=None, help="Override object scale.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos via gym.wrappers.RecordVideo.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--distributed", action="store_true", default=False, help="Multi-GPU training.")

# --- FlashSAC-specific scheduling args ---
parser.add_argument("--num_env_steps", type=int, default=None, help="Total env steps (num_interaction_steps * num_envs).")
parser.add_argument("--num_interaction_steps", type=int, default=None, help="Shortcut: sets num_env_steps = N * num_envs.")
parser.add_argument("--updates_per_step", type=float, default=None, help="Gradient updates per env step (float ok).")
parser.add_argument("--batch_size", type=int, default=None, help="Replay sample batch size.")
parser.add_argument("--buffer_min_length", type=int, default=None, help="Transitions (ring + disk) required before updates start.")
parser.add_argument("--buffer_max_length", type=int, default=None, help="Disk-backed replay buffer capacity.")
parser.add_argument("--ring_capacity", type=int, default=None, help="Shared-memory ring buffer capacity (fast path for in-progress episode data).")
parser.add_argument("--save_interval", type=int, default=None, help="Save checkpoint every N interaction steps.")
parser.add_argument("--logging_interval", type=int, default=None, help="Flush metrics to W&B every N interaction steps.")
parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable Weights & Biases logging.")
parser.add_argument("--no_compile", action="store_true", default=False, help="Disable torch.compile in FlashSAC networks.")
parser.add_argument("--no_amp", action="store_true", default=False, help="Disable mixed-precision training.")
parser.add_argument("--run_name", type=str, default="", help="Optional run-name suffix for log dir.")
parser.add_argument("--experiment_name", type=str, default=None, help="Override the logged experiment name.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Keep gym/Isaac's argv pristine (RSL-RL trainer does the same for Hydra).
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch
import tqdm

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Registers D2H tasks on import.
import src.tasks  # noqa: F401
from src.utils import apply_grasp_overrides_from_cli
from src.flash_sac.env_adapter import FlashSACIsaacLabAdapter

# FlashSAC core
from flash_rl.agent import create_agent
from flash_rl.utils import Tensor, WandbTrainerLogger, seeding

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _override_cfg(flash_sac_cfg: Any, args: argparse.Namespace, num_envs: int) -> None:
    """Apply CLI overrides onto the task's FlashSacCfg before we synthesize a flash_rl.Config."""
    if args.seed is not None:
        flash_sac_cfg.seed = args.seed
    if args.experiment_name is not None:
        flash_sac_cfg.experiment_name = args.experiment_name
    if args.run_name:
        flash_sac_cfg.run_name = args.run_name
    if args.num_env_steps is not None:
        flash_sac_cfg.num_env_steps = args.num_env_steps
    if args.updates_per_step is not None:
        flash_sac_cfg.runner.updates_per_interaction_step = args.updates_per_step
    if args.batch_size is not None:
        flash_sac_cfg.agent.sample_batch_size = args.batch_size
    if args.buffer_min_length is not None:
        flash_sac_cfg.agent.buffer_min_length = args.buffer_min_length
    if args.buffer_max_length is not None:
        flash_sac_cfg.agent.buffer_max_length = args.buffer_max_length
    if args.ring_capacity is not None:
        flash_sac_cfg.agent.ring_capacity = args.ring_capacity
    if (
        flash_sac_cfg.agent.ring_capacity > 0
        and flash_sac_cfg.agent.buffer_min_length > flash_sac_cfg.agent.ring_capacity
    ):
        raise ValueError(
            f"buffer_min_length ({flash_sac_cfg.agent.buffer_min_length}) must be <= "
            f"ring_capacity ({flash_sac_cfg.agent.ring_capacity}) when the ring is enabled; "
            f"set --ring_capacity 0 to rely on disk-based sampling instead."
        )
    if args.save_interval is not None:
        flash_sac_cfg.runner.save_checkpoint_per_interaction_step = args.save_interval
    if args.logging_interval is not None:
        flash_sac_cfg.runner.logging_per_interaction_step = args.logging_interval
    if args.device is not None:
        flash_sac_cfg.device = args.device
        flash_sac_cfg.agent.device_type = args.device
    if args.no_compile:
        flash_sac_cfg.agent.use_compile = False
    if args.no_amp:
        flash_sac_cfg.agent.use_amp = False
    # Re-run post_init so seed/num_env_steps propagate into nested configs.
    flash_sac_cfg.__post_init__()
    # Stash num_envs for downstream config synthesis.
    flash_sac_cfg._num_train_envs = num_envs  # type: ignore[attr-defined]


def _setup_log_dir(experiment_name: str, run_name: str) -> str:
    log_root = os.path.abspath(os.path.join("logs", "flash_sac", experiment_name))
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        stamp += f"_{run_name}"
    log_dir = os.path.join(log_root, stamp)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    return log_dir


def main() -> None:
    # --- 1. Load configs ------------------------------------------------------
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    flash_sac_cfg = load_cfg_from_registry(args_cli.task, "flash_sac_cfg_entry_point")

    # --- 2. Apply overrides ---------------------------------------------------
    apply_grasp_overrides_from_cli(env_cfg, args_cli)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = flash_sac_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        flash_sac_cfg.device = f"cuda:{app_launcher.local_rank}"
        flash_sac_cfg.seed = flash_sac_cfg.seed + app_launcher.local_rank
        env_cfg.seed = flash_sac_cfg.seed

    num_envs = int(args_cli.num_envs) if args_cli.num_envs is not None else int(env_cfg.scene.num_envs)
    if args_cli.num_interaction_steps is not None:
        args_cli.num_env_steps = args_cli.num_interaction_steps * num_envs
    _override_cfg(flash_sac_cfg, args_cli, num_envs)

    log_dir = _setup_log_dir(flash_sac_cfg.experiment_name, flash_sac_cfg.run_name)
    env_cfg.log_dir = log_dir

    seeding(flash_sac_cfg.seed)

    # --- 3. Build Isaac Lab env and wrap for FlashSAC -------------------------
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    train_env = FlashSACIsaacLabAdapter(
        env=env,
        device=flash_sac_cfg.device,
        action_bounds=flash_sac_cfg.action_bounds,
        obs_groups=flash_sac_cfg.obs_groups,
        to_numpy=True,
    )
    flash_sac_cfg.agent.asymmetric_observation = train_env.asymmetric_obs

    # --- 4. Assemble flash_rl.Config and agent --------------------------------
    cfg = flash_sac_cfg.to_flash_rl_config(num_train_envs=num_envs, save_path=log_dir)

    _, env_infos = train_env.reset()
    agent = create_agent(
        observation_space=train_env.observation_space,
        action_space=train_env.action_space,
        env_info=env_infos,
        cfg=cfg.agent_cfg,
    )

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), cfg)

    # --- 5. Logger ------------------------------------------------------------
    wandb_logger: Optional[WandbTrainerLogger] = None
    if not args_cli.no_wandb:
        try:
            wandb_logger = WandbTrainerLogger(cfg)
        except Exception as exc:  # pragma: no cover - exercise-path
            print(f"[WARN] WandbTrainerLogger init failed ({exc}); continuing without wandb.")
            wandb_logger = None

    def _log_metric(step: int, **metrics: Any) -> None:
        if wandb_logger is None:
            return
        wandb_logger.update_metric(**metrics)
        wandb_logger.log_metric(step=step)
        wandb_logger.reset()

    # --- 6. Training loop (adapted from FlashSAC_dex/train.py:74-160) ---------
    observations, env_infos = train_env.reset()
    actions: Optional[Tensor] = None
    transition: Optional[dict[str, Tensor]] = None
    update_counter = 0.0
    num_interaction_steps = cfg.num_interaction_steps
    start_time = time.time()

    for interaction_step in tqdm.tqdm(
        range(1, int(num_interaction_steps + 1)),
        smoothing=0.1,
        mininterval=0.5,
    ):
        env_step = interaction_step * num_envs

        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            actions = train_env.action_space.sample()

        assert actions is not None
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        for env_idx in range(num_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        if "episode_info" in env_infos and wandb_logger is not None:
            wandb_logger.update_metric(**env_infos["episode_info"])

        transition = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        agent.process_transition(transition)
        transition["next_observation"] = next_observations
        observations = next_observations

        if agent.can_start_training():
            update_counter += cfg.runner_cfg.updates_per_interaction_step
            while update_counter >= 1:
                update_info = agent.update()
                if wandb_logger is not None:
                    wandb_logger.update_metric(**update_info)
                update_counter -= 1

            if cfg.runner_cfg.metrics_per_interaction_step and interaction_step % cfg.runner_cfg.metrics_per_interaction_step == 0:
                metrics_info = agent.get_metrics()
                if wandb_logger is not None:
                    wandb_logger.update_metric(**metrics_info)

            if cfg.runner_cfg.logging_per_interaction_step and interaction_step % cfg.runner_cfg.logging_per_interaction_step == 0:
                if wandb_logger is not None:
                    wandb_logger.log_metric(step=env_step)
                    wandb_logger.reset()

            if cfg.runner_cfg.save_checkpoint_per_interaction_step and interaction_step % cfg.runner_cfg.save_checkpoint_per_interaction_step == 0:
                save_path = os.path.join(log_dir, f"step{interaction_step}")
                agent.save(save_path)

    save_path = os.path.join(log_dir, "final")
    agent.save(save_path)
    print(f"[INFO] Training finished in {round(time.time() - start_time, 2)} seconds")
    train_env.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
