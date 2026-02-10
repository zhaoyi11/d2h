# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from src.utils.helper import dump_pickle

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--grasp_path", type=str, default=None, help="Path to grasp data file (.npy).")
parser.add_argument("--obj_urdf_path", type=str, default=None, help="Path to object URDF.")
parser.add_argument("--obj_scale", type=float, default=None, help="Override object scale from grasp data.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--registry_name", type=str, default=None, help="The name of the wandb motion artifact registry.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime


from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import our custom env registry so that Gym knows about "AnyGrasp"/"AnyGrasp-v0"
# before Hydra / gym.spec() tries to look them up.
import src.env  # noqa: F401

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _safe_wandb_scalar(value):
    """Convert logger values to stable python scalars before passing to wandb."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except Exception:
        return 0.0


def _sanitize_wandb_config(value, depth: int = 0, max_depth: int = 4):
    """Recursively sanitize config values so wandb receives JSON-like payloads only."""
    if depth >= max_depth:
        return str(type(value).__name__)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _safe_wandb_scalar(value)
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        out = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= 200:
                out["__truncated__"] = f"{len(value) - 200} more entries"
                break
            out[str(key)] = _sanitize_wandb_config(item, depth + 1, max_depth)
        return out
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        out = [_sanitize_wandb_config(item, depth + 1, max_depth) for item in items[:200]]
        if len(items) > 200:
            out.append(f"...({len(items) - 200} more items)")
        return out
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _patch_rsl_wandb_writer():
    """Patch rsl-rl wandb writer to avoid crashes from unsupported payloads."""
    os.environ.setdefault("WANDB_CONSOLE", "off")

    try:
        import wandb
        from rsl_rl.utils import wandb_utils
    except Exception as exc:
        print(f"[WARN] Failed to set up wandb compatibility patch: {exc}")
        return

    writer_cls = wandb_utils.WandbSummaryWriter
    if getattr(writer_cls, "_d2h_safe_patch", False):
        return

    original_add_scalar = writer_cls.add_scalar

    def _patched_add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False):
        return original_add_scalar(
            self,
            tag,
            _safe_wandb_scalar(scalar_value),
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )

    def _patched_store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        if wandb.run is None:
            return

        cfg_updates = {
            "runner_cfg": _sanitize_wandb_config(runner_cfg),
            "policy_cfg": _sanitize_wandb_config(policy_cfg),
            "alg_cfg": _sanitize_wandb_config(alg_cfg),
        }
        for key, payload in cfg_updates.items():
            try:
                wandb.config.update({key: payload}, allow_val_change=True)
            except Exception as exc:
                print(f"[WARN] Failed to upload wandb config key '{key}': {exc}")

        env_summary = {"cfg_type": type(env_cfg).__name__}
        try:
            if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
                env_summary["num_envs"] = int(env_cfg.scene.num_envs)
            if hasattr(env_cfg, "sim") and hasattr(env_cfg.sim, "device"):
                env_summary["sim_device"] = str(env_cfg.sim.device)
        except Exception:
            pass

        try:
            wandb.config.update({"env_cfg": env_summary}, allow_val_change=True)
        except Exception as exc:
            print(f"[WARN] Failed to upload wandb env summary: {exc}")

    writer_cls.add_scalar = _patched_add_scalar
    writer_cls.store_config = _patched_store_config
    writer_cls._d2h_safe_patch = True


def _apply_grasp_overrides(env_cfg, args_cli):
    """Apply grasp/object overrides for both legacy and current env config schemas."""
    if hasattr(env_cfg, "grasp_path"):
        if args_cli.grasp_path is not None:
            env_cfg.grasp_path = args_cli.grasp_path
        if args_cli.obj_urdf_path is not None:
            env_cfg.object_urdf_path = args_cli.obj_urdf_path
        if args_cli.obj_scale is not None:
            env_cfg.object_scale_override = args_cli.obj_scale

        # Re-run post init after CLI overrides so grasp/object init state is applied.
        if env_cfg.grasp_path is not None and env_cfg.object_urdf_path is not None:
            env_cfg.__post_init__()
        return

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # rsl-rl's WandB writer reads entity from WANDB_USERNAME.
    if getattr(args_cli, "wandb_entity", None):
        os.environ["WANDB_USERNAME"] = args_cli.wandb_entity
    if getattr(agent_cfg, "logger", None) == "wandb":
        _patch_rsl_wandb_writer()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _apply_grasp_overrides(env_cfg, args_cli)

    if args_cli.motion_file is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    registry_name = args_cli.registry_name
    if registry_name is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        # load the motion file from the wandb registry
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"
        import pathlib

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
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

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name or "none"
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # save resume path before creating a new log_dir
    resume_path = None
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
