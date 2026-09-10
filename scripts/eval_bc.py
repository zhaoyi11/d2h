"""Evaluate a behavior-cloning checkpoint on Rotate_Object_BC-v0."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="Path to a BC best.pt checkpoint.")
parser.add_argument("--task", default="Rotate_Object_BC-v0", help="Registered Gym task to evaluate.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--episodes", type=int, default=100, help="Number of completed episodes to score.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import src.tasks  # noqa: F401, E402
from dataets.bc import (  # noqa: E402
    MODEL_TYPE,
    BehaviorCloningConfig,
    BehaviorCloningPolicy,
)
from src.policy.instant_dexterity_recording import (  # noqa: E402
    hand_target_to_teacher_action,
)


def _load_policy(checkpoint_path: str, device: torch.device) -> BehaviorCloningPolicy:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BC checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("model_type") != MODEL_TYPE:
        raise ValueError(f"Checkpoint is not a {MODEL_TYPE!r} model: {path}")
    config = BehaviorCloningConfig(**checkpoint["config"])
    model = BehaviorCloningPolicy(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print(
        f"[INFO]: Loaded {path} at epoch {checkpoint.get('epoch', 'unknown')} "
        f"(obs={config.state_dim}, action={config.action_dim}, future={config.future_length}).",
        flush=True,
    )
    return model


def main() -> None:
    if args_cli.num_envs < 1 or args_cli.episodes < 1:
        raise ValueError("--num_envs and --episodes must be positive.")

    env = None
    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=True,
        )
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg)
        root = env.unwrapped
        observations, _ = env.reset()

        device = torch.device(root.device)
        policy = _load_policy(args_cli.checkpoint, device)
        policy_obs = observations["policy"]
        if policy_obs.shape[-1] != policy.config.state_dim:
            raise RuntimeError(
                f"Task observation dim {policy_obs.shape[-1]} does not match "
                f"checkpoint dim {policy.config.state_dim}."
            )
        if root.action_manager.total_action_dim != policy.config.action_dim:
            raise RuntimeError(
                f"Task action dim {root.action_manager.total_action_dim} does not match "
                f"checkpoint dim {policy.config.action_dim}."
            )

        hand_action = root.action_manager.get_term("hand_action")
        if not isinstance(hand_action.cfg.alpha, float):
            raise RuntimeError("Rotate_Object_BC requires one scalar hand EMA alpha.")
        limits = hand_action._asset.data.soft_joint_pos_limits[:, hand_action._joint_ids]

        completed = 0
        successes = 0
        with torch.inference_mode():
            while completed < args_cli.episodes:
                bc_action = policy(observations["policy"])[:, 0, :]
                hand_raw_action = hand_target_to_teacher_action(
                    bc_action[:, 7:],
                    hand_action._prev_applied_actions,
                    limits[:, :, 0],
                    limits[:, :, 1],
                    alpha=hand_action.cfg.alpha,
                )
                env_action = torch.cat((bc_action[:, :7], hand_raw_action), dim=1)
                observations, _, terminated, truncated, _ = env.step(env_action)

                done_ids = (terminated | truncated).nonzero().flatten()
                if done_ids.numel() == 0:
                    continue
                success = root.termination_manager.get_term("success").index_select(0, done_ids)
                take = min(args_cli.episodes - completed, done_ids.numel())
                successes += int(success[:take].sum().item())
                completed += take

        success_rate = successes / completed
        print(
            f"[RESULT]: episodes={completed} successes={successes} "
            f"success_rate={success_rate:.3f}",
            flush=True,
        )
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
