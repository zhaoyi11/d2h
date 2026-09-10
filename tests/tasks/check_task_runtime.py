"""Smoke-test one task's manager setup, steps, and a subset reset under IsaacLab.

Example: python tests/tasks/check_task_runtime.py --task Clean_Table_HRL-v0 --headless
"""
import argparse
from pathlib import Path
import sys


def main():
    import warp  # noqa: F401 -- preserve the runtime launch order
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--checkpoint", help="Optional frozen hand-policy checkpoint to drive low_level observations.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        import src.tasks  # noqa: F401

        cfg = parse_env_cfg(args.task, device=args.device, num_envs=2)
        cfg.seed = 0
        env = gym.make(args.task, cfg=cfg).unwrapped
        observations, _ = env.reset()
        hrl = args.task.endswith("_HRL-v0")
        if hrl:
            assert cfg.rewards is None and cfg.curriculum is None
            assert not env.reward_manager.active_terms
            assert not env.curriculum_manager.active_terms
        shapes = {name: value.shape for name, value in observations.items()}
        actions = torch.zeros(env.action_space.shape, device=env.device)
        policy = None
        if args.checkpoint:
            from src.policy.low_level import load_low_level_rsl_rl_policy

            policy = load_low_level_rsl_rl_policy(
                args.checkpoint, device=env.device,
                expected_obs_dim=observations["low_level"].shape[-1],
                expected_action_dim=env.action_manager.total_action_dim,
            )
        with torch.inference_mode():
            for _ in range(args.steps):
                if policy is not None:
                    actions = policy.act(observations["low_level"])
                    assert torch.isfinite(actions).all()
                observations, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(reward).all()
                if hrl:
                    assert torch.count_nonzero(reward) == 0
                assert terminated.shape == truncated.shape == (2,)
                for name, value in observations.items():
                    assert value.shape == shapes[name]
                    assert torch.isfinite(value).all(), name
            other_steps = env.episode_length_buf[1].clone()
            other_joints = env.scene["robot"].data.joint_pos[1].clone()
            env._reset_idx(torch.tensor([0], device=env.device))
            assert env.episode_length_buf[0] == 0
            assert env.episode_length_buf[1] == other_steps
            torch.testing.assert_close(env.scene["robot"].data.joint_pos[1], other_joints)
            observations, reward, *_ = env.step(actions)
            assert torch.isfinite(reward).all()
            if hrl:
                assert torch.count_nonzero(reward) == 0
        print("RUNTIME PASS", args.task, dict(shapes), flush=True)
    except BaseException:
        import os
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Kit shutdown can otherwise hide exceptions with a zero exit status.
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
