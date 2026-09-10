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
        shapes = {name: value.shape for name, value in observations.items()}
        actions = torch.zeros(env.action_space.shape, device=env.device)
        with torch.inference_mode():
            for _ in range(args.steps):
                observations, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(reward).all()
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
        print("RUNTIME PASS", args.task, dict(shapes), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
