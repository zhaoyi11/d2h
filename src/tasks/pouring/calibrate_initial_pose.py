"""Refine the configured pouring grasp offline and print reset values; run with --headless."""

import argparse
import json
from pathlib import Path
import sys


def main():
    import warp  # noqa: F401 -- load before AppLauncher, as in instant_dexterity.py
    from isaaclab.app import AppLauncher

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    app = AppLauncher(parser.parse_args()).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.math import quat_mul, subtract_frame_transforms
        import src.tasks  # noqa: F401
        from src.policy.low_level import load_low_level_rsl_rl_policy

        cfg = parse_env_cfg("Pouring_HRL-v0", device="cuda:0", num_envs=1)
        cfg.seed = 0
        cfg.observations.low_level.enable_corruption = False
        env = gym.make("Pouring_HRL-v0", cfg=cfg).unwrapped
        observations, _ = env.reset()
        cmd = env.command_manager.get_term("object_pose")
        robot, obj = env.scene["robot"], env.scene["object"]
        policy = load_low_level_rsl_rl_policy(
            "/home/yizhao/yi/dex_reorient/model_14999.pt", device=env.device,
        )
        pose = obj.data.root_pose_w.clone()
        write_data = env.scene.write_data_to_sim

        def support_object():
            # Offline only: keep the object supported while the policy settles its fingers.
            obj.write_root_pose_to_sim(pose)
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
            write_data()

        env.scene.write_data_to_sim = support_object
        with torch.no_grad():
            for _ in range(90):
                cmd._stepper.step[:] = 0  # Stay at recording frame zero while settling.
                observations, *_ = env.step(policy.act(observations["low_level"]))
        env.scene.write_data_to_sim = write_data
        limits = robot.data.soft_joint_pos_limits
        joints = robot.data.joint_pos.clamp(limits[..., 0], limits[..., 1])
        robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
        env.sim.forward()
        env.scene.update(0.0)
        hand_pos, hand_quat = cmd._current_hand_base_pose_b()
        anchor_quat = quat_mul(pose[:, 3:], pose.new_tensor([[2**-0.5, 2**-0.5, 0., 0.]]))
        offset_pos, offset_quat = subtract_frame_transforms(
            hand_pos, hand_quat, pose[:, :3] - env.scene.env_origins, anchor_quat,
        )
        print(json.dumps({
            "joint_pos": dict(zip(robot.joint_names, joints[0].tolist())),
            "hand_base_to_anchor_pose": torch.cat((offset_pos, offset_quat), dim=-1)[0].tolist(),
        }, indent=2), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
