"""Real IsaacLab checks: python tests/tasks/check_pick_and_place_runtime.py --headless."""

import argparse
from pathlib import Path
import sys


def main():
    import warp  # noqa: F401 -- cuRobo requires site-packages Warp before AppLauncher.
    from isaaclab.app import AppLauncher

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    app = AppLauncher(parser.parse_args()).app
    env = None
    try:
        import gymnasium as gym
        import torch
        from unittest.mock import patch
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
        from pxr import PhysxSchema
        import src.tasks  # noqa: F401
        from src.policy.high_level.gate import LowLevelHandGate

        cfg = parse_env_cfg("PickAndPlace_HRL-v0", device="cuda:0", num_envs=2)
        cfg.seed = 0
        env = gym.make("PickAndPlace_HRL-v0", cfg=cfg).unwrapped
        observations, _ = env.reset()
        cmd = env.command_manager.get_term("object_pose")
        robot, obj, box = env.scene["robot"], env.scene["object"], env.scene["receptive_object"]
        arm = env.action_manager.get_term("arm_action")
        assert observations["low_level"].shape == (2, 155)
        assert cmd.command.shape == (2, 14)
        assert env.action_manager.total_action_dim == 16
        assert arm.action_dim == 0
        assert cmd._stepper.length == 21
        assert abs(arm._mpc.config.optimization_dt / arm._mpc.config.interpolation_steps - env.step_dt) < 1e-9
        assert cfg.sim.gravity == (0.0, 0.0, -1.81)
        for i in range(2):
            prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_{i}/Object")
            assert not PhysxSchema.PhysxRigidBodyAPI(prim).GetDisableGravityAttr().Get()
        for name in ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"):
            assert env.scene[f"{name}_object_s"].data.force_matrix_w.shape == (2, 1, 3, 3)

        # The table supports the pig before hand control: simulate a short settling interval.
        before_z = obj.data.root_pos_w[:, 2].clone()
        for step in range(240):
            # Isolate support from the uncontrolled arm falling during direct physics steps.
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
            env.sim.step(render=False)
            env.scene.update(cfg.sim.dt)
            if step >= 30 and (obj.data.root_lin_vel_w.norm(dim=-1) < 0.05).all():
                break
        assert (obj.data.root_pos_w[:, 2] > before_z - 0.02).all()
        assert (obj.data.root_lin_vel_w.norm(dim=-1) < 0.05).all(), obj.data.root_lin_vel_w
        cmd._steps_since_reset[:] = cmd.cfg.settle_min_steps
        cmd._update_settle_capture()
        assert cmd._grasp_goal_captured.all()
        cmd._update_metrics()
        torch.testing.assert_close(cmd.pose_command_w[:, :3], obj.data.root_pos_w, atol=0.02, rtol=0)
        print("PASS: interfaces, physical tabletop support and settled-pose capture.", flush=True)

        # Root transforms must not rotate or translate the world trajectory accidentally.
        original_root_pos = robot.data.root_pos_w.clone()
        original_root_quat = robot.data.root_quat_w.clone()
        robot.data.root_pos_w[:] += original_root_pos.new_tensor([0.2, -0.1, 0.05])
        robot.data.root_quat_w[:] = original_root_quat.new_tensor([0.70710678, 0, 0, 0.70710678])
        current_pos, current_quat = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w, obj.data.root_quat_w
        )
        ids = torch.arange(2, device=env.device)
        poses = cmd._build_object_trajectories(ids, torch.cat((current_pos, current_quat), dim=-1))
        world_pos, _ = combine_frame_transforms(
            robot.data.root_pos_w[:, None].expand(-1, cmd._stepper.length, -1),
            robot.data.root_quat_w[:, None].expand(-1, cmd._stepper.length, -1), poses[..., :3], poses[..., 3:]
        )
        expected_above, _ = combine_frame_transforms(
            box.data.root_pos_w, box.data.root_quat_w,
            poses.new_tensor(cmd.cfg.above_box_offset).expand(2, -1),
        )
        torch.testing.assert_close(world_pos[:, 0], obj.data.root_pos_w, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(world_pos[:, len(cmd._carry)], expected_above, atol=1e-6, rtol=1e-6)
        robot.data.root_pos_w[:] = original_root_pos
        robot.data.root_quat_w[:] = original_root_quat
        env.reset()

        # Contact confirmation cannot be replaced by simply reaching a pose.
        cmd._stepper.step[:] = 1
        cmd._grasp_goal_captured[:] = True
        with patch("src.tasks.clean_table.mdps.commands.good_object_contact", return_value=torch.zeros(2, dtype=torch.bool, device=env.device)):
            cmd._update_metrics()
            assert not cmd._trajectory_command_achieved.any()
        with patch("src.tasks.clean_table.mdps.commands.good_object_contact", return_value=torch.ones(2, dtype=torch.bool, device=env.device)):
            for i in range(cmd.cfg.grasp_contact_stable_steps):
                cmd._update_metrics()
                assert cmd._trajectory_command_achieved.all().item() == (i == cmd.cfg.grasp_contact_stable_steps - 1)

        # Exercise release/retreat and success gates using controlled real asset state.
        gate = LowLevelHandGate()
        cmd._stepper.step[:] = cmd._stepper.length - 2
        cmd._update_keep_hand_open_metric()
        assert not gate.use_low_level_mask(env).any()
        cmd._stepper.step[:] = cmd._stepper.length - 1
        cmd._update_hand_base_pose_command()
        cmd._update_keep_hand_open_metric()
        torch.testing.assert_close(cmd.hand_base_pose_command_b, cmd._default_hand_base_pose_b)
        assert cmd.metrics["pick_and_place_frame"].tolist() == [-1, -1]
        torch.testing.assert_close(cmd.metrics["pick_and_place_progress"], torch.ones(2, device=env.device))
        deposit_pos, deposit_quat = combine_frame_transforms(
            box.data.root_pos_w, box.data.root_quat_w,
            poses.new_tensor(cmd.cfg.box_target_offset).expand(2, -1),
        )
        state = obj.data.root_state_w.clone()
        state[:, :3], state[:, 3:7], state[:, 7:] = deposit_pos, deposit_quat, 0
        obj.write_root_state_to_sim(state)
        for i in range(5):
            cmd._update_metrics()
            assert cmd.success.all().item() == (i == 4)
        state[:, 7] = 0.1
        obj.write_root_state_to_sim(state)
        cmd._update_metrics()
        assert not cmd.success.any()
        state[:, 7] = 0
        state[:, 0] += 0.3
        obj.write_root_state_to_sim(state)
        cmd._update_metrics()
        assert not cmd.success.any()

        # Reset one env: robot, MPC reference and stages reset; the other env stays untouched.
        displaced = robot.data.default_joint_pos + 0.05
        robot.write_joint_state_to_sim(displaced, torch.full_like(displaced, 0.1))
        other_object = obj.data.root_state_w[1].clone()
        # The real runner wraps env.step(), including automatic resets, in inference_mode.
        with torch.inference_mode():
            env._reset_idx(torch.tensor([0], device=env.device))
        torch.testing.assert_close(robot.data.joint_pos[0], robot.data.default_joint_pos[0])
        torch.testing.assert_close(robot.data.joint_vel[0], torch.zeros_like(displaced[0]))
        torch.testing.assert_close(robot.data.joint_pos[1], displaced[1])
        torch.testing.assert_close(obj.data.root_state_w[1], other_object)
        torch.testing.assert_close(arm._ref_js.position[0], robot.data.joint_pos[0, arm._ordered_joint_ids])
        assert cmd._stepper.step.tolist() == [0, cmd._stepper.length - 1]
        assert cmd.metrics["keep_hand_open"].tolist() == [1, 1]
        assert not cmd._grasp_goal_captured[0]
        fallen = obj.data.root_state_w.clone()
        fallen[0, 2] = 0.10
        obj.write_root_state_to_sim(fallen)
        assert env.termination_manager.compute()[0]
        # Warm physics/MPC buffers in inference mode, then exercise an automatic timeout reset.
        with torch.inference_mode():
            action = torch.zeros(2, 16, device=env.device)
            for _ in range(8):
                env.step(gate.apply(action, env))
            env.episode_length_buf[0] = env.max_episode_length
            _, _, _, truncated, _ = env.step(gate.apply(action, env))
            assert truncated[0]
        print("PASS: world/root transforms, contact gating, release/retreat, success rejection, subset and automatic inference-mode resets.", flush=True)
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
