"""Run real manager/physics checks: python tests/tasks/check_pouring_runtime.py --headless."""

import argparse
from pathlib import Path
import sys


def main():
    # Match instant_dexterity's launch order so cuRobo gets the installed Warp.
    import warp  # noqa: F401
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
        from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, matrix_from_quat, quat_apply, quat_apply_inverse, scale_transform
        import src.tasks  # noqa: F401

        cfg = parse_env_cfg("Pouring_HRL-v0", device="cuda:0", num_envs=2)
        cfg.seed = 0
        env = gym.make("Pouring_HRL-v0", cfg=cfg).unwrapped
        observations, _ = env.reset()
        cmd = env.command_manager.get_term("object_pose")
        initial_joint_pos = cmd.robot.data.joint_pos.clone()
        assert observations["low_level"].shape == (2, 155)
        # Identical physical orientations must reach the policy as positive identity,
        # even when PhysX and the commanded hand pose use opposite quaternion signs.
        def check_initial_orientation():
            manager = env.observation_manager
            index = manager.active_terms["low_level"].index("goal_quat_diff")
            offset = sum(dim[0] for dim in manager.group_obs_term_dim["low_level"][:index])
            error = manager.compute_group("low_level")[:, offset:offset + 4]
            expected = error.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(error)
            torch.testing.assert_close(error, expected, atol=1e-4, rtol=0)

        check_initial_orientation()
        from src.policy.high_level.gate import LowLevelHandGate
        assert LowLevelHandGate().use_low_level_mask(env).all()
        assert cmd.command.shape == (2, 14)
        assert env.action_manager.get_term("arm_action").action_dim == 0
        mpc_cfg = env.action_manager.get_term("arm_action")._mpc.config
        assert abs(mpc_cfg.optimization_dt / mpc_cfg.interpolation_steps - env.step_dt) < 1e-9
        assert env.action_manager.total_action_dim == 16
        assert "receptive_object" not in env.scene.rigid_objects
        assert cmd._bottle_contact.data.force_matrix_w.shape == (2, 1, 17, 3)
        for name in ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"):
            assert env.scene[f"{name}_object_s"].data.force_matrix_w.shape == (2, 1, 2, 3)

        # Keep a rough MANO resemblance; grasp stability takes priority over exact landmarks.
        from src.tasks.pick_and_place.export_hand_keypoints import load_hand_keypoints
        recorded = load_hand_keypoints(
            cfg.commands.object_pose.trajectory_path,
            Path(__file__).resolve().parents[2] / "src/tasks/pick_and_place/mano_right.xml",
        )
        landmark_names = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3",
                          "thumb_dip", "dip", "dip_2", "dip_3"]
        landmark_ids = [cmd.robot.find_bodies(name)[0][0] for name in landmark_names]
        offsets = torch.tensor([[0, -0.06, -0.015]] + [[0, -0.048, 0.015]] * 3 + [[0, 0, 0]] * 4,
                               device=env.device).expand(2, -1, -1)
        actual = cmd.robot.data.body_pos_w[:, landmark_ids] + quat_apply(
            cmd.robot.data.body_quat_w[:, landmark_ids], offsets)
        # Frame zero is an approach pose. Compare against the recording's hand
        # shapes in object coordinates, since the robot starts ready to grasp.
        reference = torch.cat((torch.as_tensor(recorded["qpos_finger_right"][:, :4, :3], device=env.device),
                               torch.as_tensor(recorded["qpos_pip_right"][:, :4], device=env.device)), dim=1)
        object_pos = torch.as_tensor(recorded["object_pos"], device=env.device)
        object_quat = torch.as_tensor(recorded["object_quat_wxyz"], device=env.device)
        relative = quat_apply_inverse(object_quat[:, None].expand(-1, 8, -1), reference - object_pos[:, None])
        expected = quat_apply(object_quat[0].expand_as(object_quat[:, None].expand(-1, 8, -1)), relative)
        expected += expected.new_tensor(cmd.cfg.initial_object_position)
        errors = ((actual[:, None] - env.scene.env_origins[:, None, None] - expected).square().sum(-1).mean(-1)).sqrt()
        rms, closest_frame = errors.min(dim=1)
        print("Closest recorded grasp frames:", closest_frame.tolist(), "landmark RMS:", rms.tolist(), flush=True)
        assert (rms < 0.10).all(), rms
        limits = cmd.robot.data.soft_joint_pos_limits
        assert (cmd.robot.data.joint_pos >= limits[..., 0] - 1e-5).all()
        assert (cmd.robot.data.joint_pos <= limits[..., 1] + 1e-5).all()
        initial = env.scene["object"].data.root_pos_w.clone()
        cmd.robot.set_joint_position_target(initial_joint_pos)
        for _ in range(4):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(env.physics_dt)
        displacement = (env.scene["object"].data.root_pos_w - initial).norm(dim=-1)
        assert (displacement < 0.01).all(), displacement
        print("Initial landmark RMS (m):", rms.tolist(), "held-pose object displacement (m):", displacement.tolist(), flush=True)
        env.reset()

        # Regression: the first RL action used to launch the object at about 0.78 m/s.
        from src.policy.low_level import load_low_level_rsl_rl_policy
        policy = load_low_level_rsl_rl_policy(
            "/home/yizhao/yi/dex_reorient/model_14999.pt", device=env.device,
        )
        gate = LowLevelHandGate()
        observations, _ = env.reset()
        peak_speed = torch.zeros(env.num_envs, device=env.device)
        with torch.no_grad():
            for step in range(60):
                action = gate.apply(policy.act(observations["low_level"]), env)
                observations, *_ = env.step(action)
                if step < 6:
                    peak_speed = torch.maximum(peak_speed, cmd.object.data.root_lin_vel_w.norm(dim=-1))
            assert (peak_speed < 0.35).all(), peak_speed
            assert (cmd.metrics["pouring_frame"] > 0).all(), cmd.metrics["pouring_frame"]
            assert (cmd.metrics["hand_base_object_error"] < 0.06).all(), cmd.metrics["hand_base_object_error"]
            print("Policy startup peak speed (m/s):", peak_speed.tolist(),
                  "recording frames after 60 steps:", cmd.metrics["pouring_frame"].tolist(), flush=True)
            env.reset()

        # Exercise translations AND nonidentity robot roots using the real frame utilities.
        root_pos = cmd.robot.data.root_pos_w.clone()
        root_quat = cmd.robot.data.root_quat_w.clone()
        cmd.robot.data.root_pos_w[:] += root_pos.new_tensor([0.2, -0.1, 0.05])
        cmd.robot.data.root_quat_w[:] = root_quat.new_tensor([0.70710678, 0, 0, 0.70710678])
        ids = torch.arange(2, device=env.device)
        poses_b = cmd._build_object_trajectories(ids, cmd.pose_command_b)
        pos_w, quat_w = combine_frame_transforms(
            cmd.robot.data.root_pos_w[:, None].expand(-1, poses_b.shape[1], -1),
            cmd.robot.data.root_quat_w[:, None].expand(-1, poses_b.shape[1], -1),
            poses_b[..., :3], poses_b[..., 3:],
        )
        shift = cmd._demo_poses.new_tensor(cmd.cfg.initial_object_position) - cmd._demo_poses[0, :3]
        expected_pos = cmd._demo_poses[None, :, :3] + shift + env.scene.env_origins[:, None]
        torch.testing.assert_close(pos_w, expected_pos, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(quat_w, cmd._demo_poses[None, :, 3:].expand_as(quat_w))
        # Local +90-degree X rotation keeps X, maps Y to object Z, and Z to -object Y.
        cmd.pose_command_b[:] = poses_b[:, -1]
        cmd._update_hand_base_pose_command()
        torch.testing.assert_close(cmd.anchor_pose_command_b[:, :3], cmd.pose_command_b[:, :3])
        object_axes = matrix_from_quat(cmd.pose_command_b[:, 3:])
        expected_axes = object_axes[:, :, [0, 2, 1]].clone()
        expected_axes[:, :, 2] *= -1
        torch.testing.assert_close(matrix_from_quat(cmd.anchor_pose_command_b[:, 3:]), expected_axes)
        anchor_pos, anchor_quat = combine_frame_transforms(
            cmd.hand_base_pose_command_b[:, :3], cmd.hand_base_pose_command_b[:, 3:],
            cmd._hand_base_to_anchor_pose[:, :3], cmd._hand_base_to_anchor_pose[:, 3:],
        )
        torch.testing.assert_close(anchor_pos, cmd.pose_command_b[:, :3])
        torch.testing.assert_close(matrix_from_quat(anchor_quat), expected_axes)
        cmd.robot.data.root_pos_w[:] = root_pos
        cmd.robot.data.root_quat_w[:] = root_quat
        cmd.reset()

        # A reset must restore arm AND finger joints, including a reset of only one env.
        robot = env.scene["robot"]
        arm = env.action_manager.get_term("arm_action")
        displaced = initial_joint_pos + 0.05
        moving = torch.full_like(displaced, 0.1)
        robot.write_joint_state_to_sim(displaced, moving)
        env._reset_idx(torch.tensor([0], device=env.device))
        torch.testing.assert_close(robot.data.joint_pos[0], initial_joint_pos[0])
        torch.testing.assert_close(robot.data.joint_vel[0], torch.zeros_like(moving[0]))
        torch.testing.assert_close(robot.data.joint_pos[1], displaced[1])
        torch.testing.assert_close(robot.data.joint_vel[1], moving[1])
        env._reset_idx(torch.tensor([1], device=env.device))
        torch.testing.assert_close(robot.data.joint_pos, initial_joint_pos)
        torch.testing.assert_close(arm._ref_js.position, robot.data.joint_pos[:, arm._ordered_joint_ids])
        env.sim.forward()
        env.scene.update(0.0)

        def check_start_pose(position_tolerance=0.002, orientation_tolerance=0.01):
            pos, quat = cmd._current_hand_base_pose_b()
            pos_error, rot_error = compute_pose_error(
                pos, quat, cmd.hand_base_pose_command_b[:, :3], cmd.hand_base_pose_command_b[:, 3:]
            )
            assert (pos_error.norm(dim=-1) < position_tolerance).all(), pos_error
            assert (rot_error.norm(dim=-1) < orientation_tolerance).all(), rot_error

        check_start_pose()
        hand_action = env.action_manager.get_term("hand_action")
        hand_ids = hand_action._joint_ids
        hand_limits = robot.data.soft_joint_pos_limits[:, hand_ids]
        action = scale_transform(initial_joint_pos[:, hand_ids], hand_limits[..., 0], hand_limits[..., 1])
        # Fitted reset joints must not redefine the gate's zero-joint open pose.
        assert torch.count_nonzero(robot.data.default_joint_pos[:, hand_ids]) == 0
        open_action = scale_transform(torch.zeros_like(action), hand_limits[..., 0], hand_limits[..., 1])
        torch.testing.assert_close(LowLevelHandGate()._stretch_action(env), 0.5 * open_action)
        for restart in range(2):
            if restart:
                # Warm the MPC at a different target before starting another episode.
                for _ in range(12):
                    cmd.hand_base_pose_command_b[:, :3] += action.new_tensor([-0.06, 0.0, 0.08])
                    env.step(action)
                env.reset()
                check_initial_orientation()
                check_start_pose()
            # Isolate MPC reset/floating checks from intentional initial grasp contacts.
            pose = env.scene["object"].data.root_pose_w.clone()
            pose[:, 1] += 1.0
            env.scene["object"].write_root_pose_to_sim(pose)
            initial = pose[:, :3].clone()
            for _ in range(8):
                env.step(action)
                # Allow drive settling at the fitted arm configuration after exact reset checks.
                check_start_pose(position_tolerance=0.01, orientation_tolerance=0.05)
            torch.testing.assert_close(env.scene["object"].data.root_pos_w, initial, atol=1e-5, rtol=0)
            assert not cmd.gravity_enabled.any()

        # Inject deterministic filtered contact to check latch persistence and partial reset.
        force_matrix = torch.zeros_like(cmd._bottle_contact.data.force_matrix_w_history)
        force_matrix[0, -1, 0, 0, 2] = 2.0
        from types import SimpleNamespace
        sensor = cmd._bottle_contact
        cmd._bottle_contact = SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=force_matrix))
        cmd.compute(env.step_dt)
        assert cmd.gravity_enabled.tolist() == [True, False]
        assert cmd.metrics["keep_hand_open"].tolist() == [0.0, 0.0]
        torch.testing.assert_close(env.scene["object"]._external_force_b[0, 0], cmd._mass[0, 0] * cmd._gravity)
        force_matrix.zero_()
        cmd.compute(env.step_dt)
        assert cmd.gravity_enabled.tolist() == [True, False]
        cmd.reset(torch.tensor([1], device=env.device))
        assert cmd.gravity_enabled.tolist() == [True, False]
        assert env.scene["object"].has_external_wrench
        force_matrix[1, -1, 0, 0, 2] = 2.0
        cmd.compute(env.step_dt)
        assert cmd.gravity_enabled.tolist() == [True, False], "Stale contact must not relatch on reset."
        env._sim_step_counter += env.cfg.decimation
        cmd.compute(env.step_dt)
        cmd.reset(torch.tensor([0], device=env.device))
        assert cmd.gravity_enabled.tolist() == [False, True]
        assert cmd.metrics["keep_hand_open"][0] == 0
        assert env.scene["object"].has_external_wrench
        cmd._bottle_contact = sensor
        cmd.reset()

        # The contact-confirmation stage must not advance on pose tracking alone.
        grasp_step = cmd.cfg.trajectory_segment_steps[0] + 1
        cmd._stepper.step[:] = grasp_step
        hand_pos, hand_quat = cmd._current_hand_base_pose_b()
        object_pos, object_quat = cmd._current_object_pose_b()
        cmd.hand_base_pose_command_b[:] = torch.cat((hand_pos, hand_quat), dim=1)
        cmd.pose_command_b[:] = torch.cat((object_pos, object_quat), dim=1)
        with patch("src.tasks.pouring.commands.contacts", return_value=torch.ones(2, dtype=torch.bool, device=env.device)):
            for i in range(3):
                cmd._update_metrics()
                assert cmd._trajectory_command_achieved.all().item() == (i == 2)
        cmd._update_keep_hand_open_metric()
        assert not cmd.metrics["keep_hand_open"].any()
        cmd._stepper.step[:] = cmd._stepper.length - 1
        for i in range(5):
            cmd._update_metrics()
            assert cmd.success.all().item() == (i == 4)
        cmd.reset(torch.tensor([0], device=env.device))
        assert cmd.success.tolist() == [False, True]

        # Check real PhysX hand contact, rather than only the synthetic latch input above.
        env.reset()
        tip_ids, _ = cmd.robot.find_bodies(".*fingertip.*")
        pose = env.scene["object"].data.root_pose_w.clone()
        pose[:, :3] = cmd.robot.data.body_pos_w[:, tip_ids].mean(dim=1)
        env.scene["object"].write_root_pose_to_sim(pose)
        for _ in range(12):
            env.step(action)
            if cmd.gravity_enabled.all():
                break
        assert cmd.gravity_enabled.all(), "Bottle contact filter did not detect an overlapping LEAP hand."
        assert torch.isfinite(cmd.command).all()
        # Match the runner's automatic reset context after the normal-mode checks.
        with torch.inference_mode():
            env._reset_idx(torch.tensor([0], device=env.device))
        assert cmd.gravity_enabled.tolist() == [False, True]
        print("PASS: pouring frames, interfaces, aligned joint/MPC resets, floating start, grasp/success gates, gravity latch, subset resets, and real hand contacts.", flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        # Kit shutdown can terminate with status zero before an exception propagates.
        import os
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
