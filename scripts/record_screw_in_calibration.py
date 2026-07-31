"""Physically screw the table leg into its socket and record the motion for later unscrewing."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/unscrew/screw_in_calibration.npz"),
    help="Compressed NumPy recording to write.",
)
parser.add_argument("--hold_duration", type=float, default=1.0)
parser.add_argument("--turn_duration", type=float, default=12.0)
parser.add_argument("--settle_duration", type=float, default=2.0)
parser.add_argument("--stable_window", type=float, default=0.5)
parser.add_argument(
    "--insertion_force",
    type=float,
    default=10.0,
    help="Additional world -Z force applied only during helical insertion (N).",
)
parser.add_argument(
    "--insertion_torque",
    type=float,
    default=1.0,
    help="Additional world -Z torque applied only during helical insertion (Nm).",
)
parser.add_argument("--max_force", type=float, default=20.0, help="Total wrench force cap (N).")
parser.add_argument("--max_torque", type=float, default=2.0, help="Total wrench torque cap (Nm).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
for duration_name in ("hold_duration", "turn_duration", "settle_duration", "stable_window"):
    if getattr(args_cli, duration_name) <= 0.0:
        parser.error(f"--{duration_name} must be positive.")
if args_cli.stable_window > args_cli.settle_duration:
    parser.error("--stable_window cannot exceed --settle_duration.")
if args_cli.insertion_force < 0.0 or args_cli.insertion_torque < 0.0:
    parser.error("--insertion_force and --insertion_torque must be non-negative.")
if args_cli.max_force <= 0.0 or args_cli.max_torque <= 0.0:
    parser.error("--max_force and --max_torque must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


try:
    import numpy as np  # noqa: E402
    import torch  # noqa: E402
    import isaaclab.sim as sim_utils  # noqa: E402
    from isaaclab.scene import InteractiveScene  # noqa: E402
    from isaaclab.sim import SimulationContext  # noqa: E402
    from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms  # noqa: E402

    from src.tasks.unscrew import env_cfg as task_env_cfg  # noqa: E402
    from src.tasks.unscrew.mdps import physics_verifier, trajectory  # noqa: E402
    from src.tasks.unscrew.trajectory_recording import (  # noqa: E402
        build_screw_in_recording,
        save_screw_in_recording,
    )
except BaseException:
    traceback.print_exc()
    simulation_app.app.post_uncancellable_quit(1)
    simulation_app.close()
    raise SystemExit(1)


DT = 1.0 / 120.0
POSITION_STIFFNESS = 200.0
POSITION_DAMPING = 4.0
ROTATION_STIFFNESS = 0.5
ROTATION_DAMPING = 0.01
YAW_TOLERANCE = 0.35
POSITION_TOLERANCE = 0.005
ORIENTATION_TOLERANCE = 0.35
LINEAR_SPEED_TOLERANCE = 0.01
ANGULAR_SPEED_TOLERANCE = 0.1
STABLE_POSITION_DRIFT_TOLERANCE = 0.002
STABLE_ORIENTATION_DRIFT_TOLERANCE = np.deg2rad(2.0)
CALIBRATION_TWIST_SEGMENTS = 12
CALIBRATION_TWIST_TOTAL_ANGLE = 6.0 * np.pi
CALIBRATION_THREAD_PITCH = 0.015


def _step_count(duration: float) -> int:
    return max(round(duration / DT), 1)


def _object_pose_w(obj) -> torch.Tensor:
    return torch.cat((obj.data.root_pos_w, obj.data.root_quat_w), dim=-1)


def _mean_pose(poses: torch.Tensor) -> torch.Tensor:
    position = poses[:, :3].mean(dim=0)
    quaternions = poses[:, 3:7].clone()
    same_hemisphere = (quaternions @ quaternions[0]) >= 0.0
    quaternions[~same_hemisphere] *= -1.0
    quaternion = quaternions.mean(dim=0)
    quaternion /= torch.linalg.vector_norm(quaternion)
    return torch.cat((position, quaternion))


def _poses_in_receptive_frame(poses_w: torch.Tensor, receptive) -> torch.Tensor:
    count = poses_w.shape[0]
    receptive_pos_w = receptive.data.root_pos_w[:1].expand(count, -1)
    receptive_quat_w = receptive.data.root_quat_w[:1].expand(count, -1)
    pos_r, quat_r = subtract_frame_transforms(
        receptive_pos_w,
        receptive_quat_w,
        poses_w[:, :3],
        poses_w[:, 3:7],
    )
    return torch.cat((pos_r, quat_r), dim=-1)


def _ideal_screw_targets() -> torch.Tensor:
    turn_steps_per_segment = _step_count(
        args_cli.turn_duration / CALIBRATION_TWIST_SEGMENTS
    )
    segment_steps = (
        (0, 0)
        + (turn_steps_per_segment,) * CALIBRATION_TWIST_SEGMENTS
        + (0, 0)
    )
    installed_pose = torch.tensor(
        (*task_env_cfg.INSTALLED_OBJECT_POS, *task_env_cfg.IDENTITY_QUAT), dtype=torch.float32
    )
    ideal_unscrew = trajectory.build_unscrew_object_pose_sequence(
        installed_pose,
        segment_steps=segment_steps,
        twist_total_angle=CALIBRATION_TWIST_TOTAL_ANGLE,
        twist_segments=CALIBRATION_TWIST_SEGMENTS,
        thread_pitch=CALIBRATION_THREAD_PITCH,
        extraction_height=0.0,
    )
    return torch.flip(ideal_unscrew, dims=(0,))


def _make_scene_cfg(initial_pose_w: torch.Tensor):
    scene_cfg = task_env_cfg.SceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=False)
    scene_cfg.robot = None
    scene_cfg.object.init_state.pos = tuple(float(value) for value in initial_pose_w[:3])
    scene_cfg.object.init_state.rot = tuple(float(value) for value in initial_pose_w[3:7])
    return scene_cfg


def main() -> None:
    screw_targets_cpu = _ideal_screw_targets()
    hold_steps = _step_count(args_cli.hold_duration)
    settle_steps = _step_count(args_cli.settle_duration)
    stable_window_steps = _step_count(args_cli.stable_window)
    scene = None
    obj = None
    sim = None
    try:
        sim = SimulationContext(
            sim_utils.SimulationCfg(dt=DT, gravity=(0.0, 0.0, -9.81), device=args_cli.device)
        )
        scene = InteractiveScene(_make_scene_cfg(screw_targets_cpu[0]))
        sim.reset()
        scene.reset()
        scene.update(DT)

        obj = scene["object"]
        receptive = scene["receptive_object"]
        screw_targets_w = screw_targets_cpu.to(device=sim.device)
        mass = obj.root_physx_view.get_masses().to(device=sim.device)[:, 0]
        gravity_w = screw_targets_w.new_tensor((0.0, 0.0, -9.81))
        zero_wrench = torch.zeros(1, 1, 3, device=sim.device)

        def track_target(
            target_w: torch.Tensor,
            insertion_force: float = 0.0,
            insertion_torque: float = 0.0,
        ):
            current_pose_w = _object_pose_w(obj)
            command = physics_verifier.bounded_pd_wrench(
                current_pose_w,
                target_w[None],
                obj.data.root_lin_vel_w,
                obj.data.root_ang_vel_w,
                mass,
                gravity_w,
                position_stiffness=POSITION_STIFFNESS,
                position_damping=POSITION_DAMPING,
                rotation_stiffness=ROTATION_STIFFNESS,
                rotation_damping=ROTATION_DAMPING,
                max_force=args_cli.max_force,
                max_torque=args_cli.max_torque,
            )
            force_w = command.force_w.clone()
            force_w[:, 2] -= insertion_force
            force_norm = torch.linalg.vector_norm(force_w, dim=-1, keepdim=True)
            force_saturated = command.force_saturated | (force_norm[:, 0] > args_cli.max_force)
            force_w *= torch.clamp(
                force_w.new_tensor(args_cli.max_force)
                / force_norm.clamp_min(torch.finfo(force_w.dtype).eps),
                max=1.0,
            )
            torque_w = command.torque_w.clone()
            torque_w[:, 2] -= insertion_torque
            torque_norm = torch.linalg.vector_norm(torque_w, dim=-1, keepdim=True)
            torque_saturated = command.torque_saturated | (
                torque_norm[:, 0] > args_cli.max_torque
            )
            torque_w *= torch.clamp(
                torque_w.new_tensor(args_cli.max_torque)
                / torque_norm.clamp_min(torch.finfo(torque_w.dtype).eps),
                max=1.0,
            )
            command = physics_verifier.WrenchCommand(
                force_w,
                torque_w,
                force_saturated,
                torque_saturated,
            )
            obj.set_external_force_and_torque(
                command.force_w[:, None, :], command.torque_w[:, None, :], is_global=True
            )
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(DT)
            return command

        for _ in range(hold_steps):
            track_target(screw_targets_w[0])

        measured_poses = []
        geometric_yaw = torch.zeros(1, device=sim.device)
        previous_quat_w = obj.data.root_quat_w.clone()
        force_saturation_count = 0
        torque_saturation_count = 0
        peak_force = 0.0
        peak_torque = 0.0
        for sample_index, target_w in enumerate(screw_targets_w):
            command = track_target(
                target_w,
                args_cli.insertion_force,
                args_cli.insertion_torque,
            )
            current_pose_w = _object_pose_w(obj).clone()
            measured_poses.append(current_pose_w[0])
            print(
                f"OBJECT_POSE_W[{sample_index}]: {current_pose_w[0].tolist()}",
                flush=True,
            )
            geometric_yaw = physics_verifier.accumulate_geometric_world_yaw(
                geometric_yaw, previous_quat_w, current_pose_w[:, 3:7]
            )
            previous_quat_w = current_pose_w[:, 3:7].clone()
            force_saturation_count += int(command.force_saturated.item())
            torque_saturation_count += int(command.torque_saturated.item())
            peak_force = max(peak_force, float(torch.linalg.vector_norm(command.force_w).item()))
            peak_torque = max(peak_torque, float(torch.linalg.vector_norm(command.torque_w).item()))

        screw_measured_w = torch.stack(measured_poses)
        settle_poses = []
        settle_linear_speeds = []
        settle_angular_speeds = []
        for _ in range(settle_steps):
            obj.set_external_force_and_torque(zero_wrench, zero_wrench, is_global=True)
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(DT)
            settle_poses.append(_object_pose_w(obj)[0].clone())
            settle_linear_speeds.append(torch.linalg.vector_norm(obj.data.root_lin_vel_w[0]))
            settle_angular_speeds.append(torch.linalg.vector_norm(obj.data.root_ang_vel_w[0]))

        stable_window = torch.stack(settle_poses[-stable_window_steps:])
        stable_pose_w = _mean_pose(stable_window)
        stable_pose_r = _poses_in_receptive_frame(stable_pose_w[None], receptive)[0]
        screw_measured_r = _poses_in_receptive_frame(screw_measured_w, receptive)
        stable_linear_speed = torch.stack(settle_linear_speeds[-stable_window_steps:]).max()
        stable_angular_speed = torch.stack(settle_angular_speeds[-stable_window_steps:]).max()
        stable_position_drift = torch.linalg.vector_norm(
            stable_window[:, :3] - stable_pose_w[:3], dim=-1
        ).max()
        stable_orientation_drift = quat_error_magnitude(
            stable_window[:, 3:7], stable_pose_w[None, 3:7].expand(stable_window.shape[0], -1)
        ).max()
        installed_target_w = screw_targets_w[-1]
        installed_position_error = torch.linalg.vector_norm(
            stable_pose_w[:3] - installed_target_w[:3]
        )
        installed_orientation_error = quat_error_magnitude(
            stable_pose_w[None, 3:7], installed_target_w[None, 3:7]
        )[0]

        finite = bool(
            torch.isfinite(screw_measured_w).all()
            and torch.isfinite(stable_pose_w).all()
            and torch.isfinite(stable_pose_r).all()
        )
        checks = {
            "finite": finite,
            "screw_rotation": abs(
                float(geometric_yaw.item()) + CALIBRATION_TWIST_TOTAL_ANGLE
            )
            <= YAW_TOLERANCE,
            "installed_position": float(installed_position_error) <= POSITION_TOLERANCE,
            "installed_orientation": float(installed_orientation_error) <= ORIENTATION_TOLERANCE,
            "stable_linear_speed": float(stable_linear_speed) <= LINEAR_SPEED_TOLERANCE,
            "stable_angular_speed": float(stable_angular_speed) <= ANGULAR_SPEED_TOLERANCE,
            "stable_position_drift": float(stable_position_drift)
            <= STABLE_POSITION_DRIFT_TOLERANCE,
            "stable_orientation_drift": float(stable_orientation_drift)
            <= STABLE_ORIENTATION_DRIFT_TOLERANCE,
        }
        print("SCREW_IN_CALIBRATION:", flush=True)
        print(f"  checks: {checks}", flush=True)
        print(f"  measured_screw_rotation: {float(geometric_yaw.item()):.9f} rad", flush=True)
        print(f"  installed_position_error: {float(installed_position_error):.9f} m", flush=True)
        print(f"  installed_orientation_error: {float(installed_orientation_error):.9f} rad", flush=True)
        print(f"  stable_linear_speed: {float(stable_linear_speed):.9f} m/s", flush=True)
        print(f"  stable_angular_speed: {float(stable_angular_speed):.9f} rad/s", flush=True)
        print(f"  stable_position_drift: {float(stable_position_drift):.9f} m", flush=True)
        print(f"  stable_orientation_drift: {float(stable_orientation_drift):.9f} rad", flush=True)
        print(f"  peak_force: {peak_force:.9f} N", flush=True)
        print(f"  peak_torque: {peak_torque:.9f} Nm", flush=True)
        print(
            f"  force_saturation_fraction: {force_saturation_count / screw_targets_w.shape[0]:.9f}",
            flush=True,
        )
        print(
            f"  torque_saturation_fraction: {torque_saturation_count / screw_targets_w.shape[0]:.9f}",
            flush=True,
        )
        print(f"  stable_pose_w: {stable_pose_w.tolist()}", flush=True)
        print(f"  stable_pose_receptive: {stable_pose_r.tolist()}", flush=True)
        if not all(checks.values()):
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"Physical screw-in calibration failed checks: {failed}")

        recording = build_screw_in_recording(
            stable_pose_w=stable_pose_w.cpu().numpy(),
            stable_pose_receptive=stable_pose_r.cpu().numpy(),
            screw_targets_w=screw_targets_w.cpu().numpy(),
            screw_measured_w=screw_measured_w.cpu().numpy(),
            screw_measured_receptive=screw_measured_r.cpu().numpy(),
            dt=DT,
            thread_pitch=CALIBRATION_THREAD_PITCH,
            commanded_screw_rotation=-CALIBRATION_TWIST_TOTAL_ANGLE,
            measured_screw_rotation=float(geometric_yaw.item()),
            asset_scale=task_env_cfg.ASSET_SCALE,
        )
        sample_count = screw_targets_w.shape[0]
        recording.update(
            {
                "stable_linear_speed": np.asarray(float(stable_linear_speed)),
                "stable_angular_speed": np.asarray(float(stable_angular_speed)),
                "stable_position_drift": np.asarray(float(stable_position_drift)),
                "stable_orientation_drift": np.asarray(float(stable_orientation_drift)),
                "peak_force": np.asarray(peak_force),
                "peak_torque": np.asarray(peak_torque),
                "insertion_force": np.asarray(args_cli.insertion_force),
                "insertion_torque": np.asarray(args_cli.insertion_torque),
                "max_force": np.asarray(args_cli.max_force),
                "max_torque": np.asarray(args_cli.max_torque),
                "force_saturation_fraction": np.asarray(force_saturation_count / sample_count),
                "torque_saturation_fraction": np.asarray(torque_saturation_count / sample_count),
            }
        )
        save_screw_in_recording(args_cli.output, recording)
        print(f"  output: {args_cli.output.resolve()}", flush=True)
    finally:
        if obj is not None:
            zero_wrench = torch.zeros(1, 1, 3, device=obj.device)
            obj.set_external_force_and_torque(zero_wrench, zero_wrench, is_global=True)
            if scene is not None:
                scene.write_data_to_sim()
        obj = None
        scene = None
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()


def _run_main_and_close() -> None:
    status = 0
    try:
        main()
    except BaseException:
        status = 1
        traceback.print_exc()
    finally:
        simulation_app.app.post_uncancellable_quit(status)
        simulation_app.close()
    raise SystemExit(status)


if __name__ == "__main__":
    _run_main_and_close()
else:
    simulation_app.close()
