"""Physically verify the unscrew trajectory with a bounded virtual-grasp wrench."""

from __future__ import annotations

import argparse
import itertools
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


parser = argparse.ArgumentParser(description="Run the physical unscrew trajectory feasibility test.")
parser.add_argument("--settle_duration", type=float, default=1.0, help="Seconds to settle before tracking.")
parser.add_argument("--turn_duration", type=float, default=12.0, help="Seconds for all twelve twist segments.")
parser.add_argument("--extraction_duration", type=float, default=2.0, help="Seconds for vertical extraction.")
parser.add_argument("--hold_duration", type=float, default=1.0, help="Seconds to hold the cleared pose.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
for duration_name in ("settle_duration", "turn_duration", "extraction_duration", "hold_duration"):
    if getattr(args_cli, duration_name) <= 0.0:
        parser.error(f"--{duration_name} must be positive.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


try:
    import isaaclab.sim as sim_utils  # noqa: E402
    import torch  # noqa: E402
    from isaaclab.scene import InteractiveScene  # noqa: E402
    from isaaclab.sim import SimulationContext  # noqa: E402
    from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms  # noqa: E402

    from src.tasks.unscrew import env_cfg as task_env_cfg  # noqa: E402
    from src.tasks.unscrew.mdps import physics_verifier, trajectory  # noqa: E402
    from src.tasks.unscrew.mdps.task_mdps import BOLT_AABB_MAX, BOLT_AABB_MIN, SOCKET_TOP_Z  # noqa: E402
except BaseException:
    traceback.print_exc()
    simulation_app.app.post_uncancellable_quit(1)
    simulation_app.close()
    raise SystemExit(1)


DT = 1.0 / 120.0
NUM_TRIALS = 2
MAX_FORCE = 10.0
MAX_TORQUE = 0.2
POSITION_STIFFNESS = 200.0
POSITION_DAMPING = 4.0
ROTATION_STIFFNESS = 0.5
ROTATION_DAMPING = 0.01
CLEARANCE_MARGIN = 0.030
YAW_TOLERANCE = 0.35
POSITION_TOLERANCE = 0.005
ORIENTATION_TOLERANCE = 0.35
LATERAL_TOLERANCE = 0.005
SATURATION_INCONCLUSIVE_FRACTION = 0.90


def _step_count(duration: float, name: str) -> int:
    count = round(duration / DT)
    if count < 1:
        raise ValueError(f"{name} must span at least one physics step ({DT:.6f} s).")
    return count


def _make_scene_cfg():
    scene_cfg = task_env_cfg.SceneCfg(num_envs=2, env_spacing=3.0, replicate_physics=False)
    scene_cfg.robot = None
    return scene_cfg


def _object_pose_w(obj) -> torch.Tensor:
    return torch.cat((obj.data.root_pos_w, obj.data.root_quat_w), dim=-1)


def _bolt_corners(*, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lower = torch.tensor(BOLT_AABB_MIN, device=device, dtype=dtype)
    upper = torch.tensor(BOLT_AABB_MAX, device=device, dtype=dtype)
    return torch.stack(
        [
            torch.stack((x, y, z))
            for x, y, z in itertools.product(
                (lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2])
            )
        ]
    )


def _clearance(obj, receptive, bolt_corners: torch.Tensor) -> torch.Tensor:
    object_pos_r, object_quat_r = subtract_frame_transforms(
        receptive.data.root_pos_w,
        receptive.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )
    return physics_verifier.bolt_bottom_clearance(
        object_pos_r, object_quat_r, bolt_corners, SOCKET_TOP_Z
    )


def _summary_line(label: str, summary: physics_verifier.TrialSummary) -> None:
    print(f"{label}:", flush=True)
    for field_name, value in vars(summary).items():
        print(f"  {field_name}: {value}", flush=True)


def main() -> None:
    settle_steps = _step_count(args_cli.settle_duration, "settle duration")
    turn_steps_per_segment = _step_count(
        args_cli.turn_duration / trajectory.DEFAULT_UNSCREW_TWIST_SEGMENTS,
        "turn segment duration",
    )
    extract_steps = _step_count(args_cli.extraction_duration, "extraction duration")
    hold_steps = _step_count(args_cli.hold_duration, "hold duration")
    segment_steps = (
        (0, 0)
        + (turn_steps_per_segment,) * trajectory.DEFAULT_UNSCREW_TWIST_SEGMENTS
        + (extract_steps, hold_steps)
    )
    twist_start_index = 1 + sum(segment_steps[:2])
    twist_end_index = sum(
        segment_steps[: 2 + trajectory.DEFAULT_UNSCREW_TWIST_SEGMENTS]
    )
    extraction_start_index = twist_end_index + 1
    expected_twist_rise = (
        trajectory.DEFAULT_UNSCREW_THREAD_PITCH
        * trajectory.DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE
        / (2.0 * torch.pi)
    )

    print(
        "CONFIG: "
        f"dt={DT:.6f}s, settle={settle_steps} steps, "
        f"turn={turn_steps_per_segment} steps/segment, extract={extract_steps} steps, "
        f"hold={hold_steps} steps, expected_yaw={trajectory.DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE}rad, "
        f"expected_twist_rise={expected_twist_rise}m, "
        f"thread_pitch={trajectory.DEFAULT_UNSCREW_THREAD_PITCH}m, clearance={CLEARANCE_MARGIN}m",
        flush=True,
    )
    print(
        "CONTROLLER: "
        f"max_force={MAX_FORCE}N, max_torque={MAX_TORQUE}Nm, "
        f"position_kp={POSITION_STIFFNESS}, position_kd={POSITION_DAMPING}, "
        f"rotation_kp={ROTATION_STIFFNESS}, rotation_kd={ROTATION_DAMPING}",
        flush=True,
    )
    print(
        "CLASSIFICATION_TOLERANCES: "
        f"yaw={YAW_TOLERANCE}rad, position={POSITION_TOLERANCE}m, "
        f"orientation={ORIENTATION_TOLERANCE}rad, lateral={LATERAL_TOLERANCE}m, "
        f"saturation_fraction={SATURATION_INCONCLUSIVE_FRACTION}",
        flush=True,
    )

    scene = None
    obj = None
    receptive = None
    sim = None
    try:
        sim_cfg = sim_utils.SimulationCfg(
            dt=DT, gravity=(0.0, 0.0, -9.81), device=args_cli.device
        )
        sim = SimulationContext(sim_cfg)
        scene = InteractiveScene(_make_scene_cfg())
        sim.reset()
        scene.reset()
        scene.update(DT)

        obj = scene["object"]
        receptive = scene["receptive_object"]
        zero_wrench = torch.zeros(NUM_TRIALS, 1, 3, device=sim.device)
        for _ in range(settle_steps):
            obj.set_external_force_and_torque(zero_wrench, zero_wrench, is_global=True)
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(DT)

        settled_pose_w = _object_pose_w(obj).clone()
        targets_w = torch.stack(
            [
                trajectory.build_unscrew_object_pose_sequence(
                    settled_pose_w[env_index], segment_steps=segment_steps
                )
                for env_index in range(NUM_TRIALS)
            ]
        )
        targets_w[1] = physics_verifier.straight_pull_targets(targets_w[1])

        masses = obj.root_physx_view.get_masses().to(device=settled_pose_w.device)[:, 0]
        print(f"OBJECT_MASS_KG: {masses.tolist()}", flush=True)
        gravity_w = settled_pose_w.new_tensor((0.0, 0.0, -9.81))
        bolt_corners = _bolt_corners(device=settled_pose_w.device, dtype=settled_pose_w.dtype)

        geometric_yaw = torch.zeros(NUM_TRIALS, device=settled_pose_w.device)
        velocity_integrated_yaw = torch.zeros_like(geometric_yaw)
        previous_twist_quat_w = settled_pose_w[:, 3:7].clone()
        twist_end_rise = torch.full_like(geometric_yaw, float("nan"))
        max_twist_position_error = torch.zeros_like(geometric_yaw)
        max_twist_orientation_error = torch.zeros_like(geometric_yaw)
        max_lateral_drift = torch.zeros_like(geometric_yaw)
        force_saturation_count = torch.zeros(NUM_TRIALS, dtype=torch.long, device=settled_pose_w.device)
        torque_saturation_count = torch.zeros_like(force_saturation_count)
        peak_force = torch.zeros_like(geometric_yaw)
        peak_torque = torch.zeros_like(geometric_yaw)
        held_clear_count = torch.zeros_like(force_saturation_count)
        ever_cleared = torch.zeros(NUM_TRIALS, dtype=torch.bool, device=settled_pose_w.device)
        max_clearance = torch.full_like(geometric_yaw, -torch.inf)
        finite = torch.ones(NUM_TRIALS, dtype=torch.bool, device=settled_pose_w.device)
        final_clearance = torch.full_like(geometric_yaw, float("nan"))
        hold_start = targets_w.shape[1] - hold_steps

        for target_index in range(targets_w.shape[1]):
            current_pose_w = _object_pose_w(obj)
            linear_velocity_w = obj.data.root_lin_vel_w
            angular_velocity_w = obj.data.root_ang_vel_w
            command = physics_verifier.bounded_pd_wrench(
                current_pose_w,
                targets_w[:, target_index],
                linear_velocity_w,
                angular_velocity_w,
                masses,
                gravity_w,
                position_stiffness=POSITION_STIFFNESS,
                position_damping=POSITION_DAMPING,
                rotation_stiffness=ROTATION_STIFFNESS,
                rotation_damping=ROTATION_DAMPING,
                max_force=MAX_FORCE,
                max_torque=MAX_TORQUE,
            )
            obj.set_external_force_and_torque(
                command.force_w[:, None, :], command.torque_w[:, None, :], is_global=True
            )
            scene.write_data_to_sim()
            sim.step(render=not args_cli.headless)
            scene.update(DT)

            current_pose_w = _object_pose_w(obj)
            lateral_drift = torch.linalg.vector_norm(
                current_pose_w[:, :2] - settled_pose_w[:, :2], dim=-1
            )
            max_lateral_drift = torch.maximum(max_lateral_drift, lateral_drift)
            force_saturation_count += command.force_saturated.long()
            torque_saturation_count += command.torque_saturated.long()
            peak_force = torch.maximum(peak_force, torch.linalg.vector_norm(command.force_w, dim=-1))
            peak_torque = torch.maximum(peak_torque, torch.linalg.vector_norm(command.torque_w, dim=-1))
            final_clearance = _clearance(obj, receptive, bolt_corners)
            finite &= (
                torch.isfinite(current_pose_w).all(dim=-1)
                & torch.isfinite(obj.data.root_lin_vel_w).all(dim=-1)
                & torch.isfinite(obj.data.root_ang_vel_w).all(dim=-1)
                & torch.isfinite(command.force_w).all(dim=-1)
                & torch.isfinite(command.torque_w).all(dim=-1)
                & torch.isfinite(final_clearance)
            )
            if target_index < twist_start_index:
                previous_twist_quat_w = current_pose_w[:, 3:7].clone()
            elif target_index <= twist_end_index:
                geometric_yaw = physics_verifier.accumulate_geometric_world_yaw(
                    geometric_yaw, previous_twist_quat_w, current_pose_w[:, 3:7]
                )
                velocity_integrated_yaw = (
                    physics_verifier.accumulate_velocity_integrated_world_yaw(
                        velocity_integrated_yaw, obj.data.root_ang_vel_w, DT
                    )
                )
                previous_twist_quat_w = current_pose_w[:, 3:7].clone()
                twist_position_error = torch.linalg.vector_norm(
                    targets_w[:, target_index, :3] - current_pose_w[:, :3], dim=-1
                )
                twist_orientation_error = quat_error_magnitude(
                    targets_w[:, target_index, 3:7], current_pose_w[:, 3:7]
                )
                max_twist_position_error = torch.maximum(
                    max_twist_position_error, twist_position_error
                )
                max_twist_orientation_error = torch.maximum(
                    max_twist_orientation_error, twist_orientation_error
                )
                if target_index == twist_end_index:
                    twist_end_rise = current_pose_w[:, 2] - settled_pose_w[:, 2]
            if target_index >= extraction_start_index:
                clear = final_clearance >= CLEARANCE_MARGIN
                ever_cleared |= clear
                max_clearance = torch.maximum(max_clearance, final_clearance)
            if target_index >= hold_start:
                clear = final_clearance >= CLEARANCE_MARGIN
                held_clear_count = torch.where(
                    clear, held_clear_count + 1, torch.zeros_like(held_clear_count)
                )

        final_pose_w = _object_pose_w(obj)
        final_position_error = torch.linalg.vector_norm(
            targets_w[:, -1, :3] - final_pose_w[:, :3], dim=-1
        )
        final_orientation_error = quat_error_magnitude(
            targets_w[:, -1, 3:7], final_pose_w[:, 3:7]
        )
        finite &= (
            torch.isfinite(final_position_error)
            & torch.isfinite(final_orientation_error)
            & torch.isfinite(geometric_yaw)
            & torch.isfinite(velocity_integrated_yaw)
            & torch.isfinite(twist_end_rise)
            & torch.isfinite(max_twist_position_error)
            & torch.isfinite(max_twist_orientation_error)
            & torch.isfinite(max_clearance)
            & torch.isfinite(max_lateral_drift)
            & torch.isfinite(peak_force)
            & torch.isfinite(peak_torque)
        )

        sample_count = targets_w.shape[1]
        summaries = [
            physics_verifier.TrialSummary(
                finite=bool(finite[index].item()),
                held_clear=bool((held_clear_count[index] >= hold_steps).item()),
                geometric_yaw=float(geometric_yaw[index].item()),
                velocity_integrated_yaw=float(velocity_integrated_yaw[index].item()),
                twist_end_rise=float(twist_end_rise[index].item()),
                max_twist_position_error=float(max_twist_position_error[index].item()),
                max_twist_orientation_error=float(max_twist_orientation_error[index].item()),
                ever_cleared=bool(ever_cleared[index].item()),
                max_clearance=float(max_clearance[index].item()),
                final_position_error=float(final_position_error[index].item()),
                final_orientation_error=float(final_orientation_error[index].item()),
                max_lateral_drift=float(max_lateral_drift[index].item()),
                force_saturation_fraction=float(force_saturation_count[index].item() / sample_count),
                torque_saturation_fraction=float(torque_saturation_count[index].item() / sample_count),
                peak_force=float(peak_force[index].item()),
                peak_torque=float(peak_torque[index].item()),
                final_clearance=float(final_clearance[index].item()),
            )
            for index in range(NUM_TRIALS)
        ]
        _summary_line("HELIX", summaries[0])
        _summary_line("STRAIGHT_PULL", summaries[1])
        result = physics_verifier.classify_verification(
            summaries[0],
            summaries[1],
            expected_yaw=trajectory.DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE,
            expected_twist_rise=expected_twist_rise,
            yaw_tolerance=YAW_TOLERANCE,
            position_tolerance=POSITION_TOLERANCE,
            orientation_tolerance=ORIENTATION_TOLERANCE,
            lateral_tolerance=LATERAL_TOLERANCE,
            saturation_inconclusive_fraction=SATURATION_INCONCLUSIVE_FRACTION,
        )
        print(f"CLASSIFICATION: {result.value}", flush=True)
        if result is not physics_verifier.VerificationResult.PASS:
            raise RuntimeError(f"Unscrew trajectory verification result: {result.value}")
    finally:
        if obj is not None:
            zero_wrench = torch.zeros(NUM_TRIALS, 1, 3, device=obj.device)
            obj.set_external_force_and_torque(zero_wrench, zero_wrench, is_global=True)
            if scene is not None:
                scene.write_data_to_sim()
        obj = None
        receptive = None
        scene = None
        if sim is not None:
            print("CLEANUP: clearing simulation callbacks", flush=True)
            sim.clear_all_callbacks()
            print("CLEANUP: clearing simulation context", flush=True)
            sim.clear_instance()
            print("CLEANUP: simulation context cleared", flush=True)


def _run_main_and_close(main_fn, simulation_app) -> None:
    status = 0
    try:
        main_fn()
    except BaseException:
        status = 1
        print("[ERROR]: verify_unscrew_trajectory_physics.py failed.", flush=True)
        traceback.print_exc()
    finally:
        # Fast shutdown may terminate before post-close Python can raise, so post status to IApp first.
        simulation_app.app.post_uncancellable_quit(status)
        print("CLEANUP: closing simulation app", flush=True)
        simulation_app.close()
    raise SystemExit(status)


if __name__ == "__main__":
    _run_main_and_close(main, simulation_app)
else:
    simulation_app.close()
