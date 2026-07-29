"""Headless multi-environment stability check for the installed unscrew reset."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Validate that threaded unscrew resets remain assembled.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--duration", type=float, default=2.0, help="Simulated seconds to settle.")
parser.add_argument("--position_tolerance", type=float, default=0.002)
parser.add_argument("--orientation_tolerance_deg", type=float, default=2.0)
parser.add_argument("--linear_speed_tolerance", type=float, default=0.01)
parser.add_argument("--angular_speed_tolerance", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import torch  # noqa: E402
import src.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import quat_error_magnitude, subtract_frame_transforms  # noqa: E402


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("num_envs must be at least 1.")
    if args_cli.duration <= 0.0:
        raise ValueError("duration must be positive.")

    env_cfg = parse_env_cfg("Unscrew-v0", device=args_cli.device, num_envs=args_cli.num_envs)
    # Exercise the same reduced-gravity load as the frozen-policy HRL variant without constructing
    # cuRobo or moving the robot into contact with the object.
    env_cfg.events.variable_gravity.params["gravity_distribution_params"] = (
        [0.0, 0.0, -1.81],
        [0.0, 0.0, -1.81],
    )
    env = gym.make("Unscrew-v0", cfg=env_cfg)
    unwrapped = env.unwrapped
    try:
        env.reset()
        actions = torch.zeros(
            unwrapped.num_envs,
            unwrapped.action_manager.total_action_dim,
            device=unwrapped.device,
        )
        num_steps = math.ceil(args_cli.duration / unwrapped.step_dt)
        finite = torch.ones(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
        ever_done = torch.zeros_like(finite)
        max_position_drift = torch.zeros(unwrapped.num_envs, device=unwrapped.device)
        max_orientation_drift = torch.zeros_like(max_position_drift)
        max_linear_speed = torch.zeros_like(max_position_drift)
        max_angular_speed = torch.zeros_like(max_position_drift)
        expected_pos = actions.new_tensor((-0.084375, 0.084375, 0.070834)).expand(
            unwrapped.num_envs, -1
        )
        expected_quat = actions.new_tensor((1.0, 0.0, 0.0, 0.0)).expand(unwrapped.num_envs, -1)
        for _ in range(num_steps):
            _, _, terminated, truncated, _ = env.step(actions)
            ever_done |= terminated | truncated
            obj = unwrapped.scene["object"]
            receptive = unwrapped.scene["receptive_object"]
            finite &= torch.isfinite(obj.data.root_state_w).all(dim=1)
            relative_pos, relative_quat = subtract_frame_transforms(
                receptive.data.root_pos_w,
                receptive.data.root_quat_w,
                obj.data.root_pos_w,
                obj.data.root_quat_w,
            )
            max_position_drift = torch.maximum(
                max_position_drift, torch.abs(relative_pos - expected_pos).amax(dim=1)
            )
            max_orientation_drift = torch.maximum(
                max_orientation_drift, quat_error_magnitude(relative_quat, expected_quat)
            )
            max_linear_speed = torch.maximum(
                max_linear_speed, torch.linalg.vector_norm(obj.data.root_lin_vel_w, dim=1)
            )
            max_angular_speed = torch.maximum(
                max_angular_speed, torch.linalg.vector_norm(obj.data.root_ang_vel_w, dim=1)
            )

        checks = {
            "finite": finite,
            "no_termination_or_truncation": ~ever_done,
            "position": max_position_drift <= args_cli.position_tolerance,
            "orientation": max_orientation_drift <= math.radians(args_cli.orientation_tolerance_deg),
            "linear_speed": max_linear_speed <= args_cli.linear_speed_tolerance,
            "angular_speed": max_angular_speed <= args_cli.angular_speed_tolerance,
        }
        passed = torch.stack(tuple(checks.values())).all(dim=0)
        print(
            "[UNSCREW RESET] "
            f"passed={int(passed.sum())}/{unwrapped.num_envs} "
            f"max_pos_drift={float(max_position_drift.max()):.6f} m "
            f"max_rot_drift={math.degrees(float(max_orientation_drift.max())):.3f} deg "
            f"max_lin_speed={float(max_linear_speed.max()):.6f} m/s "
            f"max_ang_speed={float(max_angular_speed.max()):.6f} rad/s",
            flush=True,
        )
        if not bool(passed.all()):
            failed = (~passed).nonzero().flatten().tolist()
            failed_checks = {
                name: (~mask).nonzero().flatten().tolist() for name, mask in checks.items() if not bool(mask.all())
            }
            raise RuntimeError(f"Unscrew reset validation failed for envs {failed}: {failed_checks}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
