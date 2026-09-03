#!/usr/bin/env python3
"""Run an exported distilled knob policy on a LEAP hand."""

from __future__ import annotations

import argparse
import hashlib
import math
import pathlib
import sys
import time

import numpy as np
import torch

from src.policy.knob_interface import (
    ACTION_ALPHA,
    CONTROL_HZ,
    JOINT_LOWER,
    JOINT_UPPER,
    append_history,
    build_aria_frame,
    ema_absolute_targets,
    initialize_history,
    scale_absolute_actions,
)


CHECKPOINT_INIT = torch.zeros(16)


class MockHand:
    def __init__(self):
        self.position = CHECKPOINT_INIT.clone()

    def connect(self):
        pass

    def interp_to(self, target, duration=0.0):
        self.position = target.flatten().clone()

    def poll_joint_position(self):
        return self.position.clone()

    def command_joint_position(self, target):
        self.position = target.flatten().clone()

    def disconnect(self):
        pass


class MockKnob:
    def connect(self):
        pass

    def read_position(self):
        return 0.0, 0.0

    def disconnect(self):
        pass


class DynamixelKnob:
    """Read motor 50 and unwrap it around the startup zero."""

    def __init__(self, client_type, port: str):
        self.client = client_type([50], port, 4_000_000)
        self.previous_raw = None
        self.position = 0.0
        self.zero = None
        self.previous_time = None

    def connect(self):
        self.client.connect()

    def read_position(self):
        raw = float(np.asarray(self.client.read_pos()).flat[0])
        now = time.monotonic()
        if self.previous_raw is None:
            self.position = raw
            velocity = 0.0
            self.zero = raw
        else:
            delta = (raw - self.previous_raw + math.pi) % (2.0 * math.pi) - math.pi
            dt = max(now - self.previous_time, 1.0e-6)
            self.position += delta
            velocity = delta / dt
        self.previous_raw = raw
        self.previous_time = now
        return self.position - self.zero, velocity

    def disconnect(self):
        self.client.disconnect()


def load_hardware(aria_source: pathlib.Path, hand_port: str | None, knob_port: str | None):
    if not aria_source.is_dir():
        raise FileNotFoundError(f"ARIA source directory not found: {aria_source}")
    sys.path[:0] = [str(aria_source), str(aria_source / "deployment_scripts")]
    from leap_hand_hw import LeapHandHW, detect_ports
    from utils.leap_hand_utils.dynamixel_client import DynamixelClient

    hand_port, knob_port = detect_ports(hand_override=hand_port, knob_override=knob_port)
    if hand_port is None or knob_port is None:
        raise RuntimeError("Could not detect both hand and knob ports; pass --hand-port and --knob-port.")
    return LeapHandHW(port=hand_port, device="cpu", hz=CONTROL_HZ), DynamixelKnob(DynamixelClient, knob_port)


def file_sha256(path: pathlib.Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def policy_action(policy, observation: torch.Tensor) -> torch.Tensor:
    action = policy(observation)
    if isinstance(action, (tuple, list)):
        action = action[0]
    action = torch.as_tensor(action).reshape(-1)
    if action.numel() != 16 or not torch.isfinite(action).all():
        raise RuntimeError(f"Policy must return 16 finite actions, got shape {tuple(action.shape)}.")
    return action


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=pathlib.Path, help="Exported JIT student policy.pt")
    parser.add_argument("--target", type=float, required=True, help="Signed target angle in radians")
    parser.add_argument("--execute", action="store_true", help="Enable physical motor commands; default is dry-run")
    parser.add_argument("--aria-source", type=pathlib.Path)
    parser.add_argument("--hand-port")
    parser.add_argument("--knob-port")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--setup-seconds", type=float, default=3.0)
    parser.add_argument("--max-step", type=float, default=0.12, help="Per-cycle target slew limit in radians")
    parser.add_argument("--policy-sha256")
    args = parser.parse_args()
    if not args.policy.is_file():
        parser.error(f"policy does not exist: {args.policy}")
    if not -math.pi <= args.target <= math.pi:
        parser.error("--target must be within [-pi, pi]")
    if args.duration <= 0.0 or args.setup_seconds < 0.0 or args.max_step <= 0.0:
        parser.error("duration/max-step must be positive and setup-seconds non-negative")
    if args.execute and args.aria_source is None:
        parser.error("--execute requires --aria-source pointing to ARIA/source/ARIA")
    if args.policy_sha256 and file_sha256(args.policy) != args.policy_sha256.lower():
        parser.error("policy SHA-256 does not match --policy-sha256")
    return args


def main():
    args = parse_args()
    policy = torch.jit.load(str(args.policy), map_location="cpu").eval()
    hand, knob = (
        load_hardware(args.aria_source, args.hand_port, args.knob_port)
        if args.execute
        else (MockHand(), MockKnob())
    )
    lower = torch.tensor(JOINT_LOWER)
    upper = torch.tensor(JOINT_UPPER)
    period = 1.0 / CONTROL_HZ

    try:
        hand.connect()
        knob.connect()
        hand.interp_to(CHECKPOINT_INIT, duration=args.setup_seconds)
        joint_pos = hand.poll_joint_position().float()
        applied_target = joint_pos.clone()
        angle, velocity = knob.read_position()
        frame = build_aria_frame(
            joint_pos[None], applied_target[None], torch.tensor([angle]),
            torch.tensor([velocity]), torch.tensor([args.target]), lower[None], upper[None],
        )
        history = initialize_history(frame)
        deadline = time.monotonic() + args.duration

        with torch.inference_mode():
            while time.monotonic() < deadline:
                started = time.monotonic()
                action = policy_action(policy, history)
                scaled = scale_absolute_actions(action, lower, upper)
                target = ema_absolute_targets(scaled, applied_target, ACTION_ALPHA)
                target = torch.maximum(torch.minimum(target, joint_pos + args.max_step), joint_pos - args.max_step)
                applied_target = torch.maximum(torch.minimum(target, upper), lower)
                hand.command_joint_position(applied_target)

                joint_pos = hand.poll_joint_position().float()
                angle, velocity = knob.read_position()
                frame = build_aria_frame(
                    joint_pos[None], applied_target[None], torch.tensor([angle]),
                    torch.tensor([velocity]), torch.tensor([args.target]), lower[None], upper[None],
                )
                history = append_history(history, frame)
                if abs((args.target - angle + math.pi) % (2.0 * math.pi) - math.pi) <= 0.02 and abs(velocity) <= 0.2:
                    print(f"success: angle={angle:.4f}, velocity={velocity:.4f}")
                    break
                time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        knob.disconnect()
        hand.disconnect()


if __name__ == "__main__":
    main()
