#!/usr/bin/env python3
"""Deploy a D2H knob student with ARIA-style controls, evaluation, and a simulation mirror."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import math
from pathlib import Path
import random
import select
import sys
import termios
import time
import tty

# Support both direct execution from any directory and module imports in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import yaml

from scripts.deployment.hardware import DynamixelKnob, LeapHand, MockHand, MockKnob, detect_ports, wrap
from src.policy.knob_interface import (
    CONTROL_HZ, HISTORY_DIM, JOINT_LOWER, JOINT_UPPER, append_history,
    build_aria_frame, ema_absolute_targets, initialize_history, scale_absolute_actions,
)
from src.policy.low_level.policy import load_low_level_rsl_rl_policy

CHECKPOINT_INIT = torch.zeros(16)
LOWER, UPPER = torch.tensor(JOINT_LOWER), torch.tensor(JOINT_UPPER)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_action(policy, observation):
    if observation.shape != (1, HISTORY_DIM) or not torch.isfinite(observation).all():
        raise ValueError("Policy observation must be finite with shape (1, 105)")
    action = torch.as_tensor(policy.act(observation)).flatten()
    if action.shape != (16,) or not torch.isfinite(action).all():
        raise ValueError("Policy must return 16 finite actions")
    return action


def observation_frame(q, applied, angle, velocity, target):
    if q.shape != (16,) or not torch.isfinite(q).all() or not all(math.isfinite(v) for v in (angle, velocity, target)):
        raise ValueError("Invalid joint or knob sensor reading")
    return build_aria_frame(q[None], applied[None], torch.tensor([angle]), torch.tensor([velocity]),
                            torch.tensor([target]), LOWER[None], UPPER[None]).float()


def applied_targets(action, previous, q, max_step):
    target = ema_absolute_targets(scale_absolute_actions(action, LOWER, UPPER), previous)
    return limit_targets(target, q, max_step)


def limit_targets(target, q, max_step):
    lower = torch.maximum(LOWER, q - max_step)
    upper = torch.minimum(UPPER, q + max_step)
    if torch.any(lower > upper):
        raise ValueError("Measured joints are too far outside limits; check hand calibration")
    return target.clamp(lower, upper)


def load_episodes(path):
    if path is None:
        return []
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list) or not data["episodes"]:
        raise ValueError("Evaluation YAML must contain a nonempty episodes list")
    episodes = []
    for entry in data["episodes"]:
        if not isinstance(entry, dict):
            raise ValueError("Each evaluation episode must contain knob_start and target")
        values = [entry.get(key) for key in ("knob_start", "target")]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Evaluation knob_start and target must be finite numbers in [0, 1]")
        episodes.append(tuple(wrap((v * 2 - 1) * math.pi) for v in values))
    return episodes


@dataclass
class DeployState:
    phase: str = "paused"
    running: bool = True
    elapsed: float = 0.0
    episode: int = 0
    steps: int = 0
    hz: float = 0.0
    target: float = 0.0
    angle: float = 0.0
    velocity: float = 0.0
    q: torch.Tensor = field(default_factory=lambda: torch.zeros(16))
    applied: torch.Tensor = field(default_factory=lambda: torch.zeros(16))
    sim_q: torch.Tensor | None = None
    sim_angle: float = 0.0
    sim_velocity: float = 0.0
    records: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def log(self, text):
        self.events = (self.events + [text])[-6:]


def statistics(records, duration):
    successes = [r for r in records if r["success"]]
    failures = [r["dist_to_target"] for r in records if not r["success"]]
    left = [max(0.0, duration - r["elapsed"]) for r in successes]
    def mean_std(values):
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    mean, std = mean_std(left)
    dist_mean, dist_std = mean_std(failures)
    sr = len(successes) / len(records) if records else 0.0
    return dict(n_episodes=len(records), success_rate=sr, sec_left_mean=mean, sec_left_std=std,
                dist_to_target_mean=dist_mean, dist_to_target_std=dist_std,
                fitness=sr * sum(left) / len(records) if records else 0.0)


class KnobDeployer:
    def __init__(self, args, policy, hand, knob, sim=None, leds=None, app=None):
        self.args, self.policy, self.hand, self.knob = args, policy, hand, knob
        self.sim, self.leds, self.app = sim, leds, app
        self.state = DeployState()
        self.episodes = load_episodes(args.eval)
        self.eval_index = 0
        self.history = self.sim_history = None
        self.setup_angle = None
        self.reset_elapsed = 0.0
        self.rng = random.Random(args.seed)

    def read(self):
        s = self.state
        s.q = self.hand.poll_joint_position().float()
        s.angle, s.velocity = self.knob.read_position()
        observation_frame(s.q, s.applied, s.angle, s.velocity, s.target)
        if self.sim is not None:
            s.sim_q, s.sim_angle, s.sim_velocity = self.sim.read()

    def refill_history(self):
        s = self.state
        self.history = initialize_history(observation_frame(s.q, s.applied, s.angle, s.velocity, s.target))
        if self.sim is not None:
            self.sim_history = initialize_history(observation_frame(s.sim_q, s.applied, s.sim_angle, s.sim_velocity, s.target))

    def start_episode(self, new=True):
        s = self.state
        self.read()
        if new:
            s.episode += 1
            if self.episodes:
                self.setup_angle, s.target = self.episodes[self.eval_index]
            else:
                angle = s.sim_angle if self.sim is not None and not self.args.execute else s.angle
                s.target = self.args.target if self.args.target is not None else wrap(angle + self.rng.choice((-1, 1)) * self.rng.uniform(math.pi / 3, math.pi / 2))
        self.reset_knob = (s.sim_angle, s.sim_velocity) if self.sim is not None and not self.args.execute else None
        self.reset_origin = s.q.clone()
        self.reset_elapsed = 0.0
        s.elapsed, s.steps = 0.0, 0
        s.phase = "resetting"
        s.log(f"Episode {s.episode}: target {math.degrees(s.target):+.1f}°")
        if self.setup_angle is not None:
            s.log(f"Setup: position knob at {math.degrees(self.setup_angle):+.1f}° and press Space")

    def finish_reset(self):
        s = self.state
        self.read()
        s.applied = s.q.clone()
        if not self.args.execute and self.setup_angle is not None:
            self.knob.angle = self.setup_angle
            self.read()
        if self.sim is not None:
            angle, velocity = self.reset_knob if self.reset_knob is not None else (s.angle, s.velocity)
            if not self.args.execute and self.setup_angle is not None:
                angle, velocity = self.setup_angle, 0.0
            self.sim.reset(s.q, angle, s.target, velocity)
            self.read()
        self.refill_history()
        automatic = self.args.non_interactive or (s.episode > 1 and (self.episodes or self.args.auto_reset))
        s.phase = "setup" if self.episodes and self.args.execute else ("running" if automatic else "paused")

    def key(self, key):
        s = self.state
        if key.lower() == "q" or key == "\x03":
            s.running = False
        elif key.lower() == "r":
            self.start_episode(new=False)
        elif key.lower() == "n":
            self.advance()
        elif key == " ":
            if s.phase == "setup":
                if abs(wrap(s.angle - self.setup_angle)) > 0.02:
                    s.log("Position knob within 0.02 rad of the setup angle before confirming")
                    return
                if self.sim is not None:
                    self.sim.reset(s.q, s.angle, s.target, s.velocity)
                    self.read()
                self.refill_history()
                s.phase = "running"
            elif s.phase in ("running", "paused"):
                if s.phase == "paused":
                    self.read()
                    self.refill_history()
                s.phase = "paused" if s.phase == "running" else "running"

    def advance(self):
        if self.episodes:
            self.eval_index += 1
            if self.eval_index >= len(self.episodes):
                self.state.running = False
                return
        self.start_episode()

    def complete(self, success):
        s = self.state
        if s.phase != "running":
            return
        angle = s.sim_angle if self.sim is not None and not self.args.execute else s.angle
        s.records.append(dict(episode=s.episode, success=success, elapsed=s.elapsed,
                              dist_to_target=abs(wrap(angle - s.target)) / math.pi, target=s.target,
                              knob_start=self.setup_angle))
        s.log("Success" if success else "Timeout")
        s.phase = "completed"
        if self.episodes or self.args.auto_reset:
            self.advance()
        elif self.args.non_interactive:
            s.running = False

    def tick(self, dt):
        s = self.state
        if s.phase == "resetting":
            self.read()
            self.reset_elapsed += dt
            fraction = min(1.0, self.reset_elapsed / self.args.setup_seconds) if self.args.setup_seconds else 1.0
            target = torch.lerp(self.reset_origin, CHECKPOINT_INIT, fraction)
            target = limit_targets(target, s.q, self.args.max_step)
            self.hand.command_joint_position(target)
            s.applied = target
            self.read()
            if fraction == 1 and float(s.q.abs().max()) <= 0.02:
                self.finish_reset()
            elif self.reset_elapsed > self.args.setup_seconds + 5.0:
                raise RuntimeError("Hand did not reach the initial pose within setup time plus five seconds")
            return
        self.read()
        if s.phase == "running":
            s.elapsed += dt
            history = self.sim_history if self.args.sim_obs else self.history
            action = policy_action(self.policy, history)
            q = s.sim_q if self.args.sim_obs and not self.args.execute else s.q
            s.applied = applied_targets(action, s.applied, q, self.args.max_step)
            self.hand.command_joint_position(s.applied)
            if self.sim is not None:
                self.sim.step(s.applied)
            self.read()
            self.history = append_history(self.history, observation_frame(s.q, s.applied, s.angle, s.velocity, s.target))
            if self.sim is not None:
                self.sim_history = append_history(self.sim_history, observation_frame(s.sim_q, s.applied, s.sim_angle, s.sim_velocity, s.target))
            s.steps += 1
            angle, velocity = (s.sim_angle, s.sim_velocity) if self.sim is not None and not self.args.execute else (s.angle, s.velocity)
            if abs(wrap(s.target - angle)) <= 0.02 and abs(velocity) <= 0.2:
                self.complete(True)
            elif s.elapsed >= self.args.duration:
                self.complete(False)
        if self.leds is not None:
            self.leds.update(s.angle, self.setup_angle if s.phase == "setup" else s.target, s.phase == "setup")

    def run(self):
        self.start_episode()
        with terminal_keys(self.args.non_interactive) as keys:
            with ExitStack() as stack:
                live = None
                if not self.args.non_interactive:
                    from rich.live import Live
                    live = stack.enter_context(Live(render(self.state), refresh_per_second=10, screen=True, auto_refresh=False))
                last, displayed = time.monotonic(), 0.0
                while self.state.running and (self.app is None or self.app.is_running()):
                    started = time.monotonic()
                    dt, last = started - last, started
                    previous_phase = self.state.phase
                    for key in keys():
                        self.key(key)
                    if not self.state.running:
                        break
                    if previous_phase != "running" and self.state.phase == "running":
                        dt = 0.0
                    with torch.inference_mode():
                        self.tick(dt)
                    if previous_phase != "running" and self.state.phase == "running":
                        last = time.monotonic()
                    if self.sim is not None and self.state.phase != "running":
                        self.sim.render()
                    if live is not None and started - displayed >= 0.1:
                        live.update(render(self.state), refresh=True)
                        displayed = started
                    time.sleep(max(0.0, 1 / CONTROL_HZ - (time.monotonic() - started)))
                    self.state.hz = 1 / max(time.monotonic() - started, 1e-6)

    def save(self):
        args = self.args
        args.log_dir.mkdir(parents=True, exist_ok=True)
        output = dict(policy=str(args.policy.resolve()), policy_sha256=file_sha256(args.policy),
                      mode="hardware" if args.execute else "simulation" if args.sim else "mock",
                      sim_obs=args.sim_obs, duration=args.duration, control_hz=CONTROL_HZ,
                      max_step=args.max_step, setup_seconds=args.setup_seconds, seed=args.seed,
                      calibration=dict(knob_zero_offset=args.knob_zero_offset, knob_direction=args.knob_direction,
                                       joint_offset=args.joint_offset, led_size=args.led_size,
                                       led_offset=args.led_offset, led_direction=args.led_direction),
                      motor_settings=dict(kp=args.kp, ki=args.ki, kd=args.kd, current_limit=args.current_limit),
                      summary=statistics(self.state.records, args.duration), episodes=self.state.records)
        path = args.log_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S_%f}.yaml"
        path.write_text(yaml.safe_dump(output, sort_keys=False))
        print(f"{output['summary']['n_episodes']} episodes; results: {path}", flush=True)


@contextmanager
def terminal_keys(non_interactive):
    if non_interactive:
        yield lambda: ""
        return
    fd = sys.stdin.fileno()
    previous = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        def read():
            return sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else ""
        yield read
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def render(s):
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table

    table = Table("Joint", "Measured °", "Target °", "Sim °", "Real–sim °")
    for i in range(16):
        sim = "—" if s.sim_q is None else f"{math.degrees(float(s.sim_q[i])):+.1f}"
        delta = "—" if s.sim_q is None else f"{math.degrees(float(s.q[i] - s.sim_q[i])):+.1f}"
        table.add_row(str(i), f"{math.degrees(float(s.q[i])):+.1f}", f"{math.degrees(float(s.applied[i])):+.1f}", sim, delta)
    successes = sum(r["success"] for r in s.records)
    return Group(Panel(f"D2H knob | {s.phase.upper()} | {s.hz:.1f} Hz | episode {s.episode} | {s.elapsed:.2f}s | step {s.steps}"),
                 table, Panel(f"Knob {math.degrees(s.angle):+.1f}° | velocity {s.velocity:+.3f} rad/s | target {math.degrees(s.target):+.1f}°\n"
                              f"Sim {math.degrees(s.sim_angle):+.1f}° | velocity {s.sim_velocity:+.3f} rad/s\n"
                              f"Completed {len(s.records)} | success {successes}/{len(s.records)}"),
                 Panel("Space: pause/resume/confirm setup | R: reset | N: next | Q: quit\n" + "\n".join(s.events)))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="D2H training checkpoint or exported TorchScript policy")
    parser.add_argument("--target", type=float, help="Absolute calibrated goal in radians; omitted selects signed 60–90° offsets")
    parser.add_argument("--execute", action="store_true", help="Enable physical hand/knob I/O")
    parser.add_argument("--hand-port")
    parser.add_argument("--knob-port")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--setup-seconds", type=float, default=3.0)
    parser.add_argument("--max-step", type=float, default=0.12, help="Target distance from measured joints per cycle, radians")
    parser.add_argument("--policy-sha256")
    parser.add_argument("--auto-reset", action="store_true")
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("logs/deployment"))
    parser.add_argument("--non-interactive", action="store_true", help="Start automatically; no terminal input or dashboard")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--sim-obs", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sim-device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knob-zero-offset", type=float, default=0.0)
    parser.add_argument("--knob-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--joint-offset", type=float, default=math.pi)
    for name, default in (("kp", 800), ("ki", 0), ("kd", 200), ("current-limit", 500)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--led-port")
    parser.add_argument("--led-size", type=int, default=88)
    parser.add_argument("--led-offset", type=int, default=0)
    parser.add_argument("--led-direction", type=int, choices=(-1, 1), default=1)
    args = parser.parse_args(argv)
    if not args.policy.is_file():
        parser.error(f"Policy does not exist: {args.policy}")
    for name in ("duration", "setup_seconds", "max_step", "knob_zero_offset", "joint_offset"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"{name} must be finite")
    if args.duration <= 0 or args.setup_seconds < 0 or args.max_step <= 0:
        parser.error("duration/max-step must be positive; setup-seconds must be nonnegative")
    if args.target is not None and (not math.isfinite(args.target) or not -math.pi <= args.target <= math.pi):
        parser.error("target must be within [-pi, pi]")
    if (args.sim_obs or args.headless) and not args.sim:
        parser.error("--sim-obs and --headless require --sim")
    if args.led_port and not args.execute:
        parser.error("--led-port requires --execute")
    if not 1 <= args.led_size <= 300:
        parser.error("led-size must be within [1, 300]")
    if any(not 0 <= value <= 65535 for value in (args.kp, args.ki, args.kd, args.current_limit)):
        parser.error("PID gains and current-limit must fit unsigned 16-bit motor registers")
    if args.eval and args.target is not None:
        parser.error("--eval and --target are mutually exclusive")
    if args.eval and args.execute and args.non_interactive:
        parser.error("Hardware evaluation requires interactive setup confirmation")
    if not args.non_interactive and not sys.stdin.isatty():
        parser.error("No interactive terminal; use --non-interactive")
    try:
        load_episodes(args.eval)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(str(exc))
    if args.policy_sha256 and file_sha256(args.policy) != args.policy_sha256.lower():
        parser.error("Policy SHA-256 mismatch")
    return args


@contextmanager
def report_errors():
    try:
        yield
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        raise


def main(argv=None):
    args = parse_args(argv)
    policy = load_low_level_rsl_rl_policy(args.policy, device="cpu", expected_obs_dim=105, expected_action_dim=16)
    policy_action(policy, torch.zeros(1, HISTORY_DIM))
    with ExitStack() as resources, report_errors():
        sim = app = None
        if args.sim:
            from isaaclab.app import AppLauncher
            launcher = AppLauncher(headless=args.headless, device=args.sim_device)
            app = launcher.app
            resources.callback(app.close)
            from scripts.deployment.simulation import SimMirror
            sim = SimMirror(args.sim_device, args.seed)
            resources.callback(sim.disconnect)
        if args.execute:
            hand_port, knob_port = detect_ports(args.hand_port, args.knob_port)
            if args.led_port in (hand_port, knob_port):
                raise ValueError("LED, hand, and knob require separate serial ports")
            hand = LeapHand(hand_port, joint_offset=args.joint_offset, kp=args.kp, ki=args.ki, kd=args.kd, current_limit=args.current_limit)
            knob = DynamixelKnob(knob_port, zero_offset=args.knob_zero_offset, direction=args.knob_direction)
        else:
            hand, knob = MockHand(), MockKnob()
        resources.callback(knob.disconnect)
        resources.callback(hand.disconnect)
        knob.connect()
        hand.connect()
        leds = None
        if args.led_port:
            from scripts.deployment.led import LedRing
            leds = LedRing(args.led_port, args.led_size, args.led_offset, args.led_direction)
            resources.callback(leds.disconnect)
        deployer = KnobDeployer(args, policy, hand, knob, sim, leds, app)
        try:
            deployer.run()
        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            deployer.save()


if __name__ == "__main__":
    main()
