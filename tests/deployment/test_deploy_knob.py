"""Hardware-free checks for the deployment contract and operator transitions."""

import math
from types import SimpleNamespace

import pytest
import torch

from scripts.deployment import hardware
from scripts.deployment.deploy_knob_student import (
    KnobDeployer, LOWER, UPPER, applied_targets, load_episodes, observation_frame,
    parse_args, policy_action, statistics,
)
from scripts.deployment.led import LedRing
from src.policy.knob_interface import REAL_TO_SIM, SIM_TO_REAL


class Policy:
    def act(self, observation):
        return torch.zeros(1, 16)


def controller(tmp_path, **overrides):
    options = dict(eval=None, execute=False, non_interactive=True, sim_obs=False, target=1.5,
                   setup_seconds=0.0, duration=0.1, max_step=0.12, seed=0, auto_reset=False)
    options.update(overrides)
    return KnobDeployer(SimpleNamespace(**options), Policy(), hardware.MockHand(), hardware.MockKnob())


def test_action_history_and_single_completion(tmp_path):
    d = controller(tmp_path)
    d.start_episode()
    d.tick(1 / 60)
    initial = d.history.clone()
    d.tick(1 / 60)
    expected = applied_targets(torch.zeros(16), torch.zeros(16), torch.zeros(16), 0.12)
    torch.testing.assert_close(d.state.applied, expected)
    torch.testing.assert_close(d.history[:, :70], initial[:, 35:])
    torch.testing.assert_close(d.history[:, -35:], observation_frame(d.state.q, expected, 0, 0, 1.5))
    d.complete(False)
    d.complete(False)
    assert len(d.state.records) == 1
    assert not d.state.running
    assert torch.all(expected >= LOWER) and torch.all(expected <= UPPER)


def test_pause_reset_next_and_quit(tmp_path):
    d = controller(tmp_path, non_interactive=False)
    d.start_episode()
    d.tick(0.01)
    assert d.state.phase == "paused"
    d.tick(5)
    assert d.state.elapsed == 0
    d.key(" ")
    d.tick(0.01)
    d.key(" ")
    elapsed = d.state.elapsed
    d.tick(10)
    assert d.state.elapsed == elapsed
    target = d.state.target
    d.key("r")
    d.tick(0.01)
    assert d.state.elapsed == 0 and d.state.target == target
    torch.testing.assert_close(d.history[:, :35], d.history[:, -35:])
    d.key("n")
    assert d.state.episode == 2
    d.key("q")
    assert not d.state.running


def test_eval_conversion_progression_and_setup(tmp_path):
    path = tmp_path / "episodes.yaml"
    path.write_text("episodes:\n- knob_start: 0.5\n  target: 0.5\n- knob_start: 0.75\n  target: 0.75\n")
    assert load_episodes(path) == [(0, 0), (math.pi / 2, math.pi / 2)]
    d = controller(tmp_path, eval=path)
    d.start_episode()
    for _ in range(4):
        d.tick(0.01)
    assert not d.state.running
    assert len(d.state.records) == 2
    assert all(r["success"] for r in d.state.records)
    d = controller(tmp_path, eval=path, execute=True, non_interactive=False)
    d.start_episode()
    d.tick(0.01)
    assert d.state.phase == "setup"
    d.state.angle = 0.5
    d.key(" ")
    assert d.state.phase == "setup"
    d.state.angle = 0
    d.key(" ")
    assert d.state.phase == "running"
    path.write_text("episodes:\n- knob_start: .nan\n  target: 1\n")
    with pytest.raises(ValueError):
        load_episodes(path)


def test_nonfinite_rejected_before_command(tmp_path):
    d = controller(tmp_path)
    d.start_episode()
    d.tick(0.01)
    before = d.hand.position.clone()
    d.knob.angle = float("nan")
    with pytest.raises(ValueError):
        d.tick(0.01)
    torch.testing.assert_close(d.hand.position, before)
    with pytest.raises(ValueError):
        policy_action(SimpleNamespace(act=lambda obs: torch.full((1, 16), float("inf"))), torch.zeros(1, 105))


def test_knob_calibration_wrap_and_joint_roundtrip(monkeypatch):
    class Bus:
        def __init__(self, *args):
            self.values = torch.tensor([3.13])
        def read_positions(self):
            return self.values
        def command(self, positions):
            self.values = positions
    monkeypatch.setattr(hardware, "MotorBus", Bus)
    times = iter((1.0, 1.1, 1.2))
    monkeypatch.setattr(hardware.time, "monotonic", lambda: next(times))
    knob = hardware.DynamixelKnob("unused", zero_offset=0.2, direction=-1)
    angle, velocity = knob.read_position()
    assert angle == pytest.approx(-2.93) and velocity == 0
    knob.bus.values = torch.tensor([-3.13])
    angle, velocity = knob.read_position()
    assert -0.24 < velocity < -0.22
    hand = hardware.LeapHand("unused", joint_offset=math.pi)
    target = (LOWER + UPPER) / 2
    hand.command_joint_position(target)
    torch.testing.assert_close(hand.bus.values, target[SIM_TO_REAL] + math.pi)
    torch.testing.assert_close(hand.poll_joint_position(), target)
    assert list(torch.arange(16)[SIM_TO_REAL][REAL_TO_SIM]) == list(range(16))


def test_transport_rejects_missing_data_and_closes_all_motors():
    bus = hardware.MotorBus.__new__(hardware.MotorBus)
    bus.dxl = SimpleNamespace(COMM_SUCCESS=0)
    bus.ids = [0, 1]
    bus.reader = SimpleNamespace(txRxPacket=lambda: 0, isAvailable=lambda m, a, n: m == 0)
    with pytest.raises(OSError):
        bus.read_positions()
    calls = []
    def write(motor, *args):
        calls.append(motor)
        if motor == 0:
            raise OSError("disconnected")
    hand = hardware.LeapHand.__new__(hardware.LeapHand)
    hand.connected = True
    hand.bus = SimpleNamespace(write_register=write, disconnect=lambda: calls.append("closed"))
    with pytest.raises(OSError):
        hand.disconnect()
    assert calls == list(range(16)) + ["closed"]


def test_led_protocol_and_statistics():
    ring = LedRing.__new__(LedRing)
    ring.size, ring.offset, ring.direction = 88, 3, -1
    assert ring.index(-math.pi) == ring.index(math.pi) == 3
    assert LedRing.packet(299, (1, 2, 3)) == bytes([1, 43, 1, 2, 3])
    records = [dict(success=True, elapsed=2, dist_to_target=0), dict(success=False, elapsed=8, dist_to_target=0.5)]
    stats = statistics(records, 8)
    assert stats["success_rate"] == 0.5 and stats["fitness"] == 1.5


def test_cli_rejects_invalid_modes_and_numbers(tmp_path):
    policy = tmp_path / "policy.pt"
    policy.touch()
    for flags in (("--duration", "nan"), ("--sim-obs",), ("--led-port", "unused"), ("--max-step", "0")):
        with pytest.raises(SystemExit):
            parse_args([str(policy), "--non-interactive", *flags])


def test_reset_checks_reads_and_limits_each_target(tmp_path):
    d = controller(tmp_path)
    d.hand.position = torch.full((16,), 0.3)
    d.start_episode()
    d.tick(0.01)
    torch.testing.assert_close(d.hand.position, torch.full((16,), 0.18))
    before = d.hand.position.clone()
    d.knob.read_position = lambda: (float("nan"), 0)
    with pytest.raises(ValueError):
        d.tick(0.01)
    torch.testing.assert_close(d.hand.position, before)


def test_interactive_auto_reset_resumes(tmp_path):
    d = controller(tmp_path, auto_reset=True, non_interactive=False, target=0)
    d.start_episode()
    d.tick(0.01)
    assert d.state.phase == "paused"
    d.key(" ")
    d.complete(True)
    assert d.state.phase == "resetting"
    d.tick(0.01)
    assert d.state.phase == "running" and d.state.episode == 2


def test_mock_main_never_scans_ports_and_saves_results(tmp_path, monkeypatch):
    from scripts.deployment import deploy_knob_student as deploy
    policy_path = tmp_path / "policy.pt"
    policy_path.touch()
    monkeypatch.setattr(deploy, "load_low_level_rsl_rl_policy", lambda *a, **k: Policy())
    def forbidden(*args, **kwargs):
        raise AssertionError("Hardware was touched")
    monkeypatch.setattr(deploy, "detect_ports", forbidden)
    monkeypatch.setattr(hardware, "MotorBus", forbidden)
    deploy.main([str(policy_path), "--non-interactive", "--target", "0", "--setup-seconds", "0", "--log-dir", str(tmp_path)])
    import yaml
    output = yaml.safe_load(next(tmp_path.glob("eval_*.yaml")).read_text())
    assert output["mode"] == "mock" and output["summary"]["success_rate"] == 1


def test_all_devices_close_on_connection_failure(tmp_path, monkeypatch):
    from scripts.deployment import deploy_knob_student as deploy
    policy_path = tmp_path / "policy.pt"
    policy_path.touch()
    monkeypatch.setattr(deploy, "load_low_level_rsl_rl_policy", lambda *a, **k: Policy())
    closed = []
    class Hand(hardware.MockHand):
        def connect(self):
            raise OSError("connection failed")
        def disconnect(self):
            closed.append("hand")
            raise OSError("shutdown failed")
    class Knob(hardware.MockKnob):
        def disconnect(self):
            closed.append("knob")
    monkeypatch.setattr(deploy, "MockHand", Hand)
    monkeypatch.setattr(deploy, "MockKnob", Knob)
    with pytest.raises(OSError):
        deploy.main([str(policy_path), "--non-interactive"])
    assert closed == ["hand", "knob"]


class ExportedPolicy(torch.nn.Module):
    def __init__(self, actor, mean, std):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, obs):
        return self.actor((obs - self.mean) / (self.std + 0.01))


def test_training_checkpoint_and_jit_action_parity(tmp_path):
    from src.policy.low_level.policy import load_low_level_rsl_rl_policy
    actor = torch.nn.Sequential(torch.nn.Linear(105, 4), torch.nn.ELU(), torch.nn.Linear(4, 16))
    state = {f"student.{key}": value for key, value in actor.state_dict().items()}
    mean, std = torch.full((1, 105), 0.2), torch.full((1, 105), 2.0)
    state.update({"student_obs_normalizer._mean": mean, "student_obs_normalizer._std": std})
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state_dict": state}, checkpoint)
    jit = tmp_path / "policy.pt"
    torch.jit.trace(ExportedPolicy(actor, mean, std), torch.zeros(1, 105)).save(str(jit))
    policies = [load_low_level_rsl_rl_policy(p, device="cpu", expected_obs_dim=105, expected_action_dim=16) for p in (checkpoint, jit)]
    observation = torch.randn(1, 105)
    torch.testing.assert_close(policy_action(policies[0], observation), policy_action(policies[1], observation))


def test_simulation_random_goal_preserves_start_angle(tmp_path):
    class Sim:
        angle, velocity = 1.4, 0.1
        def read(self):
            return torch.zeros(16), self.angle, self.velocity
        def reset(self, q, angle, target, velocity=0):
            self.angle, self.velocity = angle, velocity
    d = controller(tmp_path, target=None)
    d.sim = Sim()
    d.start_episode()
    goal = d.state.target
    d.tick(0.01)
    assert d.sim.angle == 1.4 and d.sim.velocity == 0.1
    assert math.pi / 3 <= abs(hardware.wrap(goal - d.sim.angle)) <= math.pi / 2


def test_invalid_calibration_cannot_bypass_target_limiter():
    with pytest.raises(ValueError, match="calibration"):
        applied_targets(torch.zeros(16), torch.zeros(16), torch.full((16,), 4.0), 0.12)
