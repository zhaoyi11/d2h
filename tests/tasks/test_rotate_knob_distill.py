import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

from src.policy.knob_interface import FRAME_DIM, HISTORY_DIM


ROOT = Path(__file__).resolve().parents[2]
COMMANDS_PATH = ROOT / "src/tasks/rotate_knob_distill/mdps/commands.py"


def _load_commands_module():
    module_names = (
        "isaaclab",
        "isaaclab.assets",
        "isaaclab.managers",
        "isaaclab.markers",
        "isaaclab.markers.config",
        "isaaclab.utils",
        "isaaclab.utils.math",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    modules = {name: types.ModuleType(name) for name in module_names}
    modules["isaaclab.assets"].Articulation = object
    modules["isaaclab.assets"].RigidObject = object
    modules["isaaclab.managers"].CommandTerm = object
    modules["isaaclab.managers"].CommandTermCfg = object
    modules["isaaclab.markers"].VisualizationMarkers = object
    modules["isaaclab.markers"].VisualizationMarkersCfg = object
    marker_cfg = types.SimpleNamespace(
        replace=lambda **kwargs: types.SimpleNamespace(
            markers={"arrow": types.SimpleNamespace(scale=None)}
        )
    )
    modules["isaaclab.markers.config"].BLUE_ARROW_X_MARKER_CFG = marker_cfg
    modules["isaaclab.markers.config"].GREEN_ARROW_X_MARKER_CFG = marker_cfg
    modules["isaaclab.utils"].configclass = lambda cls: cls
    modules["isaaclab.utils.math"].quat_from_angle_axis = lambda angle, axis: torch.cat(
        (torch.cos(angle / 2).unsqueeze(-1), axis * torch.sin(angle / 2).unsqueeze(-1)), dim=-1
    )
    modules["isaaclab.utils.math"].quat_mul = _quat_mul
    modules["isaaclab.utils.math"].subtract_frame_transforms = (
        lambda robot_pos, robot_quat, goal_pos, goal_quat: (goal_pos, goal_quat)
    )
    sys.modules.update(modules)

    spec = importlib.util.spec_from_file_location("knob_commands_under_test", COMMANDS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return module


def _yaw_quat(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (torch.cos(angles / 2), torch.zeros_like(angles), torch.zeros_like(angles), torch.sin(angles / 2)), dim=-1
    )


def _quat_mul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = left.unbind(-1)
    rw, rx, ry, rz = right.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def test_knob_distillation_contract_is_locked() -> None:
    env = (ROOT / "src/tasks/rotate_knob_distill/env_cfg.py").read_text()
    runner = (ROOT / "src/tasks/rotate_knob_distill/rsl_rl_distillation_cfg.py").read_text()
    registry = (ROOT / "src/tasks/__init__.py").read_text()

    assert FRAME_DIM == 35
    assert HISTORY_DIM == 105
    assert "history_length=3" in env
    assert "EMAJointPositionToLimitsActionCfg" in env
    assert "alpha=0.5" in env
    assert "rescale_to_limits=True" in env
    assert "dt=1.0 / 240.0" in env
    assert "self.decimation = 4" in env
    assert "self.episode_length_s = 15.0" in env
    assert "success = DoneTerm" not in env
    assert 'obs_groups = {"policy": ["policy"], "teacher": ["low_level"]}' in runner
    assert "student_hidden_dims=[512, 256, 128]" in runner
    assert "teacher_hidden_dims=[512, 256, 128]" in runner
    assert 'id="Rotate_Knob_Distill-v0"' in registry


def test_knob_command_resamples_achieved_goals_without_velocity_gate() -> None:
    module = _load_commands_module()
    assert module.KnobTargetCommandCfg.magnitude_range == (math.pi / 3.0, math.pi / 2.0)
    assert module.KnobTargetCommandCfg.angle_threshold == 0.1
    current_angles = torch.tensor([0.4, -0.7, 1.2])
    command = object.__new__(module.KnobTargetCommand)
    command.device = "cpu"
    command.cfg = types.SimpleNamespace(
        magnitude_range=(math.pi / 3.0, math.pi / 3.0),
        angle_threshold=0.1,
    )
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.zeros(3, 3),
            root_quat_w=_yaw_quat(torch.zeros(3)),
        )
    )
    command.object = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.zeros(3, 3),
            root_quat_w=_yaw_quat(current_angles),
        )
    )
    command.pose_command_b = torch.zeros(3, 7)
    command.pose_command_w = torch.zeros(3, 7)
    command.pose_command_w[:, 3:] = _yaw_quat(current_angles)
    command.metrics = {"angle_error": torch.tensor([0.1, 0.09, 0.11])}

    command._update_command()

    target_delta = module.wrap_to_pi(command.target_angle - current_angles)
    torch.testing.assert_close(target_delta[:2].abs(), torch.full((2,), math.pi / 3.0))
    torch.testing.assert_close(target_delta[2], torch.tensor(0.0))


def test_knob_command_preserves_quaternion_branch_across_pi(monkeypatch) -> None:
    module = _load_commands_module()
    current_angles = torch.tensor([5.0 * math.pi / 6.0, -5.0 * math.pi / 6.0])
    current_quat = _yaw_quat(current_angles)
    command = object.__new__(module.KnobTargetCommand)
    command.device = "cpu"
    command.cfg = types.SimpleNamespace(magnitude_range=(math.pi / 3.0, math.pi / 3.0))
    command.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=_yaw_quat(torch.zeros(2)))
    )
    command.object = types.SimpleNamespace(
        data=types.SimpleNamespace(root_pos_w=torch.zeros(2, 3), root_quat_w=current_quat)
    )
    command.pose_command_b = torch.zeros(2, 7)
    command.pose_command_w = torch.zeros(2, 7)
    monkeypatch.setattr(module.torch, "randint", lambda *args, **kwargs: torch.tensor([1, 0]))

    command._resample_command([0, 1])

    expected_angles = torch.tensor([-5.0 * math.pi / 6.0, 5.0 * math.pi / 6.0])
    torch.testing.assert_close(command.target_angle, expected_angles)
    assert torch.all(torch.sum(current_quat * command.pose_command_w[:, 3:], dim=-1) > 0.0)
    goal_conjugate = command.pose_command_w[:, 3:].clone()
    goal_conjugate[:, 1:] *= -1.0
    relative_quat = _quat_mul(current_quat, goal_conjugate)
    torch.testing.assert_close(relative_quat[:, 0], torch.full((2,), math.cos(math.pi / 6.0)))


def test_knob_debug_visualization_shows_goal_and_current_yaw() -> None:
    module = _load_commands_module()
    command = object.__new__(module.KnobTargetCommand)
    command.robot = types.SimpleNamespace(is_initialized=True)
    command.pose_command_w = torch.tensor([[0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.7]])
    command.object = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_pos_w=torch.tensor([[0.1, 0.2, 0.3]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    calls = []
    command.goal_visualizer = types.SimpleNamespace(visualize=lambda *args: calls.append(args))
    command.current_visualizer = types.SimpleNamespace(visualize=lambda *args: calls.append(args))

    command._debug_vis_callback(None)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], command.object.data.root_pos_w)
    torch.testing.assert_close(calls[0][1], command.pose_command_w[:, 3:])
    torch.testing.assert_close(calls[1][0], command.object.data.root_pos_w)
    torch.testing.assert_close(calls[1][1], command.object.data.root_quat_w)
