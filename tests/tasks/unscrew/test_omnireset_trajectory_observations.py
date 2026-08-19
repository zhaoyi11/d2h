from __future__ import annotations

import ast
import importlib.util
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

from src.policy.high_level.trajectory_stepper import TrajectoryStepper


REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_CFG = REPO_ROOT / "src/tasks/unscrew_omnireset/env_cfg.py"


def _load_file_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_command_module(monkeypatch):
    captured = {}

    def subtract_frame_transforms(parent_pos, parent_quat, child_pos, child_quat):
        captured["frames"] = (
            parent_pos.clone(),
            parent_quat.clone(),
            child_pos.clone(),
            child_quat.clone(),
        )
        return child_pos - parent_pos, child_quat

    isaaclab = ModuleType("isaaclab")
    assets = ModuleType("isaaclab.assets")
    assets.Articulation = object
    assets.RigidObject = object
    managers = ModuleType("isaaclab.managers")
    managers.CommandTerm = object
    managers.CommandTermCfg = object
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.combine_frame_transforms = lambda *args: (args[2], args[3])
    math_utils.compute_pose_error = lambda *args: (args[2] - args[0], args[3] - args[1])
    math_utils.subtract_frame_transforms = subtract_frame_transforms

    for name, module in {
        "isaaclab": isaaclab,
        "isaaclab.assets": assets,
        "isaaclab.managers": managers,
        "isaaclab.utils": utils,
        "isaaclab.utils.math": math_utils,
        "src.tasks.unscrew.mdps.trajectory": trajectory,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module = _load_file_module(
        "omnireset_commands", "src/tasks/unscrew_omnireset/mdps/commands.py"
    )
    return module, captured


trajectory = _load_file_module("unscrew_trajectory", "src/tasks/unscrew/mdps/trajectory.py")
contact_filters = _load_file_module(
    "omnireset_contact_filters", "src/tasks/unscrew_omnireset/mdps/contact_filters.py"
)
build_unscrew_object_pose_sequence = trajectory.build_unscrew_object_pose_sequence
external_indices = contact_filters.external_indices
object_indices = contact_filters.object_indices


def _nested_class_terms(source: str, outer_name: str, inner_name: str) -> list[str]:
    tree = ast.parse(source)
    outer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == outer_name
    )
    inner = next(
        node for node in outer.body if isinstance(node, ast.ClassDef) and node.name == inner_name
    )
    return [
        node.targets[0].id
        for node in inner.body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    ]


def test_twist_lift_trajectory_shape_and_offsets() -> None:
    initial = torch.tensor([0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0])
    trajectory = build_unscrew_object_pose_sequence(
        initial,
        segment_steps=(0, 0) + (2,) * 6 + (1, 1),
        twist_total_angle=2.0 * math.pi,
        twist_segments=6,
        thread_pitch=0.02,
        extraction_height=0.10,
    )

    assert trajectory.shape == (15, 7)
    torch.testing.assert_close(trajectory[0], initial)
    torch.testing.assert_close(trajectory[-3, 2], initial[2] + 0.02)
    torch.testing.assert_close(trajectory[-2:, 2], torch.full((2,), initial[2] + 0.12))
    for index in range(1, 13):
        angle = index * math.pi / 6.0
        expected_quat = torch.tensor(
            [
                math.cos(angle / 2.0),
                0.0,
                0.0,
                math.sin(angle / 2.0),
            ]
        )
        alignment = torch.abs(torch.dot(trajectory[index, 3:7], expected_quat))
        torch.testing.assert_close(alignment, torch.tensor(1.0))


def test_command_rebases_each_environment_from_its_live_reset_pose(monkeypatch) -> None:
    command_module, captured = _load_command_module(monkeypatch)
    cfg = SimpleNamespace(
        trajectory_segment_steps=(2,) * 6 + (1, 1),
        twist_total_angle=2.0 * math.pi,
        twist_segments=6,
        thread_pitch=0.02,
        extraction_height=0.10,
    )
    robot_pos_w = torch.tensor([[10.0, 0.0, 0.0], [-5.0, 2.0, 1.0]])
    object_pos_w = torch.tensor([[10.4, 0.1, 0.3], [-4.5, 2.2, 1.4]])
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)

    term = object.__new__(command_module.UnscrewOmniResetTrajectoryCommand)
    term.cfg = cfg
    term.num_envs = 2
    term.device = "cpu"
    term.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=robot_pos_w,
            root_quat_w=identity_quat,
        )
    )
    term.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=object_pos_w,
            root_quat_w=identity_quat,
        )
    )
    term.pose_command_b = torch.zeros(2, 7)
    term._stepper = TrajectoryStepper(
        num_envs=2,
        device="cpu",
        segment_steps=cfg.trajectory_segment_steps,
        stage_position_tolerance=torch.full((8,), 0.005),
        stage_orientation_tolerance=torch.full((8,), 0.20),
    )

    env_ids = torch.tensor([0, 1])
    term._resample_command(env_ids)

    parent_pos, parent_quat, child_pos, child_quat = captured["frames"]
    torch.testing.assert_close(parent_pos, robot_pos_w)
    torch.testing.assert_close(parent_quat, identity_quat)
    torch.testing.assert_close(child_pos, object_pos_w)
    torch.testing.assert_close(child_quat, identity_quat)

    expected_start = torch.cat((object_pos_w - robot_pos_w, identity_quat), dim=1)
    torch.testing.assert_close(term.pose_command_b, expected_start)
    torch.testing.assert_close(term._stepper._trajectory[:, 0], expected_start)
    expected_final_pos = expected_start[:, :3] + torch.tensor([0.0, 0.0, 0.12])
    torch.testing.assert_close(term._stepper._trajectory[:, -1, :3], expected_final_pos)
    assert term._stepper._trajectory.shape == (2, 15, 7)
    assert term._stepper.step.tolist() == [0, 0]


def test_stepper_uses_fixed_thresholds_and_clamps_final_step() -> None:
    stepper = TrajectoryStepper(
        num_envs=2,
        device="cpu",
        segment_steps=(2,) * 6 + (1, 1),
        stage_position_tolerance=torch.full((8,), 0.005),
        stage_orientation_tolerance=torch.full((8,), 0.20),
    )
    trajectories = torch.zeros(2, 15, 7)
    trajectories[:, :, 3] = 1.0
    env_ids = torch.tensor([0, 1])
    stepper.reset(env_ids, trajectories)

    assert stepper.object_target_achieved(
        torch.tensor([0.004, 0.006]), torch.tensor([0.19, 0.19]), position_only=False
    ).tolist() == [True, False]
    for _ in range(20):
        stepper.advance(env_ids)
    assert stepper.step.tolist() == [14, 14]


def test_contact_filter_groups_keep_object_and_external_contacts_separate() -> None:
    assert object_indices() == [0]
    assert external_indices() == [1, 2]


def test_observation_groups_match_expected_current_step_contract() -> None:
    source = ENV_CFG.read_text()
    assert _nested_class_terms(source, "ObservationsCfg", "LowLevelObsCfg") == [
        "joint_pos",
        "joint_vel",
        "fingertip_pose",
        "contact_mask",
        "contact_force_mag",
        "contact_pose",
        "external_contact_mask",
        "external_contact_force_mag",
        "external_contact_pose",
        "object_pos",
        "object_quat",
        "object_lin_vel",
        "object_ang_vel",
        "gravity_dir",
        "goal_pos_diff",
        "goal_quat_diff",
        "last_action",
    ]
    assert _nested_class_terms(source, "ObservationsCfg", "ResidualObsCfg") == [
        "arm_joint_pos",
        "arm_joint_vel",
        "hand_base_state",
        "receptive_pose",
        "object_pose_receptive",
        "trajectory_command",
        "last_arm_action",
    ]
    low_level_dims = (16, 16, 52, 4, 4, 8, 4, 4, 8, 3, 4, 3, 3, 3, 3, 4, 16)
    residual_dims = (7, 7, 13, 7, 7, 7, 7)
    assert sum(low_level_dims) == 155
    assert sum(residual_dims) == 55
    assert source.count("self.history_length = 1") == 2
    assert 'contact_pose_range_deg": 90.0' in source
    assert "FrameTransformerCfg(" in source
    assert "fingers_contact_force_b" not in source


def test_env_wires_command_rewards_and_210_dim_training_groups() -> None:
    source = ENV_CFG.read_text()
    assert "class CommandsCfg" in source
    assert "commands: CommandsCfg = CommandsCfg()" in source
    assert 'command_name": "object_pose"' in source
    assert '"std": 0.20' in source
    assert '"std": 0.50' in source

    trainer_source = (
        REPO_ROOT / "src/tasks/unscrew_omnireset/rsl_rl_ppo_cfg.py"
    ).read_text()
    assert '"policy": ["low_level", "residual"]' in trainer_source
    assert '"critic": ["low_level", "residual"]' in trainer_source
