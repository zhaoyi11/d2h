from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.policy.instant_dexterity_recording import (
    InstantDexterityEpisodeRecorder,
    build_bc_action,
    build_bc_observation,
    flatten_scene_state,
    hand_target_to_teacher_action,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTANT_DEXTERITY_PATH = REPO_ROOT / "scripts/instant_dexterity.py"
TRAJECTORY_COMMAND_PATH = REPO_ROOT / "src/policy/high_level/trajectory_command.py"
ROTATE_OBJECT_ENV_CFG_PATH = REPO_ROOT / "src/tasks/rotate_knob/env_cfg.py"


def _load_script_functions(*names: str, globals_dict: dict | None = None) -> dict[str, object]:
    tree = ast.parse(INSTANT_DEXTERITY_PATH.read_text())
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
    assert set(functions) == set(names)
    module = ast.Module(
        body=[*ast.parse("from __future__ import annotations").body, *(functions[name] for name in names)],
        type_ignores=[],
    )
    namespace = {} if globals_dict is None else dict(globals_dict)
    exec(compile(ast.fix_missing_locations(module), INSTANT_DEXTERITY_PATH, "exec"), namespace)
    return {name: namespace[name] for name in names}


def test_build_bc_observation_uses_documented_feature_order() -> None:
    low_level = torch.arange(2 * 155, dtype=torch.float32).reshape(2, 155)
    arm_pos = torch.full((2, 7), 1.0)
    arm_vel = torch.full((2, 7), 2.0)
    hand_base_command = torch.full((2, 7), 3.0)

    observation = build_bc_observation(low_level, arm_pos, arm_vel, hand_base_command)

    assert observation.shape == (2, 176)
    torch.testing.assert_close(observation[:, :155], low_level)
    torch.testing.assert_close(observation[:, 155:162], arm_pos)
    torch.testing.assert_close(observation[:, 162:169], arm_vel)
    torch.testing.assert_close(observation[:, 169:176], hand_base_command)


def test_build_bc_action_preserves_absolute_and_relative_arm_targets() -> None:
    arm_pos = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    arm_target = arm_pos + 0.05
    hand_target = torch.arange(16, dtype=torch.float32).unsqueeze(0)

    action, arm_delta = build_bc_action(arm_target, arm_pos, hand_target)

    assert action.shape == (1, 23)
    torch.testing.assert_close(arm_delta, torch.full((1, 7), 0.05))
    torch.testing.assert_close(action[:, :7], arm_delta)
    torch.testing.assert_close(action[:, 7:], hand_target)


def test_hand_target_to_teacher_action_inverts_ema_and_joint_scaling() -> None:
    lower = torch.tensor([[-2.0, -1.0]])
    upper = torch.tensor([[2.0, 3.0]])
    previous_target = torch.tensor([[0.0, 1.0]])
    teacher_action = torch.tensor([[0.5, -0.5]])
    pre_ema_target = (teacher_action + 1.0) * 0.5 * (upper - lower) + lower
    desired_target = 0.5 * pre_ema_target + 0.5 * previous_target

    reconstructed = hand_target_to_teacher_action(
        desired_target,
        previous_target,
        lower,
        upper,
        alpha=0.5,
    )

    torch.testing.assert_close(reconstructed, teacher_action)


def test_flatten_scene_state_records_articulations_and_rigid_objects() -> None:
    scene_state = {
        "articulation": {
            "robot": {
                "root_pose": torch.zeros(2, 7),
                "root_velocity": torch.ones(2, 6),
                "joint_position": torch.full((2, 23), 2.0),
                "joint_velocity": torch.full((2, 23), 3.0),
            }
        },
        "rigid_object": {
            "object": {
                "root_pose": torch.full((2, 7), 4.0),
                "root_velocity": torch.full((2, 6), 5.0),
            }
        },
        "gripper": {"ignored": torch.ones(2, 1)},
    }

    flattened = flatten_scene_state(scene_state)

    assert set(flattened) == {
        "state.articulation.robot.root_pose",
        "state.articulation.robot.root_velocity",
        "state.articulation.robot.joint_position",
        "state.articulation.robot.joint_velocity",
        "state.rigid_object.object.root_pose",
        "state.rigid_object.object.root_velocity",
    }
    assert flattened["state.articulation.robot.joint_position"].dtype == np.float32


def _step_data(num_envs: int, offset: float) -> dict[str, np.ndarray]:
    return {
        "observation.low_level": np.full((num_envs, 155), offset, dtype=np.float32),
        "observation.bc": np.full((num_envs, 176), offset + 1.0, dtype=np.float32),
        "action": np.full((num_envs, 23), offset + 2.0, dtype=np.float32),
        "state.articulation.robot.joint_position": np.full(
            (num_envs, 23), offset + 3.0, dtype=np.float32
        ),
    }


def test_episode_recorder_splits_completed_envs_and_flushes_partial_episodes(tmp_path: Path) -> None:
    recorder = InstantDexterityEpisodeRecorder(
        output_dir=tmp_path,
        num_envs=2,
        metadata={"task": "Clean_Table_HRL-v0", "action_dim": 23},
    )

    recorder.add_step(
        _step_data(2, 0.0),
        reward=np.array([1.0, 2.0], dtype=np.float32),
        terminated=np.array([True, False]),
        truncated=np.array([False, False]),
    )
    recorder.add_step(
        _step_data(2, 10.0),
        reward=np.array([3.0, 4.0], dtype=np.float32),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
    )
    recorder.close()

    episode_paths = sorted(tmp_path.glob("*.npz"))
    assert [path.name for path in episode_paths] == ["0000000000.npz", "0000000001.npz", "0000000002.npz"]

    with np.load(episode_paths[0]) as episode:
        assert episode["action"].shape == (1, 23)
        assert bool(episode["complete"])
        assert int(episode["source_env_id"]) == 0
        np.testing.assert_array_equal(episode["terminated"], np.array([True]))

    with np.load(episode_paths[1]) as episode:
        assert episode["action"].shape == (1, 23)
        assert not bool(episode["complete"])
        assert int(episode["source_env_id"]) == 0

    with np.load(episode_paths[2]) as episode:
        assert episode["action"].shape == (2, 23)
        assert not bool(episode["complete"])
        assert int(episode["source_env_id"]) == 1

    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata == {"action_dim": 23, "task": "Clean_Table_HRL-v0"}


def test_episode_recorder_close_is_idempotent(tmp_path: Path) -> None:
    recorder = InstantDexterityEpisodeRecorder(tmp_path, num_envs=1, metadata={})
    recorder.add_step(
        _step_data(1, 0.0),
        reward=np.zeros(1, dtype=np.float32),
        terminated=np.zeros(1, dtype=bool),
        truncated=np.zeros(1, dtype=bool),
    )

    recorder.close()
    recorder.close()

    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_episode_recorder_refuses_to_overwrite_existing_dataset(tmp_path: Path) -> None:
    (tmp_path / "0000000000.npz").write_bytes(b"existing recording")

    with pytest.raises(FileExistsError, match="already contains recorded data"):
        InstantDexterityEpisodeRecorder(tmp_path, num_envs=1, metadata={})


def _method_source(class_node: ast.ClassDef, name: str) -> str:
    method = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.unparse(method)


def test_curobo_action_exposes_last_target_and_order_without_resetting_the_target() -> None:
    source_path = REPO_ROOT / "src/tasks/common/mdps/action_manager/curobo_mpc.py"
    tree = ast.parse(source_path.read_text())
    action_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CommandHandBaseCuroboMpcAction"
    )
    method_names = {
        node.name for node in action_class.body if isinstance(node, ast.FunctionDef)
    }

    assert "last_joint_position_target" in method_names
    assert "ordered_joint_ids" in method_names
    assert "ordered_joint_names" in method_names

    init_source = _method_source(action_class, "__init__")
    process_source = _method_source(action_class, "process_actions")
    reset_source = _method_source(action_class, "reset")
    assert "self._last_joint_position_target = self._cmd_pos.clone()" in init_source
    assert "self._last_joint_position_target[:] = self._cmd_pos" in process_source
    assert "_last_joint_position_target" not in reset_source


def test_instant_dexterity_exposes_recording_cli_and_student_action_contract() -> None:
    script_path = REPO_ROOT / "scripts/instant_dexterity.py"
    source = script_path.read_text()
    tree = ast.parse(source)

    assert 'parser.add_argument("--record_data", action="store_true"' in source
    assert 'parser.add_argument("--record_dir", type=str, default=None' in source
    assert "InstantDexterityEpisodeRecorder" in source
    assert "build_bc_action" in source
    assert "build_bc_observation" in source
    assert "flatten_scene_state" in source
    assert "env_unwrapped.scene.get_state(is_relative=True)" in source
    assert "arm_action.last_joint_position_target" in source
    assert "hand_action.processed_actions" in source

    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_source = ast.unparse(main)
    assert "observations, _ = env.reset()" in main_source
    assert "recorder.close()" in main_source
    assert "hand_policy_action = low_level_policy.act(low_level_obs)" in main_source
    assert "'observation.low_level': low_level_obs" in main_source


def test_instant_dexterity_enables_object_command_frames_without_shadow_marker() -> None:
    tree = ast.parse(INSTANT_DEXTERITY_PATH.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    main_source = ast.unparse(main)
    command_source = TRAJECTORY_COMMAND_PATH.read_text()
    env_source = ROTATE_OBJECT_ENV_CFG_PATH.read_text()

    assert "env_cfg.commands.object_pose.debug_vis = True" in main_source
    assert "_make_target_object_marker" not in INSTANT_DEXTERITY_PATH.read_text()
    assert 'prim_path="/Visuals/Command/goal_pose"' in command_source
    assert 'prim_path="/Visuals/Command/body_pose"' in command_source
    assert "self.commands.object_pose.position_only = False" in env_source
