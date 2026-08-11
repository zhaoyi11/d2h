from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = REPO_ROOT / "src/tasks/clean_table_omnireset/env_cfg.py"

ROBOT_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "a_1",
    "a_12",
    "a_5",
    "a_9",
    "a_0",
    "a_13",
    "a_4",
    "a_8",
    "a_2",
    "a_14",
    "a_6",
    "a_10",
    "a_3",
    "a_15",
    "a_7",
    "a_11",
]

STATE_FIELDS = [
    "state.articulation.robot.root_pose",
    "state.articulation.robot.root_velocity",
    "state.articulation.robot.joint_position",
    "state.articulation.robot.joint_velocity",
    "state.rigid_object.object.root_pose",
    "state.rigid_object.object.root_velocity",
    "state.rigid_object.receptive_object.root_pose",
    "state.rigid_object.receptive_object.root_velocity",
    "state.rigid_object.table.root_pose",
    "state.rigid_object.table.root_velocity",
]


def _metadata() -> dict:
    return {
        "schema_version": 1,
        "task": "Clean_Table_HRL-v0",
        "num_envs": 1,
        "joint_order": {"articulations": {"robot": ROBOT_JOINT_NAMES}},
        "scene_state": {"relative_to_env_origin": True, "fields": STATE_FIELDS},
    }


def _episode(length: int, offset: float, source_env_id: int = 0) -> dict[str, np.ndarray]:
    def poses(value: float) -> np.ndarray:
        result = np.zeros((length, 7), dtype=np.float32)
        result[:, :3] = value
        result[:, 3] = 1.0
        return result

    return {
        "state.articulation.robot.root_pose": poses(offset),
        "state.articulation.robot.root_velocity": np.full((length, 6), 9.0, dtype=np.float32),
        "state.articulation.robot.joint_position": np.full((length, 23), offset + 1.0, dtype=np.float32),
        "state.articulation.robot.joint_velocity": np.full((length, 23), 9.0, dtype=np.float32),
        "state.rigid_object.object.root_pose": poses(offset + 2.0),
        "state.rigid_object.object.root_velocity": np.full((length, 6), 9.0, dtype=np.float32),
        "state.rigid_object.receptive_object.root_pose": poses(offset + 3.0),
        "state.rigid_object.receptive_object.root_velocity": np.full((length, 6), 9.0, dtype=np.float32),
        "state.rigid_object.table.root_pose": poses(offset + 4.0),
        "state.rigid_object.table.root_velocity": np.full((length, 6), 9.0, dtype=np.float32),
        "complete": np.asarray(True),
        "source_env_id": np.asarray(source_env_id, dtype=np.int64),
    }


def _write_dataset(path: Path, episodes: list[dict[str, np.ndarray]], metadata: dict | None = None) -> None:
    path.mkdir()
    (path / "metadata.json").write_text(json.dumps(_metadata() if metadata is None else metadata))
    for index, episode in enumerate(episodes):
        np.savez_compressed(path / f"{index:010d}.npz", **episode)


def test_reset_pool_loads_all_complete_and_partial_frames_and_zeros_velocities(tmp_path: Path) -> None:
    from src.tasks.clean_table_omnireset.mdps.reset_dataset import load_reset_state_pool

    first = _episode(length=2, offset=10.0)
    second = _episode(length=1, offset=20.0)
    second["complete"] = np.asarray(False)
    _write_dataset(tmp_path / "dataset", [first, second])

    pool = load_reset_state_pool(tmp_path / "dataset", ROBOT_JOINT_NAMES, device="cpu")

    assert pool.num_states == 3
    state = pool.scene_state(torch.tensor([2, 0]))
    torch.testing.assert_close(
        state["articulation"]["robot"]["joint_position"][0],
        torch.full((23,), 21.0),
    )
    assert not torch.any(state["articulation"]["robot"]["joint_velocity"])
    for asset_name in ("object", "receptive_object", "table"):
        assert not torch.any(state["rigid_object"][asset_name]["root_velocity"])


def test_reset_pool_rejects_incompatible_metadata_and_source_environment(tmp_path: Path) -> None:
    from src.tasks.clean_table_omnireset.mdps.reset_dataset import load_reset_state_pool

    wrong_task = _metadata()
    wrong_task["task"] = "Pick_Insert_HRL-v0"
    _write_dataset(tmp_path / "wrong_task", [_episode(1, 0.0)], wrong_task)
    with pytest.raises(ValueError, match="Clean_Table_HRL-v0"):
        load_reset_state_pool(tmp_path / "wrong_task", ROBOT_JOINT_NAMES, device="cpu")

    _write_dataset(tmp_path / "wrong_env", [_episode(1, 0.0, source_env_id=1)])
    with pytest.raises(ValueError, match="source_env_id"):
        load_reset_state_pool(tmp_path / "wrong_env", ROBOT_JOINT_NAMES, device="cpu")


def test_reset_pool_rejects_bad_shapes_and_non_finite_values(tmp_path: Path) -> None:
    from src.tasks.clean_table_omnireset.mdps.reset_dataset import load_reset_state_pool

    bad_shape = _episode(1, 0.0)
    bad_shape["state.articulation.robot.joint_position"] = np.zeros((1, 22), dtype=np.float32)
    _write_dataset(tmp_path / "bad_shape", [bad_shape])
    with pytest.raises(ValueError, match="joint_position"):
        load_reset_state_pool(tmp_path / "bad_shape", ROBOT_JOINT_NAMES, device="cpu")

    non_finite = _episode(1, 0.0)
    non_finite["state.rigid_object.object.root_pose"][0, 0] = np.nan
    _write_dataset(tmp_path / "non_finite", [non_finite])
    with pytest.raises(ValueError, match="finite"):
        load_reset_state_pool(tmp_path / "non_finite", ROBOT_JOINT_NAMES, device="cpu")

    missing_complete = _episode(1, 0.0)
    missing_complete.pop("complete")
    _write_dataset(tmp_path / "missing_complete", [missing_complete])
    with pytest.raises(ValueError, match="complete"):
        load_reset_state_pool(tmp_path / "missing_complete", ROBOT_JOINT_NAMES, device="cpu")


def _fake_env() -> SimpleNamespace:
    robot = SimpleNamespace(
        data=SimpleNamespace(body_pos_w=torch.tensor([[[0.55, 0.0, 0.60]]], dtype=torch.float32))
    )
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.55, -0.35, 0.336]], dtype=torch.float32),
            root_lin_vel_w=torch.zeros(1, 3),
        )
    )
    box = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.55, -0.35, 0.271]], dtype=torch.float32),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        )
    )
    table = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.55, 0.0, 0.235]])))
    return SimpleNamespace(scene={"robot": robot, "object": object_asset, "receptive_object": box, "table": table})


def _load_task_mdps(monkeypatch):
    assets = SimpleNamespace(Articulation=object, RigidObject=object)

    class SceneEntityCfg:
        def __init__(self, name, body_names=None):
            self.name = name
            self.body_names = body_names
            self.body_ids = [0]

    math_module = SimpleNamespace(
        quat_apply_inverse=lambda quat, value: value,
        quat_inv=lambda quat: quat,
        quat_mul=lambda first, second: second,
        subtract_frame_transforms=lambda parent_pos, parent_quat, child_pos, child_quat: (
            child_pos - parent_pos,
            child_quat,
        ),
    )
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(SceneEntityCfg=SceneEntityCfg))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)

    path = REPO_ROOT / "src/tasks/clean_table_omnireset/mdps/task_mdps.py"
    spec = importlib.util.spec_from_file_location("clean_table_omnireset_task_mdps", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_rewards_and_success_are_progress_free(monkeypatch) -> None:
    mdp = _load_task_mdps(monkeypatch)

    env = _fake_env()
    object_cfg = SimpleNamespace(name="object")
    box_cfg = SimpleNamespace(name="receptive_object")
    table_cfg = SimpleNamespace(name="table")
    hand_cfg = SimpleNamespace(name="robot", body_ids=slice(None))

    torch.testing.assert_close(mdp.lift_reward(env, object_cfg=object_cfg, table_cfg=table_cfg), torch.ones(1))
    torch.testing.assert_close(
        mdp.transport_reward(env, object_cfg=object_cfg, box_cfg=box_cfg, table_cfg=table_cfg),
        torch.ones(1),
    )
    assert bool(mdp.inside_box(env, object_cfg=object_cfg, box_cfg=box_cfg)[0])
    assert bool(
        mdp.placed_and_released(env, object_cfg=object_cfg, box_cfg=box_cfg, hand_base_cfg=hand_cfg)[0]
    )

    env.scene["robot"].data.body_pos_w[0, 0] = env.scene["object"].data.root_pos_w[0]
    assert not bool(
        mdp.placed_and_released(env, object_cfg=object_cfg, box_cfg=box_cfg, hand_base_cfg=hand_cfg)[0]
    )
    env.scene["robot"].data.body_pos_w[0, 0] = torch.tensor([0.55, 0.0, 0.60])
    env.scene["object"].data.root_lin_vel_w[0, 0] = 0.051
    assert not bool(
        mdp.placed_and_released(env, object_cfg=object_cfg, box_cfg=box_cfg, hand_base_cfg=hand_cfg)[0]
    )


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def test_env_config_is_independent_rl_only_and_uses_split_controllers() -> None:
    source = ENV_CFG_PATH.read_text()
    tree = ast.parse(source)

    assert "from src.tasks.clean_table " not in source
    assert "from src.tasks.clean_table." not in source
    assert "import src.tasks.clean_table " not in source
    assert "import src.tasks.clean_table." not in source
    assert "104738/104738.usd" in source
    assert "CommandHandBaseCuroboMpcActionCfg" not in source
    assert "CleanTableTrajectory" not in source
    assert "LowLevelObsCfg" not in source

    actions = ast.unparse(_class(tree, "ActionsCfg"))
    assert "RelativeJointPositionActionCfg" in actions
    assert "joint_names=['panda_joint.*']" in actions
    assert "scale=0.1" in actions
    assert "EMAJointPositionToLimitsActionCfg" in actions
    assert "joint_names=['a_.*']" in actions
    assert "alpha=0.5" in actions
    assert "rescale_to_limits=True" in actions

    env_cfg = ast.unparse(_class(tree, "DexsuiteFrankaLeapCleanTableOmniResetEnvCfg"))
    assert "commands = None" in env_cfg
    assert "reset_dataset_dir" in env_cfg
    assert "self.decimation = 2" in env_cfg

    rewards = ast.unparse(_class(tree, "RewardsCfg"))
    assert "hand_base_cfg" in rewards

    events_source = (
        REPO_ROOT / "src/tasks/clean_table_omnireset/mdps/events.py"
    ).read_text()
    assert "env.cfg.reset_dataset_dir" in events_source
    assert "self.episode_length_s = 20.0" in env_cfg


def test_omnireset_task_is_registered_with_own_trainer_config() -> None:
    script = """
import gymnasium as gym
import src.tasks
spec = gym.spec('Clean_Table_OmniReset-v0')
assert spec.kwargs['env_cfg_entry_point'].endswith(
    'clean_table_omnireset.env_cfg:DexsuiteFrankaLeapCleanTableOmniResetEnvCfg'
)
assert spec.kwargs['rsl_rl_cfg_entry_point'].endswith(
    'clean_table_omnireset.rsl_rl_ppo_cfg:CleanTableOmniResetRslRlPpoCfg'
)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    trainer_source = (REPO_ROOT / "src/tasks/clean_table_omnireset/rsl_rl_ppo_cfg.py").read_text()
    assert 'experiment_name = "clean_table_omnireset"' in trainer_source
    assert '"policy": ["policy", "proprio", "perception"]' in trainer_source
