from __future__ import annotations

import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = REPO_ROOT / "src/tasks/pick_insert_omnireset/env_cfg.py"


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
    "state.articulation.robot.joint_position",
    "state.articulation.robot.joint_velocity",
    "state.articulation.robot.root_pose",
    "state.articulation.robot.root_velocity",
    "state.rigid_object.object.root_pose",
    "state.rigid_object.object.root_velocity",
    "state.rigid_object.receptive_object.root_pose",
    "state.rigid_object.receptive_object.root_velocity",
    "state.rigid_object.static_obstacle.root_pose",
    "state.rigid_object.static_obstacle.root_velocity",
    "state.rigid_object.table.root_pose",
    "state.rigid_object.table.root_velocity",
]


def _metadata() -> dict:
    return {
        "schema_version": 1,
        "task": "Pick_Insert_HRL-v0",
        "num_envs": 1,
        "joint_order": {"articulations": {"robot": ROBOT_JOINT_NAMES}},
        "scene_state": {"relative_to_env_origin": True, "fields": STATE_FIELDS},
    }


def _poses(length: int, xyz: tuple[float, float, float]) -> np.ndarray:
    poses = np.zeros((length, 7), dtype=np.float32)
    poses[:, :3] = xyz
    poses[:, 3] = 1.0
    return poses


def _episode() -> dict[str, np.ndarray]:
    length = 2
    object_pose = _poses(length, (0.45, 0.20, 0.30))
    # Frame one is exactly at the configured insertion target and must be filtered.
    object_pose[1, :3] = (0.35, 0.0, 0.29)
    return {
        "state.articulation.robot.root_pose": _poses(length, (0.0, 0.0, 0.0)),
        "state.articulation.robot.root_velocity": np.ones((length, 6), dtype=np.float32),
        "state.articulation.robot.joint_position": np.arange(length * 23, dtype=np.float32).reshape(length, 23),
        "state.articulation.robot.joint_velocity": np.ones((length, 23), dtype=np.float32),
        "state.rigid_object.object.root_pose": object_pose,
        "state.rigid_object.object.root_velocity": np.ones((length, 6), dtype=np.float32),
        "state.rigid_object.receptive_object.root_pose": _poses(length, (0.35, 0.0, 0.275)),
        "state.rigid_object.receptive_object.root_velocity": np.ones((length, 6), dtype=np.float32),
        "state.rigid_object.static_obstacle.root_pose": _poses(length, (0.5, -0.25, 0.38)),
        "state.rigid_object.static_obstacle.root_velocity": np.ones((length, 6), dtype=np.float32),
        "state.rigid_object.table.root_pose": _poses(length, (0.55, 0.0, 0.235)),
        "state.rigid_object.table.root_velocity": np.ones((length, 6), dtype=np.float32),
        "complete": np.asarray(True),
        "source_env_id": np.asarray(0, dtype=np.int64),
    }


def _write_dataset(path: Path, episode: dict[str, np.ndarray], metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True)
    (path.parent / "metadata.json").write_text(json.dumps(_metadata() if metadata is None else metadata))
    np.savez_compressed(path, **episode)


def test_reset_pool_loads_only_named_archive_filters_success_and_zeros_velocities(tmp_path: Path) -> None:
    from src.tasks.pick_insert_omnireset.mdps.reset_dataset import load_reset_state_pool

    archive = tmp_path / "dataset" / "0000000000.npz"
    _write_dataset(archive, _episode())
    # An invalid sibling proves the loader does not glob the directory.
    np.savez_compressed(archive.parent / "0000000001.npz", unexpected=np.zeros(1))

    pool = load_reset_state_pool(archive, ROBOT_JOINT_NAMES, device="cpu")

    assert pool.num_states == 1
    state = pool.scene_state(torch.tensor([0]))
    torch.testing.assert_close(
        state["articulation"]["robot"]["joint_position"][0],
        torch.arange(23, dtype=torch.float32),
    )
    assert not torch.any(state["articulation"]["robot"]["joint_velocity"])
    for asset_name in ("object", "receptive_object", "static_obstacle", "table"):
        assert not torch.any(state["rigid_object"][asset_name]["root_velocity"])


def test_reset_pool_rejects_metadata_source_and_invalid_values(tmp_path: Path) -> None:
    from src.tasks.pick_insert_omnireset.mdps.reset_dataset import load_reset_state_pool

    wrong_task = _metadata()
    wrong_task["task"] = "Clean_Table_HRL-v0"
    wrong_task_path = tmp_path / "wrong_task" / "0000000000.npz"
    _write_dataset(wrong_task_path, _episode(), wrong_task)
    with pytest.raises(ValueError, match="Pick_Insert_HRL-v0"):
        load_reset_state_pool(wrong_task_path, ROBOT_JOINT_NAMES, device="cpu")

    wrong_source = _episode()
    wrong_source["source_env_id"] = np.asarray(1, dtype=np.int64)
    wrong_source_path = tmp_path / "wrong_source" / "0000000000.npz"
    _write_dataset(wrong_source_path, wrong_source)
    with pytest.raises(ValueError, match="source_env_id"):
        load_reset_state_pool(wrong_source_path, ROBOT_JOINT_NAMES, device="cpu")

    non_finite = _episode()
    non_finite["state.rigid_object.object.root_pose"][0, 0] = np.nan
    non_finite_path = tmp_path / "non_finite" / "0000000000.npz"
    _write_dataset(non_finite_path, non_finite)
    with pytest.raises(ValueError, match="finite"):
        load_reset_state_pool(non_finite_path, ROBOT_JOINT_NAMES, device="cpu")


def test_reset_pool_rejects_joint_order_and_field_shape(tmp_path: Path) -> None:
    from src.tasks.pick_insert_omnireset.mdps.reset_dataset import load_reset_state_pool

    wrong_order = _metadata()
    wrong_order["joint_order"]["articulations"]["robot"] = list(reversed(ROBOT_JOINT_NAMES))
    wrong_order_path = tmp_path / "wrong_order" / "0000000000.npz"
    _write_dataset(wrong_order_path, _episode(), wrong_order)
    with pytest.raises(ValueError, match="joint order"):
        load_reset_state_pool(wrong_order_path, ROBOT_JOINT_NAMES, device="cpu")

    bad_shape = _episode()
    bad_shape["state.articulation.robot.joint_position"] = np.zeros((2, 22), dtype=np.float32)
    bad_shape_path = tmp_path / "bad_shape" / "0000000000.npz"
    _write_dataset(bad_shape_path, bad_shape)
    with pytest.raises(ValueError, match="joint_position"):
        load_reset_state_pool(bad_shape_path, ROBOT_JOINT_NAMES, device="cpu")


def test_reset_pool_rejects_archive_with_only_inserted_states(tmp_path: Path) -> None:
    from src.tasks.pick_insert_omnireset.mdps.reset_dataset import load_reset_state_pool

    episode = _episode()
    for key, value in list(episode.items()):
        if isinstance(value, np.ndarray) and value.ndim == 2:
            episode[key] = value[1:]
    archive = tmp_path / "inserted" / "0000000000.npz"
    _write_dataset(archive, episode)

    with pytest.raises(ValueError, match="unfinished"):
        load_reset_state_pool(archive, ROBOT_JOINT_NAMES, device="cpu")


def _load_task_mdps(monkeypatch):
    assets = SimpleNamespace(Articulation=object, RigidObject=object)

    class SceneEntityCfg:
        def __init__(self, name, body_names=None):
            self.name = name
            self.body_names = body_names

    def subtract_frame_transforms(parent_pos, parent_quat, child_pos, child_quat):
        return child_pos - parent_pos, child_quat

    math_module = SimpleNamespace(
        quat_apply=lambda quat, value: value,
        quat_apply_inverse=lambda quat, value: value,
        quat_inv=lambda quat: quat,
        quat_mul=lambda first, second: second,
        subtract_frame_transforms=subtract_frame_transforms,
    )
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", SimpleNamespace(SceneEntityCfg=SceneEntityCfg))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)

    path = REPO_ROOT / "src/tasks/pick_insert_omnireset/mdps/task_mdps.py"
    spec = importlib.util.spec_from_file_location("pick_insert_omnireset_task_mdps", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_hole_geometry_and_success_use_same_thresholds(monkeypatch) -> None:
    mdp = _load_task_mdps(monkeypatch)
    object_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.35, 0.0, 0.29]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    hole_asset = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.35, 0.0, 0.275]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.zeros(1, 3),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )
    )
    env = SimpleNamespace(
        num_envs=1,
        scene={"robot": robot, "object": object_asset, "receptive_object": hole_asset},
    )
    object_cfg = SimpleNamespace(name="object")
    hole_cfg = SimpleNamespace(name="receptive_object")
    robot_cfg = SimpleNamespace(name="robot")

    torch.testing.assert_close(
        mdp.object_pose_hole(env, object_cfg=object_cfg, hole_cfg=hole_cfg),
        torch.tensor([[0.0, 0.0, 0.015, 1.0, 0.0, 0.0, 0.0]]),
    )
    torch.testing.assert_close(
        mdp.hole_pose_b(env, robot_cfg=robot_cfg, hole_cfg=hole_cfg),
        torch.tensor([[0.35, 0.0, 0.275, 1.0, 0.0, 0.0, 0.0]]),
    )
    assert bool(mdp.peg_inserted_success(env, object_cfg=object_cfg, hole_cfg=hole_cfg)[0])


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def test_env_config_is_independent_direct_rl_and_uses_split_controllers() -> None:
    source = ENV_CFG_PATH.read_text()
    tree = ast.parse(source)

    assert "from src.tasks.pick_insert " not in source
    assert "from src.tasks.pick_insert." not in source
    assert "import src.tasks.pick_insert " not in source
    assert "import src.tasks.pick_insert." not in source
    assert "static_obstacle" in source
    assert "CommandHandBaseCuroboMpcActionCfg" not in source
    assert "PickInsertTrajectory" not in source
    assert "LowLevelObsCfg" not in source

    actions = ast.unparse(_class(tree, "ActionsCfg"))
    assert "RelativeJointPositionActionCfg" in actions
    assert "joint_names=['panda_joint.*']" in actions
    assert "scale=0.1" in actions
    assert "EMAJointPositionToLimitsActionCfg" in actions
    assert "joint_names=['a_.*']" in actions
    assert "alpha=0.5" in actions
    assert "rescale_to_limits=True" in actions

    policy = ast.unparse(_class(tree, "PolicyCfg"))
    assert "hole_pose_b" in policy
    assert "object_pose_hole" in policy
    env_cfg = ast.unparse(_class(tree, "DexsuiteFrankaLeapPickInsertOmniResetEnvCfg"))
    assert "commands = None" in env_cfg
    assert "curriculum = None" in env_cfg
    assert "reset_dataset_path" in env_cfg
    assert "0000000000.npz" in source
    assert "self.decimation = 4" in env_cfg

    events = ast.unparse(_class(tree, "EventCfg"))
    assert "reset_from_dataset" in events
    assert "reset_object" not in events
    terminations = ast.unparse(_class(tree, "TerminationsCfg"))
    assert "peg_inserted_success" in terminations


def test_reset_event_samples_pool_and_restores_relative_scene_state() -> None:
    event_path = REPO_ROOT / "src/tasks/pick_insert_omnireset/mdps/events.py"
    source = event_path.read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert "env.cfg.reset_dataset_path" in source
    assert any(
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "torch"
        and call.func.attr == "randint"
        for call in calls
    )
    reset_to = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "reset_to"
    )
    keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in reset_to.keywords}
    assert keywords["env_ids"] == "env_ids_t"
    assert keywords["is_relative"] == "True"


def test_omnireset_task_is_registered_with_own_trainer_config() -> None:
    script = """
import gymnasium as gym
import src.tasks
spec = gym.spec('Pick_Insert_OmniReset-v0')
assert spec.kwargs['env_cfg_entry_point'].endswith(
    'pick_insert_omnireset.env_cfg:DexsuiteFrankaLeapPickInsertOmniResetEnvCfg'
)
assert spec.kwargs['rsl_rl_cfg_entry_point'].endswith(
    'pick_insert_omnireset.rsl_rl_ppo_cfg:PickInsertOmniResetRslRlPpoCfg'
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

    trainer_source = (REPO_ROOT / "src/tasks/pick_insert_omnireset/rsl_rl_ppo_cfg.py").read_text()
    assert 'experiment_name = "pick_insert_omnireset"' in trainer_source
    assert '"policy": ["policy", "proprio", "perception"]' in trainer_source
