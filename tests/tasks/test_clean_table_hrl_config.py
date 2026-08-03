from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = REPO_ROOT / "src/tasks/clean_table/env_cfg.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def _assignments(class_node: ast.ClassDef) -> dict[str, ast.expr]:
    result = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node.value
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node.value
    return result


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _dict_value(node: ast.Dict, key: str) -> ast.expr:
    for dict_key, value in zip(node.keys, node.values, strict=True):
        if isinstance(dict_key, ast.Constant) and dict_key.value == key:
            return value
    raise AssertionError(f"Missing dictionary key {key!r}.")


def test_clean_table_low_level_observation_contract_is_155_dimensional() -> None:
    tree = ast.parse(ENV_CFG_PATH.read_text())
    terms = _assignments(_class(tree, "LowLevelObsCfg"))
    widths = {
        "joint_pos": 16,
        "joint_vel": 16,
        "fingertip_pose": 52,
        "contact_mask": 4,
        "contact_force_mag": 4,
        "contact_pose": 8,
        "external_contact_mask": 4,
        "external_contact_force_mag": 4,
        "external_contact_pose": 8,
        "object_pos": 3,
        "object_quat": 4,
        "object_lin_vel": 3,
        "object_ang_vel": 3,
        "gravity_dir": 3,
        "goal_pos_diff": 3,
        "goal_quat_diff": 4,
        "last_action": 16,
    }

    assert list(terms) == list(widths)
    assert sum(widths.values()) == 155
    for name in ("goal_pos_diff", "goal_quat_diff"):
        params = _keyword(terms[name], "params")
        assert ast.literal_eval(_dict_value(params, "command_name")) == "object_pose"
    last_action_params = _keyword(terms["last_action"], "params")
    assert ast.literal_eval(_dict_value(last_action_params, "action_name")) == "hand_action"


def test_clean_table_hrl_actions_and_command_match_runner_contract() -> None:
    tree = ast.parse(ENV_CFG_PATH.read_text())
    command = _assignments(_class(tree, "CommandsCfg"))["object_pose"]
    actions = _assignments(_class(tree, "HrlActionsCfg"))

    assert ast.unparse(command.func).endswith("CleanTableTrajectoryObjectAndHandBasePoseCommandCfg")
    assert ast.literal_eval(_keyword(command, "asset_name")) == "robot"
    assert ast.literal_eval(_keyword(command, "object_name")) == "object"
    assert ast.literal_eval(_keyword(command, "box_name")) == "receptive_object"
    assert set(actions) == {"arm_action", "hand_action"}
    assert ast.unparse(actions["arm_action"].func).endswith("CommandHandBaseCuroboMpcActionCfg")
    assert ast.literal_eval(_keyword(actions["arm_action"], "command_name")) == "object_pose"
    assert ast.unparse(actions["hand_action"].func).endswith("EMAJointPositionToLimitsActionCfg")
    hrl_source = ast.unparse(_class(tree, "DexsuiteFrankaLeapCleanTableHrlEnvCfg"))
    assert "self.observations.low_level = ObservationsCfg.LowLevelObsCfg()" in hrl_source
    assert "self.scene.robot.actuators['joints'].stiffness = 0.0" in hrl_source
    assert "self.scene.robot.actuators['joints'].damping = 0.0" in hrl_source


def test_clean_table_resets_keep_object_and_box_in_separate_regions() -> None:
    tree = ast.parse(ENV_CFG_PATH.read_text())
    events = _assignments(_class(tree, "EventCfg"))
    box_params = _keyword(events["reset_receptive_object"], "params")
    object_params = _keyword(events["reset_object"], "params")
    box_pose = _dict_value(box_params, "pose_range")
    object_pose = _dict_value(object_params, "pose_range")

    assert tuple(ast.literal_eval(_dict_value(box_pose, "x"))) == (-0.10, 0.10)
    assert tuple(ast.literal_eval(_dict_value(box_pose, "y"))) == (-0.05, 0.05)
    assert tuple(ast.literal_eval(_dict_value(object_pose, "x"))) == (-0.05, 0.05)
    assert tuple(ast.literal_eval(_dict_value(object_pose, "y"))) == (-0.05, 0.05)

    mass_params = _keyword(events["object_scale_mass"], "params")
    assert tuple(ast.literal_eval(_dict_value(mass_params, "mass_distribution_params"))) == (0.010, 0.100)
    assert ast.literal_eval(_dict_value(mass_params, "operation")) == "abs"
    gravity_params = _keyword(events["variable_gravity"], "params")
    assert ast.literal_eval(_dict_value(gravity_params, "gravity_distribution_params")) == (
        [0.0, 0.0, -1.81],
        [0.0, 0.0, -1.81],
    )


def test_clean_table_contact_filter_roles_are_stable() -> None:
    path = REPO_ROOT / "src/tasks/clean_table/mdps/contact_filters.py"
    spec = importlib.util.spec_from_file_location("clean_table_contact_filters_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.object_indices() == [0]
    assert module.external_indices() == [1, 2]
    assert module.contact_filter_prim_paths()[0].endswith("/Object/.*")


def test_clean_table_hrl_task_is_registered() -> None:
    script = """
import gymnasium as gym
import src.tasks
spec = gym.spec('Clean_Table_HRL-v0')
assert spec.kwargs['env_cfg_entry_point'].endswith(
    'clean_table.env_cfg:DexsuiteFrankaLeapCleanTableHrlEnvCfg'
)
assert spec.kwargs['rsl_rl_cfg_entry_point'].endswith('CleanTableRslRlPpoCfg')
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


def test_instant_dexterity_reports_clean_table_success_metrics() -> None:
    source = (REPO_ROOT / "scripts/instant_dexterity.py").read_text()
    assert 'if "inside_box" in getattr(command_term, "metrics", {}):' in source
    assert "clean-table placement" in source
    assert "success={bool(metrics['success'][0])}" in source
