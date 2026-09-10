from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = REPO_ROOT / "src/tasks/clean_table/env_cfg.py"
COMMON_ENV_CFG_PATH = REPO_ROOT / "src/tasks/common/env_cfg.py"
COMMON_OBS_CFG_PATH = REPO_ROOT / "src/tasks/common/observations_cfg.py"


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
    assert "low_level: LowLevelObsCfg = LowLevelObsCfg()" in ast.unparse(tree)
    terms = _assignments(_class(ast.parse(COMMON_OBS_CFG_PATH.read_text()), "LowLevelObsCfg"))
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
    common_tree = ast.parse(COMMON_ENV_CFG_PATH.read_text())
    actions = _assignments(_class(common_tree, "HrlActionsCfg"))

    assert ast.unparse(command.func).endswith("CleanTableTrajectoryObjectAndHandBasePoseCommandCfg")
    assert ast.literal_eval(_keyword(command, "asset_name")) == "robot"
    assert ast.literal_eval(_keyword(command, "object_name")) == "object"
    assert ast.literal_eval(_keyword(command, "box_name")) == "receptive_object"
    assert ast.literal_eval(_keyword(command, "hand_base_hold_until_stage")) == -1
    assert set(actions) == {"arm_action", "hand_action"}
    assert ast.unparse(actions["arm_action"].func).endswith("CommandHandBaseCuroboMpcActionCfg")
    assert ast.literal_eval(_keyword(actions["arm_action"], "command_name")) == "object_pose"
    assert ast.unparse(actions["hand_action"].func).endswith("EMAJointPositionToLimitsActionCfg")
    hrl_source = ast.unparse(_class(tree, "DexsuiteFrankaLeapCleanTableHrlEnvCfg"))
    assert "low_level: LowLevelObsCfg = LowLevelObsCfg()" in ast.unparse(tree)
    assert "self.commands.object_pose.hand_base_hold_until_stage = 1" in hrl_source
    assert "self.commands.object_pose.drop_object_hand_distance = 0.12" in hrl_source
    assert "self.commands.object_pose.lift_height = 0.1" in hrl_source
    assert hrl_source.count("StageObjTol(0.02, 0.3)") == 2
    assert "*self.commands.object_pose.stage_object_tolerances[2:]" in hrl_source
    assert "self.decimation = 4" in hrl_source
    assert "self.episode_length_s = 30.0" in hrl_source
    assert "self.sim.render_interval = self.decimation" in hrl_source
    assert "configure_contact_physics(self.sim)" in hrl_source
    physics_fn = next(node for node in common_tree.body if isinstance(node, ast.FunctionDef)
                      and node.name == "configure_contact_physics")
    namespace = {}
    exec(compile(ast.Module(body=[physics_fn], type_ignores=[]), str(COMMON_ENV_CFG_PATH), "exec"), namespace)
    sim = SimpleNamespace(physx=SimpleNamespace())
    namespace["configure_contact_physics"](sim)
    assert sim.physx.gpu_found_lost_aggregate_pairs_capacity == 2**23
    assert sim.physx.gpu_total_aggregate_pairs_capacity == 2**23
    assert sim.physx.gpu_max_rigid_contact_count == 2**23
    assert sim.physx.gpu_max_rigid_patch_count == 2**23
    assert sim.physx.gpu_collision_stack_size == 2**31
    assert "self.scene.robot.actuators['joints'].stiffness = 0.0" in hrl_source
    assert "self.scene.robot.actuators['joints'].damping = 0.0" in hrl_source
    assert [ast.unparse(base) for base in _class(tree, "DexsuiteFrankaLeapCleanTableHrlEnvCfg").bases] == ["ManagerBasedRLEnvCfg"]
    assert "rewards = None" in hrl_source
    assert "curriculum = None" in hrl_source


def test_clean_table_uses_pick_anyrotate_scene_and_reset_baseline() -> None:
    tree = ast.parse(ENV_CFG_PATH.read_text())
    scene = _assignments(_class(tree, "SceneCfg"))
    common_tree = ast.parse(COMMON_ENV_CFG_PATH.read_text())
    assert "events: TabletopEventsCfg = TabletopEventsCfg()" in ast.unparse(tree)
    events = _assignments(_class(common_tree, "TabletopEventsCfg"))
    box_params = _keyword(events["reset_receptive_object"], "params")
    object_params = _keyword(events["reset_object"], "params")
    box_pose = _dict_value(box_params, "pose_range")
    object_pose = _dict_value(object_params, "pose_range")

    assert tuple(ast.literal_eval(_dict_value(box_pose, "x"))) == (-0.10, 0.10)
    assert tuple(ast.literal_eval(_dict_value(box_pose, "y"))) == (-0.05, 0.05)
    assert ast.literal_eval(object_pose) == {
        "x": [-0.03, 0.03],
        "y": [-0.03, 0.03],
        "yaw": [0.0, 0.0],
    }

    assert [ast.unparse(base) for base in _class(tree, "SceneCfg").bases] == ["TabletopSceneCfg"]
    shared_scene = _assignments(_class(common_tree, "TabletopSceneCfg"))
    assert ast.unparse(shared_scene["robot"]) == "franka_robot_cfg()"
    robot_fn = next(node for node in common_tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "franka_robot_cfg")
    robot_cfg = next(node.value for node in robot_fn.body if isinstance(node, ast.Return))
    robot_spawn = _keyword(robot_cfg, "spawn")
    articulation_props = _keyword(robot_spawn, "articulation_props")
    assert ast.literal_eval(_keyword(articulation_props, "enabled_self_collisions")) is False

    object_cfg = scene["object"]
    object_spawn = _keyword(scene["object"], "spawn")
    assert ast.literal_eval(_keyword(object_spawn, "random_choice")) is False
    mass_props = _keyword(object_spawn, "mass_props")
    assert ast.literal_eval(_keyword(mass_props, "mass")) == 0.2
    object_init = _keyword(object_cfg, "init_state")
    assert tuple(ast.literal_eval(_keyword(object_init, "pos"))) == (0.55, 0.10, 0.34)
    usd_root = REPO_ROOT / "src/assets/visdex_objects/USD"
    usd_paths = [path / f"{path.name}.usd" for path in sorted(usd_root.iterdir()) if path.is_dir()]
    assert len([path for path in usd_paths if path.is_file()]) == 152
    assert usd_paths[0].parts[-2:] == ("104738", "104738.usd")

    mass_params = _keyword(events["object_scale_mass"], "params")
    assert tuple(ast.literal_eval(_dict_value(mass_params, "mass_distribution_params"))) == (0.2, 2.0)
    assert ast.literal_eval(_dict_value(mass_params, "operation")) == "scale"
    gravity_params = _keyword(events["variable_gravity"], "params")
    assert ast.literal_eval(_dict_value(gravity_params, "gravity_distribution_params")) == (
        [0.0, 0.0, -1.81],
        [0.0, 0.0, -1.81],
    )


def test_clean_table_contact_filter_roles_are_stable() -> None:
    tree = ast.parse(ENV_CFG_PATH.read_text())
    setup = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "configure_fingertip_contacts")
    assert ast.literal_eval(setup.args[1]) == [
        "{ENV_REGEX_NS}/Object/baseLink*",
        "{ENV_REGEX_NS}/ReceptiveObject",
        "{ENV_REGEX_NS}/Table",
    ]
    observations = _assignments(_class(ast.parse(COMMON_OBS_CFG_PATH.read_text()), "LowLevelObsCfg"))
    for name, indices in (("contact_pose", [0]), ("external_contact_pose", [1, 2])):
        assert ast.literal_eval(_dict_value(_keyword(observations[name], "params"), "filter_indices")) == indices


def test_clean_table_hrl_task_is_registered() -> None:
    script = """
import gymnasium as gym
import src.tasks
spec = gym.spec('Clean_Table_HRL-v0')
assert spec.kwargs['env_cfg_entry_point'].endswith(
    'clean_table.env_cfg:DexsuiteFrankaLeapCleanTableHrlEnvCfg'
)
assert 'rsl_rl_cfg_entry_point' not in spec.kwargs
assert 'Clean_Table-v0' not in gym.registry
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


def test_instant_dexterity_reports_the_term_that_reset_the_scene() -> None:
    source = (REPO_ROOT / "scripts/instant_dexterity.py").read_text()

    assert "observations, reward, terminated, truncated, _ = env.step(actions)" in source
    assert "termination_manager.active_terms" in source
    assert "termination_manager.get_term(term_name)" in source
    assert "scene reset" in source


def test_instant_dexterity_supports_deterministic_clean_table_diagnostics() -> None:
    source = (REPO_ROOT / "scripts/instant_dexterity.py").read_text()
    tree = ast.parse(source)

    assert 'parser.add_argument("--seed", type=int, default=None' in source
    assert source.index("env_cfg.seed = args_cli.seed") < source.index("env = gym.make")
    clean_table_if = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Constant) and child.value == "inside_box"
            for child in ast.walk(node.test)
        )
    )
    clean_table_source = ast.unparse(ast.Module(body=clean_table_if.body, type_ignores=[]))
    assert "clean-table grasp" in clean_table_source
    for field in ("stage=", "contact=", "streak=", "phase=", "keep_open=", "height=", "lifted="):
        assert field in clean_table_source
