from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = REPO_ROOT / "src/tasks/rotate_object_once"
ENV_CFG_PATH = TASK_ROOT / "env_cfg.py"
COMMANDS_PATH = TASK_ROOT / "mdps/commands.py"
TRAJECTORY_PATH = TASK_ROOT / "mdps/trajectory.py"
TASK_MDPS_PATH = TASK_ROOT / "mdps/task_mdps.py"
PPO_CFG_PATH = TASK_ROOT / "rsl_rl_ppo_cfg.py"
TASKS_INIT_PATH = REPO_ROOT / "src/tasks/__init__.py"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def _assignments(class_node: ast.ClassDef) -> dict[str, ast.expr]:
    result = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node.value
    return result


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _load_trajectory_module():
    module_names = ("isaaclab", "isaaclab.utils", "isaaclab.utils.math")
    previous = {name: sys.modules.get(name) for name in module_names}
    previous_utils = sys.modules.pop("src.policy.high_level.utils", None)
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")
    for name in ("apply_delta_pose", "combine_frame_transforms", "subtract_frame_transforms"):
        setattr(isaaclab_math, name, lambda *args, **kwargs: None)
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.utils": isaaclab_utils,
            "isaaclab.utils.math": isaaclab_math,
        }
    )
    spec = importlib.util.spec_from_file_location("rotate_object_once_trajectory_under_test", TRAJECTORY_PATH)
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
        sys.modules.pop("src.policy.high_level.utils", None)
        if previous_utils is not None:
            sys.modules["src.policy.high_level.utils"] = previous_utils
    return module


def _load_commands_module():
    trajectory = _load_trajectory_module()
    module_names = (
        "isaaclab",
        "isaaclab.utils",
        "src.policy.high_level.trajectory_command",
        "src.tasks.rotate_object_once.mdps.trajectory",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls

    trajectory_command = types.ModuleType("src.policy.high_level.trajectory_command")

    class FakeTrajectoryCommand:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.num_envs = env.num_envs
            self.device = env.device
            self.metrics = {}

        def _resample_command(self, env_ids):
            self.resampled.append(env_ids.clone())
            self._stepper.step[env_ids] = 0

        def _apply_objanchor_correction(self, env_ids):
            self.corrected.append(env_ids.clone())

        def _update_hand_base_pose_command(self, env_ids):
            self.hand_base_updated.append(env_ids.clone())

    class FakeTrajectoryCommandCfg:
        def replace(self, **kwargs):
            replacement = type(self)()
            replacement.__dict__.update(self.__dict__)
            replacement.__dict__.update(kwargs)
            return replacement

    trajectory_command.TrajectoryObjectAndHandBasePoseCommand = FakeTrajectoryCommand
    trajectory_command.TrajectoryObjectAndHandBasePoseCommandCfg = FakeTrajectoryCommandCfg
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.utils": isaaclab_utils,
            "src.policy.high_level.trajectory_command": trajectory_command,
            "src.tasks.rotate_object_once.mdps.trajectory": trajectory,
        }
    )
    spec = importlib.util.spec_from_file_location("rotate_object_once_commands_under_test", COMMANDS_PATH)
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


def _load_task_mdps_module():
    module_names = (
        "isaaclab",
        "isaaclab.sim",
        "isaaclab.sim.utils",
        "isaaclab.sim.utils.stage",
        "isaaclab.utils",
        "isaaclab.utils.math",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    isaaclab = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    sim.find_matching_prim_paths = lambda _: ()
    sim_utils = types.ModuleType("isaaclab.sim.utils")
    stage_utils = types.ModuleType("isaaclab.sim.utils.stage")
    stage_utils.get_current_stage = lambda: None
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.sim": sim,
            "isaaclab.sim.utils": sim_utils,
            "isaaclab.sim.utils.stage": stage_utils,
            "isaaclab.utils": isaaclab_utils,
            "isaaclab.utils.math": isaaclab_math,
        }
    )
    spec = importlib.util.spec_from_file_location("rotate_object_once_task_mdps_under_test", TASK_MDPS_PATH)
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


def test_standalone_package_has_no_rotate_object_task_imports() -> None:
    source_files = sorted(TASK_ROOT.rglob("*.py"))

    assert source_files
    for path in source_files:
        assert "src.tasks.rotate_object." not in path.read_text(), path


def test_final_yaw_goal_latches_success_without_resampling() -> None:
    module = _load_commands_module()

    class Stepper:
        length = 2

        def __init__(self):
            self.step = torch.tensor([0, 1, 1])

        def advance(self, env_ids):
            self.step[env_ids] += 1

        def current_object_pose(self, env_ids):
            result = torch.zeros(env_ids.numel(), 7)
            result[:, 3] = 1.0
            return result

    class Correction:
        def clear_stall(self, env_ids):
            pass

    command = object.__new__(module.RotateObjectOnceTrajectoryObjectAndHandBasePoseCommand)
    command._stepper = Stepper()
    command._corr = Correction()
    command.pose_command_b = torch.zeros(3, 7)
    command._trajectory_command_achieved = torch.tensor([True, True, False])
    command.metrics = {
        "trajectory_command_achieved": torch.tensor([1.0, 1.0, 0.0]),
        "yaw_target_active": torch.zeros(3),
    }
    command.resampled = []
    command.corrected = []
    command.hand_base_updated = []

    command._update_command()

    assert torch.equal(command._stepper.step, torch.ones(3, dtype=torch.long))
    assert command.resampled == []
    assert torch.equal(command._trajectory_command_achieved, torch.tensor([False, True, False]))
    assert torch.equal(command.metrics["trajectory_command_achieved"], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(command.metrics["yaw_target_active"], torch.ones(3))
    assert torch.equal(command.corrected[0], torch.tensor([0, 1, 2]))
    assert torch.equal(command.hand_base_updated[0], torch.tensor([0, 1, 2]))


def test_yaw_target_achieved_requires_active_achieved_target() -> None:
    module = _load_task_mdps_module()
    command = types.SimpleNamespace(
        metrics={
            "yaw_target_active": torch.tensor([0.0, 1.0, 1.0, 0.0]),
            "trajectory_command_achieved": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        }
    )
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: command if name == "object_pose" else None)
    )

    success = module.yaw_target_achieved(env, command_name="object_pose")

    assert success.dtype == torch.bool
    assert torch.equal(success, torch.tensor([False, False, True, False]))


def test_reset_restores_robot_default_joint_pose() -> None:
    env_tree = ast.parse(ENV_CFG_PATH.read_text())
    events = _assignments(_class(env_tree, "EventCfg"))
    reset_robot_joints = events["reset_robot_joints"]

    assert ast.unparse(_keyword(reset_robot_joints, "func")) == "mdp.reset_joints_by_offset"
    assert ast.literal_eval(_keyword(reset_robot_joints, "mode")) == "reset"
    assert ast.literal_eval(_keyword(reset_robot_joints, "params")) == {
        "position_range": [0.0, 0.0],
        "velocity_range": [0.0, 0.0],
    }


def test_config_registration_and_trainer_are_standalone() -> None:
    env_tree = ast.parse(ENV_CFG_PATH.read_text())
    terminations = _assignments(_class(env_tree, "TerminationsCfg"))
    success = terminations["success"]

    assert ast.unparse(_keyword(success, "func")) == "mdp.yaw_target_achieved"
    assert ast.literal_eval(_keyword(success, "params")) == {"command_name": "object_pose"}
    assert _class(env_tree, "DexsuiteFrankaLeapRotateObjectOnceEnvCfg")
    assert _class(env_tree, "DexsuiteFrankaLeapRotateObjectOnceHrlEnvCfg")

    registrations = TASKS_INIT_PATH.read_text()
    assert 'id="Rotate_Object_Once-v0"' in registrations
    assert 'id="Rotate_Object_Once_HRL-v0"' in registrations
    assert "rotate_object_once.env_cfg:" in registrations
    assert "DexsuiteFrankaLeapRotateObjectOnceEnvCfg" in registrations
    assert "DexsuiteFrankaLeapRotateObjectOnceHrlEnvCfg" in registrations
    assert "src.tasks.rotate_object_once.rsl_rl_ppo_cfg:" in registrations
    assert "RotateObjectOnceRslRlPpoCfg" in registrations

    trainer = _assignments(_class(ast.parse(PPO_CFG_PATH.read_text()), "RotateObjectOnceRslRlPpoCfg"))
    assert ast.literal_eval(trainer["experiment_name"]) == "rotate_object_once"
