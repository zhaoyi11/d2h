from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_trajectory_module():
    utils_name = "src.policy.high_level.utils"
    prior_utils = sys.modules.get(utils_name)
    utils = _load_file(
        "unscrew_test_utils",
        REPO_ROOT / "src/policy/high_level/utils.py",
    )
    sys.modules[utils_name] = utils
    try:
        return _load_file(
            "unscrew_trajectory_under_test",
            REPO_ROOT / "src/tasks/unscrew/mdps/trajectory.py",
        )
    finally:
        if prior_utils is None:
            sys.modules.pop(utils_name, None)
        else:
            sys.modules[utils_name] = prior_utils


@contextmanager
def _temporary_modules(modules: dict[str, types.ModuleType]):
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qw = quat[:, :1]
    qvec = quat[:, 1:]
    uv = torch.cross(qvec, vector, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return vector + 2.0 * (qw * uv + uuv)


def _load_task_mdps_module():
    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self._env = env

    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    assets = types.ModuleType("isaaclab.assets")
    assets.RigidObject = object
    managers = types.ModuleType("isaaclab.managers")
    managers.ManagerTermBase = ManagerTermBase
    managers.SceneEntityCfg = SceneEntityCfg
    managers.TerminationTermCfg = object
    math_module = types.ModuleType("isaaclab.utils.math")
    math_module.quat_apply = _quat_apply
    math_module.subtract_frame_transforms = (
        lambda rpos, rquat, opos, oquat: (opos - rpos, oquat)
    )
    rewards = types.ModuleType("src.tasks.common.mdps.rewards")
    rewards.contacts = lambda env, threshold: env.good_contact
    with _temporary_modules(
        {
            "isaaclab.assets": assets,
            "isaaclab.managers": managers,
            "isaaclab.utils.math": math_module,
            "src.tasks.common.mdps.rewards": rewards,
        }
    ):
        return _load_file(
            "unscrew_task_mdps_under_test",
            REPO_ROOT / "src/tasks/unscrew/mdps/task_mdps.py",
        )


def _load_commands_module():
    stepper = _load_file(
        "unscrew_stepper_dependency",
        REPO_ROOT / "src/policy/high_level/trajectory_stepper.py",
    )
    trajectory = _load_trajectory_module()

    class BaseCommand:
        def _env_ids_tensor(self, env_ids):
            return torch.as_tensor(env_ids, dtype=torch.long)

        def _resample_command(self, env_ids):
            return None

        def _update_command(self):
            return None

    class BaseCommandCfg:
        pass

    trajectory_command = types.ModuleType("src.policy.high_level.trajectory_command")
    trajectory_command.TrajectoryObjectAndHandBasePoseCommand = BaseCommand
    trajectory_command.TrajectoryObjectAndHandBasePoseCommandCfg = BaseCommandCfg
    stepper_module = types.ModuleType("src.policy.high_level.trajectory_stepper")
    stepper_module.StageObjTol = stepper.StageObjTol
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = lambda cls: cls
    math_module = types.ModuleType("isaaclab.utils.math")
    math_module.subtract_frame_transforms = lambda *args: (args[2] - args[0], args[3])
    rewards = types.ModuleType("src.tasks.common.mdps.rewards")
    rewards.contacts = lambda env, threshold: env.good_contact
    task_mdps = types.ModuleType("src.tasks.unscrew.mdps.task_mdps")
    task_mdps.bolt_aabb_corners = lambda device: torch.zeros(8, 3, device=device)
    task_mdps.bolt_bottom_clearance = lambda *args: torch.zeros(args[0].shape[0])
    with _temporary_modules(
        {
            "isaaclab.utils": utils,
            "isaaclab.utils.math": math_module,
            "src.policy.high_level.trajectory_command": trajectory_command,
            "src.policy.high_level.trajectory_stepper": stepper_module,
            "src.tasks.common.mdps.rewards": rewards,
            "src.tasks.unscrew.mdps.task_mdps": task_mdps,
            "src.tasks.unscrew.mdps.trajectory": trajectory,
        }
    ):
        return _load_file(
            "unscrew_commands_under_test",
            REPO_ROOT / "src/tasks/unscrew/mdps/commands.py",
        )


def test_stepper_transition_replaces_only_selected_environment() -> None:
    module = _load_file(
        "trajectory_stepper_under_test",
        REPO_ROOT / "src/policy/high_level/trajectory_stepper.py",
    )
    stepper = module.TrajectoryStepper(
        2,
        "cpu",
        (0, 1, 2, 1, 1),
        torch.ones(5),
        torch.ones(5),
    )
    original = torch.arange(2 * stepper.length * 7, dtype=torch.float32).reshape(
        2, stepper.length, 7
    )
    stepper.reset(torch.tensor([0, 1]), original)
    replacement = torch.full((1, stepper.length - 4, 7), 99.0)

    stepper.transition_to_step(torch.tensor([1]), 4, replacement)

    assert stepper.step.tolist() == [0, 4]
    torch.testing.assert_close(stepper.current_object_pose(torch.tensor([1])), replacement[:, 0])
    torch.testing.assert_close(stepper._trajectory[0], original[0])
    torch.testing.assert_close(stepper._trajectory[1, :4], original[1, :4])
    torch.testing.assert_close(stepper._trajectory[1, 4:], replacement[0])


def test_stepper_transition_validates_remaining_shape() -> None:
    module = _load_file(
        "trajectory_stepper_shape_under_test",
        REPO_ROOT / "src/policy/high_level/trajectory_stepper.py",
    )
    stepper = module.TrajectoryStepper(1, "cpu", (1, 1), torch.ones(2), torch.ones(2))

    with pytest.raises(ValueError, match="remaining_waypoints"):
        stepper.transition_to_step(torch.tensor([0]), 1, torch.zeros(1, 1, 7))


def test_vertical_lift_starts_from_live_pose_and_freezes_orientation() -> None:
    module = _load_trajectory_module()
    current = torch.tensor([0.4, -0.2, 0.3, 0.5, 0.5, 0.5, 0.5])

    trajectory = module.build_unscrew_vertical_lift_pose_sequence(
        current,
        segment_steps=(2, 1),
        extraction_height=0.1,
    )

    assert trajectory.shape == (4, 7)
    torch.testing.assert_close(trajectory[0], current)
    torch.testing.assert_close(trajectory[:, :2], current[:2].expand(4, -1))
    torch.testing.assert_close(trajectory[:, 3:7], current[3:7].expand(4, -1))
    torch.testing.assert_close(trajectory[:, 2], torch.tensor([0.3, 0.35, 0.4, 0.4]))


def test_bolt_clearance_uses_rotated_bounds() -> None:
    module = _load_task_mdps_module()
    half = 2.0**-0.5
    object_pos_r = torch.tensor([[0.0, 0.0, 0.08], [0.0, 0.0, 0.08]])
    object_quat_r = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [half, 0.0, half, 0.0]]
    )
    corners = module.bolt_aabb_corners("cpu")

    clearance = module.bolt_bottom_clearance(object_pos_r, object_quat_r, corners)

    assert clearance[0] < 0.0
    assert clearance[1] > clearance[0]


def test_success_is_immediate_when_clear_and_grasped() -> None:
    module = _load_task_mdps_module()
    object_data = SimpleNamespace(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.2]]),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    receptive_data = SimpleNamespace(
        root_pos_w=torch.zeros(1, 3),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        scene={
            "object": SimpleNamespace(data=object_data),
            "receptive_object": SimpleNamespace(data=receptive_data),
        },
        good_contact=torch.ones(1, dtype=torch.bool),
    )
    cfg = SimpleNamespace(
        params={
            "object_cfg": module.SceneEntityCfg("object"),
            "receptive_cfg": module.SceneEntityCfg("receptive_object"),
        }
    )
    term = module.StableUnscrewSuccess(cfg, env)

    assert bool(term(env)[0])


def test_clearance_streak_requires_twist_clearance_and_grasp() -> None:
    commands = _load_commands_module()
    command = object.__new__(commands.UnscrewTrajectoryObjectAndHandBasePoseCommand)
    command.cfg = SimpleNamespace(
        enable_clearance_transition=True,
        twist_segments=2,
        clearance_transition_margin=0.005,
        clearance_transition_stable_steps=3,
        grasp_contact_force_threshold=1.0,
    )
    command.num_envs = 2
    command._stepper = SimpleNamespace(
        step=torch.tensor([2, 2]),
        step_to_stage=torch.tensor([0, 1, 2]),
    )
    command._clearance_streak = torch.zeros(2, dtype=torch.long)
    command._clearance_transition_pending = torch.zeros(2, dtype=torch.bool)
    command._clearance_transitioned = torch.zeros(2, dtype=torch.bool)
    command.metrics = {
        "bolt_clearance": torch.tensor([0.006, 0.006]),
        "clearance_transitioned": torch.zeros(2),
        "vertical_lift_complete": torch.zeros(2),
    }
    command._compute_bolt_clearance = lambda: torch.tensor([0.006, 0.006])
    command._env = SimpleNamespace(good_contact=torch.tensor([True, False]))

    command._update_clearance_transition()
    command._update_clearance_transition()
    assert command._clearance_transition_pending.tolist() == [False, False]
    command._update_clearance_transition()

    assert command._clearance_transition_pending.tolist() == [True, False]
    assert command._clearance_streak.tolist() == [3, 0]

    command._stepper.step_to_stage = torch.tensor([0, 1, 4])
    command._update_clearance_transition()
    assert command._clearance_streak.tolist() == [0, 0]


def test_transition_uses_live_pose_and_jumps_to_lift() -> None:
    commands = _load_commands_module()
    command = object.__new__(commands.UnscrewTrajectoryObjectAndHandBasePoseCommand)
    command.cfg = SimpleNamespace(
        twist_segments=2,
        trajectory_segment_steps=(0, 1, 2, 2, 1, 1),
        extraction_height=0.1,
    )
    command._clearance_transition_pending = torch.tensor([True, False])
    command._clearance_transitioned = torch.zeros(2, dtype=torch.bool)
    command._trajectory_command_achieved = torch.ones(2, dtype=torch.bool)
    command._grasp_stall_counter = torch.ones(2, dtype=torch.long)
    command.metrics = {"clearance_transitioned": torch.zeros(2)}
    command.pose_command_b = torch.zeros(2, 7)
    command.pose_command_b[:, 3] = 1.0
    live_pos = torch.tensor([[0.4, -0.2, 0.3], [0.0, 0.0, 0.0]])
    live_quat = torch.tensor([[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]])
    command._current_object_pose_b = lambda env_ids: (live_pos[env_ids], live_quat[env_ids])

    class Stepper:
        def __init__(self):
            self.call = None

        def transition_to_step(self, env_ids, step, remaining_waypoints):
            self.call = (env_ids.clone(), step, remaining_waypoints.clone())

        def current_object_pose(self, env_ids):
            assert self.call is not None
            return self.call[2][:, 0]

    class Correction:
        def __init__(self):
            self.reset_ids = None

        def reset(self, env_ids):
            self.reset_ids = env_ids.clone()

    command._stepper = Stepper()
    command._corr = Correction()

    command._transition_cleared_to_lift()

    env_ids, step, remaining = command._stepper.call
    assert env_ids.tolist() == [0]
    assert step == 6
    torch.testing.assert_close(remaining[:, -1, :2], live_pos[0, :2].reshape(1, 2))
    torch.testing.assert_close(remaining[:, -1, 2], torch.tensor([0.4]))
    torch.testing.assert_close(remaining[:, :, 3:7], live_quat[0].reshape(1, 1, 4).expand_as(remaining[:, :, 3:7]))
    assert not bool(command._trajectory_command_achieved[0])
    assert bool(command._trajectory_command_achieved[1])
    assert command._clearance_transitioned.tolist() == [True, False]
    assert command._corr.reset_ids.tolist() == [0]


def test_normal_trajectory_can_complete_without_early_transition() -> None:
    commands = _load_commands_module()
    command = object.__new__(commands.UnscrewTrajectoryObjectAndHandBasePoseCommand)
    command.cfg = SimpleNamespace(twist_segments=2)
    command._stepper = SimpleNamespace(
        step=torch.tensor([0, 1]),
        step_to_stage=torch.tensor([5, 4]),
    )
    command._trajectory_command_achieved = torch.tensor([True, True])
    command.metrics = {"vertical_lift_complete": torch.zeros(2)}

    command._update_vertical_lift_complete()

    assert command.metrics["vertical_lift_complete"].tolist() == [1.0, 0.0]


def test_resample_clears_only_selected_transition_state() -> None:
    commands = _load_commands_module()
    command = object.__new__(commands.UnscrewTrajectoryObjectAndHandBasePoseCommand)
    command._clearance_streak = torch.tensor([2, 3])
    command._clearance_transition_pending = torch.tensor([True, True])
    command._clearance_transitioned = torch.tensor([True, True])
    command._grasp_contact_streak = torch.tensor([2, 2])
    command._grasp_phase_steps = torch.tensor([4, 4])
    command.metrics = {
        "bolt_clearance": torch.ones(2),
        "clearance_transitioned": torch.ones(2),
        "vertical_lift_complete": torch.ones(2),
    }

    command._resample_command([1])

    assert command._clearance_streak.tolist() == [2, 0]
    assert command._clearance_transition_pending.tolist() == [True, False]
    assert command._clearance_transitioned.tolist() == [True, False]
    assert command.metrics["bolt_clearance"].tolist() == [1.0, 0.0]
    assert command.metrics["vertical_lift_complete"].tolist() == [1.0, 0.0]


def test_hrl_config_enables_transition_and_immediate_clearance_success() -> None:
    source = (REPO_ROOT / "src/tasks/unscrew/env_cfg.py").read_text()

    assert "enable_clearance_transition=True" in source
    assert "clearance_transition_margin=0.005" in source
    assert "clearance_transition_stable_steps=1" in source
    assert "require_lift_complete=True" not in source
