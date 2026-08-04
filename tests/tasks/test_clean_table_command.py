from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_commands_module():
    module_names = (
        "isaaclab",
        "isaaclab.assets",
        "isaaclab.utils",
        "isaaclab.utils.math",
        "src.policy.high_level.trajectory_command",
        "src.tasks.common.mdps.rewards",
        "src.tasks.clean_table.mdps",
        "src.tasks.clean_table.mdps.trajectory",
    )
    previous = {name: sys.modules.get(name) for name in module_names}

    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    utils = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")
    assets.RigidObject = object
    utils.configclass = lambda cls: cls
    math_module.subtract_frame_transforms = (
        lambda root_pos, root_quat, body_pos, body_quat: (body_pos - root_pos, body_quat)
    )

    trajectory_command = types.ModuleType("src.policy.high_level.trajectory_command")

    class BaseCommand:
        def _update_metrics(self):
            self._trajectory_command_achieved[:] = True

        def _object_target_achieved(self):
            return self._base_object_achieved.clone()

        def _apply_objanchor_correction(self, env_ids):
            self.corrected_env_ids = env_ids.clone()

        def _anchor_correction(self, env_ids=slice(None)):
            return self._base_correction[env_ids]

    class BaseCommandCfg:
        pass

    trajectory_command.TrajectoryObjectAndHandBasePoseCommand = BaseCommand
    trajectory_command.TrajectoryObjectAndHandBasePoseCommandCfg = BaseCommandCfg

    common_rewards = types.ModuleType("src.tasks.common.mdps.rewards")
    common_rewards.contacts = lambda env, threshold: torch.zeros(
        env.num_envs, dtype=torch.bool
    )

    mdps_package = types.ModuleType("src.tasks.clean_table.mdps")
    mdps_package.__path__ = []
    trajectory = types.ModuleType("src.tasks.clean_table.mdps.trajectory")
    trajectory.DEFAULT_BOX_RETREAT_OFFSET = (0.0, 0.0, 0.25)
    trajectory.DEFAULT_BOX_TARGET_OFFSET = (0.0, 0.0, 0.065)
    trajectory.DEFAULT_CLEAN_TABLE_SEGMENT_STEPS = (0, 1, 1, 2, 2, 12, 2)
    trajectory.DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES = tuple(
        SimpleNamespace(object_position=0.02, object_orientation=0.3) for _ in range(7)
    )
    trajectory.build_clean_table_object_pose_sequence = lambda *args, **kwargs: torch.zeros(1, 7)

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.assets"] = assets
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_module
    sys.modules["src.policy.high_level.trajectory_command"] = trajectory_command
    sys.modules["src.tasks.common.mdps.rewards"] = common_rewards
    sys.modules["src.tasks.clean_table.mdps"] = mdps_package
    sys.modules["src.tasks.clean_table.mdps.trajectory"] = trajectory

    spec = importlib.util.spec_from_file_location(
        "clean_table_commands_under_test",
        REPO_ROOT / "src/tasks/clean_table/mdps/commands.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


commands = _load_commands_module()


def _command_with_stages(stages: torch.Tensor):
    command = object.__new__(commands.CleanTableTrajectoryObjectAndHandBasePoseCommand)
    command._stepper = SimpleNamespace(
        step=torch.arange(stages.numel()),
        step_to_stage=stages,
    )
    command.cfg = SimpleNamespace(hand_open_until_stage=0)
    command.metrics = {
        "keep_hand_open": torch.zeros(stages.numel()),
    }
    command._base_object_achieved = torch.zeros(stages.numel(), dtype=torch.bool)
    command._base_correction = torch.ones(stages.numel(), 6)
    return command


def test_grasp_release_and_retreat_ignore_anchor_correction() -> None:
    command = _command_with_stages(torch.tensor([0, 1, 2, 5, 6]))

    achieved = command._object_target_achieved()
    command._update_keep_hand_open_metric()
    command._apply_objanchor_correction(torch.arange(5))

    torch.testing.assert_close(achieved, torch.tensor([False, False, False, True, True]))
    torch.testing.assert_close(
        command.metrics["keep_hand_open"], torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])
    )
    torch.testing.assert_close(command.corrected_env_ids, torch.tensor([0, 2]))


def test_grasp_release_and_retreat_do_not_apply_accumulated_anchor_correction() -> None:
    command = _command_with_stages(torch.tensor([1, 4, 5, 6]))

    correction = command._anchor_correction()

    torch.testing.assert_close(correction[1], torch.ones(6))
    torch.testing.assert_close(correction[[0, 2, 3]], torch.zeros(3, 6))


def test_success_requires_stable_containment_after_retreat_and_hand_clearance() -> None:
    command = _command_with_stages(torch.tensor([6, 6]))
    command.cfg = SimpleNamespace(
        capture_goal_after_settle=False,
        grasp_contact_force_threshold=1.0,
        grasp_contact_stable_steps=3,
        grasp_timeout_steps=30,
        table_half_height=0.02,
        lift_height=0.08,
        hand_clear_distance=0.10,
        success_settle_speed=0.05,
        success_stable_steps=2,
    )
    command._env = SimpleNamespace(num_envs=2)
    command._trajectory_command_achieved = torch.zeros(2, dtype=torch.bool)
    command._grasp_goal_captured = torch.ones(2, dtype=torch.bool)
    command._grasp_phase_steps = torch.zeros(2, dtype=torch.long)
    command._grasp_contact_streak = torch.zeros(2, dtype=torch.long)
    command._success_streak = torch.zeros(2, dtype=torch.long)
    command.object_height_above_table = torch.zeros(2)
    command.object_to_box_distance = torch.zeros(2)
    command.object_pos_box = torch.zeros(2, 3)
    command.lifted = torch.zeros(2, dtype=torch.bool)
    command.inside_box = torch.zeros(2, dtype=torch.bool)
    command.released = torch.zeros(2, dtype=torch.bool)
    command.hand_clear = torch.zeros(2, dtype=torch.bool)
    command.success = torch.zeros(2, dtype=torch.bool)
    command._box_target = torch.tensor([0.0, 0.0, 0.065])
    command._box_min = torch.tensor([-0.09, -0.15, 0.005])
    command._box_max = torch.tensor([0.09, 0.15, 0.105])
    command.metrics.update(
        {
            "hand_base_object_error": torch.tensor([0.20, 0.20]),
            "trajectory_command_achieved": torch.zeros(2),
        }
    )
    command.table = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.0]] * 2)))
    command.box = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.5, 0.0, 0.25]] * 2),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
        )
    )
    command.object = SimpleNamespace(
        data=SimpleNamespace(
            root_pos_w=torch.tensor([[0.5, 0.0, 0.315], [0.7, 0.0, 0.315]]),
            root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
            root_lin_vel_w=torch.zeros(2, 3),
        )
    )

    command._update_metrics()
    torch.testing.assert_close(command.success, torch.tensor([False, False]))
    command._update_metrics()

    torch.testing.assert_close(command.inside_box, torch.tensor([True, False]))
    torch.testing.assert_close(command.released, torch.tensor([True, True]))
    torch.testing.assert_close(command.hand_clear, torch.tensor([True, True]))
    torch.testing.assert_close(command.success, torch.tensor([True, False]))
