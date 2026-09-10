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
        "isaaclab.managers",
        "isaaclab.utils",
        "isaaclab.utils.math",
        "src.policy.high_level.trajectory_command",
        "src.tasks.common.mdps.contacts",
        "src.tasks.common.mdps.placement",
        "src.tasks.clean_table.mdps",
        "src.tasks.clean_table.mdps.trajectory",
    )
    previous = {name: sys.modules.get(name) for name in module_names}

    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    utils = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")
    assets.RigidObject = object
    assets.Articulation = object
    managers = types.ModuleType("isaaclab.managers")
    managers.SceneEntityCfg = lambda name: SimpleNamespace(name=name)
    utils.configclass = lambda cls: cls
    math_module.quat_apply_inverse = lambda quat, value: value
    math_module.quat_inv = lambda quat: quat
    math_module.quat_mul = lambda first, second: second
    math_module.subtract_frame_transforms = (
        lambda root_pos, root_quat, body_pos, body_quat: (body_pos - root_pos, body_quat)
    )

    trajectory_command = types.ModuleType("src.policy.high_level.trajectory_command")

    class BaseCommand:
        def _current_hand_base_pose_b(self, env_ids=slice(None)):
            pose = self._base_current_hand_pose_b[env_ids]
            return pose[:, :3], pose[:, 3:7]

        def _current_object_pose_b(self, env_ids=slice(None)):
            pose = self._base_current_object_pose_b[env_ids]
            return pose[:, :3], pose[:, 3:7]

        def _update_hand_base_pose_command(self, env_ids=slice(None)):
            self.hand_base_pose_command_b[env_ids] = self._base_hand_base_pose[env_ids]

        def _update_metrics(self):
            self._trajectory_command_achieved[:] = getattr(
                self, "_base_trajectory_achieved", True
            )

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

    common_rewards = types.ModuleType("src.tasks.common.mdps.contacts")
    common_rewards.contacts = lambda env, threshold: torch.zeros(
        env.num_envs, dtype=torch.bool
    )

    mdps_package = types.ModuleType("src.tasks.clean_table.mdps")
    mdps_package.__path__ = []
    trajectory = types.ModuleType("src.tasks.clean_table.mdps.trajectory")
    trajectory.DEFAULT_BOX_ABOVE_OFFSET = (0.0, 0.0, 0.25)
    trajectory.DEFAULT_BOX_TARGET_OFFSET = (0.0, 0.0, 0.065)
    trajectory.DEFAULT_CLEAN_TABLE_SEGMENT_STEPS = (0, 1, 3, 2, 12, 2)
    trajectory.DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES = tuple(
        SimpleNamespace(object_position=0.02, object_orientation=0.3) for _ in range(6)
    )
    trajectory.build_clean_table_object_pose_sequence = lambda *args, **kwargs: torch.zeros(1, 7)

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.assets"] = assets
    sys.modules["isaaclab.managers"] = managers
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_module
    sys.modules["src.policy.high_level.trajectory_command"] = trajectory_command
    sys.modules["src.tasks.common.mdps.contacts"] = common_rewards
    sys.modules["src.tasks.clean_table.mdps"] = mdps_package
    sys.modules["src.tasks.clean_table.mdps.trajectory"] = trajectory

    placement_spec = importlib.util.spec_from_file_location(
        "src.tasks.common.mdps.placement",
        REPO_ROOT / "src/tasks/common/mdps/placement.py",
    )
    placement = importlib.util.module_from_spec(placement_spec)

    spec = importlib.util.spec_from_file_location(
        "clean_table_commands_under_test",
        REPO_ROOT / "src/tasks/clean_table/mdps/commands.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        sys.modules[placement_spec.name] = placement
        placement_spec.loader.exec_module(placement)
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module, placement


commands, placement = _load_commands_module()


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
    command = _command_with_stages(torch.tensor([0, 1, 2, 4, 5]))

    achieved = command._object_target_achieved()
    command._update_keep_hand_open_metric()
    command._apply_objanchor_correction(torch.arange(5))

    torch.testing.assert_close(achieved, torch.tensor([False, False, False, True, True]))
    torch.testing.assert_close(
        command.metrics["keep_hand_open"], torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0])
    )
    torch.testing.assert_close(command.corrected_env_ids, torch.tensor([0, 2]))


def test_grasp_keeps_applied_correction_while_release_and_retreat_zero_it() -> None:
    command = _command_with_stages(torch.tensor([1, 3, 4, 5]))

    correction = command._anchor_correction()

    torch.testing.assert_close(correction[:2], torch.ones(2, 6))
    torch.testing.assert_close(correction[2:], torch.zeros(2, 6))


def test_retreat_targets_default_hand_base_pose() -> None:
    command = _command_with_stages(torch.tensor([3, 5]))
    command.hand_base_pose_command_b = torch.zeros(2, 7)
    command._base_hand_base_pose = torch.tensor(
        [[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0]]
    )
    command._default_hand_base_pose_b = torch.tensor(
        [[0.7, 0.8, 0.9, 1.0, 0.0, 0.0, 0.0], [1.0, 1.1, 1.2, 1.0, 0.0, 0.0, 0.0]]
    )

    command._update_hand_base_pose_command()

    torch.testing.assert_close(command.hand_base_pose_command_b[0], command._base_hand_base_pose[0])
    torch.testing.assert_close(command.hand_base_pose_command_b[1], command._default_hand_base_pose_b[1])


def test_grasp_confirmation_matches_pick_anyrotate_contact_streak() -> None:
    command = _command_with_stages(torch.tensor([1]))
    command.cfg = SimpleNamespace(
        capture_goal_after_settle=True,
        grasp_contact_force_threshold=1.0,
        grasp_contact_stable_steps=3,
        grasp_timeout_steps=30,
    )
    command._env = SimpleNamespace(num_envs=1)
    command._trajectory_command_achieved = torch.zeros(1, dtype=torch.bool)
    command._grasp_goal_captured = torch.zeros(1, dtype=torch.bool)
    command._grasp_phase_steps = torch.zeros(1, dtype=torch.long)
    command._grasp_contact_streak = torch.tensor([2], dtype=torch.long)
    command._steps_since_reset = torch.zeros(1, dtype=torch.long)
    previous_contacts = placement.good_object_contact
    placement.good_object_contact = lambda env, threshold: torch.ones(1, dtype=torch.bool)
    try:
        command._update_grasp_establish()
    finally:
        placement.good_object_contact = previous_contacts

    torch.testing.assert_close(command._grasp_contact_streak, torch.tensor([3]))
    torch.testing.assert_close(command._trajectory_command_achieved, torch.tensor([True]))


def test_success_requires_stable_containment_after_retreat_and_hand_clearance() -> None:
    command = _command_with_stages(torch.tensor([5, 5]))
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
    command._base_trajectory_achieved = torch.zeros(2, dtype=torch.bool)
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
            "hand_base_object_error": torch.zeros(2),
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
    command._base_current_hand_pose_b = torch.tensor(
        [[0.0, 0.0, 0.50, 1.0, 0.0, 0.0, 0.0]] * 2
    )
    command._base_current_object_pose_b = torch.cat(
        (command.object.data.root_pos_w, command.object.data.root_quat_w), dim=1
    )

    command._update_metrics()
    torch.testing.assert_close(command.success, torch.tensor([False, False]))
    command._base_trajectory_achieved[:] = True
    command._update_metrics()

    torch.testing.assert_close(command.success, torch.tensor([False, False]))
    command._update_metrics()

    torch.testing.assert_close(command.inside_box, torch.tensor([True, False]))
    torch.testing.assert_close(command.released, torch.tensor([True, True]))
    torch.testing.assert_close(command.hand_clear, torch.tensor([True, True]))
    torch.testing.assert_close(command.success, torch.tensor([True, False]))
