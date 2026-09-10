from __future__ import annotations

import ast
import unittest
from pathlib import Path

import gymnasium as gym


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = REPO_ROOT / "src" / "tasks" / "pick_anyrotate" / "env_cfg.py"
INSTANT_DEXTERITY_PATH = REPO_ROOT / "scripts" / "instant_dexterity.py"


class PickAnyRotateHrlRegistrationTest(unittest.TestCase):
    def test_hrl_task_is_registered(self) -> None:
        import src.tasks  # noqa: F401

        spec = gym.spec("Pick_AnyRotate_HRL-v0")
        self.assertTrue(spec.kwargs["env_cfg_entry_point"].endswith("DexsuiteFrankaLeapAnyRotateHrlEnvCfg"))
        self.assertTrue(spec.kwargs["rsl_rl_cfg_entry_point"].endswith("PickAnyRotateRslRlPpoCfg"))


class PickAnyRotateHrlConfigStructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(ENV_CFG_PATH.read_text())
        cls.classes = {node.name: node for node in ast.walk(cls.tree) if isinstance(node, ast.ClassDef)}
        common_env = ast.parse((REPO_ROOT / "src/tasks/common/env_cfg.py").read_text())
        cls.common_classes = {node.name: node for node in common_env.body if isinstance(node, ast.ClassDef)}
        common_obs = ast.parse((REPO_ROOT / "src/tasks/common/observations_cfg.py").read_text())
        cls.shared_low_level = next(node for node in common_obs.body
                                    if isinstance(node, ast.ClassDef) and node.name == "LowLevelObsCfg")

    def test_hrl_config_uses_trajectory_command_and_split_actions(self) -> None:
        command_cls = self.classes["HrlCommandsCfg"]
        command_assignments = [ast.unparse(node) for node in command_cls.body if isinstance(node, ast.Assign)]
        self.assertTrue(any("PickAnyRotateTrajectoryObjectAndHandBasePoseCommandCfg" in item for item in command_assignments))

        action_cls = self.common_classes["HrlActionsCfg"]
        hrl_source = ast.unparse(self.classes["DexsuiteFrankaLeapAnyRotateHrlEnvCfg"])
        self.assertIn("actions: HrlActionsCfg = HrlActionsCfg()", hrl_source)
        names = [
            target.id
            for node in action_cls.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        self.assertEqual(names, ["arm_action", "hand_action"])

    def test_low_level_observation_contract_matches_155_dim_checkpoint(self) -> None:
        low_level = self.shared_low_level
        self.assertEqual(ast.unparse(self.classes["LowLevelObsCfg"].bases[0]), "SharedLowLevelObsCfg")
        self.assertIn("self.observations.low_level = LowLevelObsCfg()", ast.unparse(self.tree))
        names = [
            target.id
            for node in low_level.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        self.assertEqual(
            names,
            [
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
            ],
        )
        term_dims = {
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
        self.assertEqual(sum(term_dims[name] for name in names), 155)

    def test_hrl_contact_sensors_filter_object_and_table(self) -> None:
        source = ENV_CFG_PATH.read_text()

        self.assertIn('"{ENV_REGEX_NS}/Object/baseLink",', source)
        self.assertIn('"{ENV_REGEX_NS}/Table",', source)
        external_loop = next(node for node in ast.walk(self.classes["LowLevelObsCfg"])
                             if isinstance(node, ast.For)
                             and "external_contact_mask" in ast.literal_eval(node.iter))
        self.assertEqual(ast.literal_eval(external_loop.iter), (
            "external_contact_mask", "external_contact_force_mag", "external_contact_pose",
        ))
        self.assertEqual(ast.unparse(external_loop.body[0]), "getattr(self, name).params['filter_indices'] = [1]")


class PickAnyRotateMarkerTest(unittest.TestCase):
    def test_multi_usd_marker_uses_one_usd_path(self) -> None:
        source = INSTANT_DEXTERITY_PATH.read_text()

        self.assertIn("if isinstance(usd_path, list):", source)
        self.assertIn("usd_path = usd_path[0]", source)
        self.assertIn("usd_path=usd_path", source)


if __name__ == "__main__":
    unittest.main()
