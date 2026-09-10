import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_isaaclab_stubs():
    module_names = [
        "isaaclab",
        "isaaclab.envs",
        "isaaclab.assets",
        "isaaclab.managers",
        "isaaclab.utils",
        "isaaclab.utils.math",
        "src.tasks.common.obj_point_cloud",
        "src.tasks.common.mdps.contacts",
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_envs = types.ModuleType("isaaclab.envs")
    isaaclab_assets = types.ModuleType("isaaclab.assets")
    isaaclab_managers = types.ModuleType("isaaclab.managers")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")

    isaaclab_envs.ManagerBasedRLEnv = object
    isaaclab_assets.Articulation = object
    isaaclab_assets.RigidObject = object

    class ManagerTermBase:
        def __init__(self, cfg, env):
            pass

    class SceneEntityCfg:
        def __init__(self, name, body_ids=None, **kwargs):
            self.name = name
            self.body_ids = body_ids
            self.__dict__.update(kwargs)

    isaaclab_managers.ManagerTermBase = ManagerTermBase
    isaaclab_managers.SceneEntityCfg = SceneEntityCfg

    def quat_apply(quat, vec):
        return vec

    def quat_apply_inverse(quat, vec):
        return vec

    def quat_inv(quat):
        return quat

    def quat_mul(lhs, rhs):
        return rhs

    def subtract_frame_transforms(root_pos, root_quat, body_pos, body_quat):
        return body_pos - root_pos, body_quat

    def combine_frame_transforms(parent_pos, parent_quat, child_pos, child_quat):
        return parent_pos + child_pos, child_quat

    isaaclab_math.combine_frame_transforms = combine_frame_transforms
    isaaclab_math.quat_apply = quat_apply
    isaaclab_math.quat_apply_inverse = quat_apply_inverse
    isaaclab_math.quat_inv = quat_inv
    isaaclab_math.quat_mul = quat_mul
    isaaclab_math.subtract_frame_transforms = subtract_frame_transforms
    isaaclab_utils.math = isaaclab_math
    isaaclab.utils = isaaclab_utils

    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.envs"] = isaaclab_envs
    sys.modules["isaaclab.assets"] = isaaclab_assets
    sys.modules["isaaclab.managers"] = isaaclab_managers
    sys.modules["isaaclab.utils"] = isaaclab_utils
    sys.modules["isaaclab.utils.math"] = isaaclab_math

    obj_point_cloud = types.ModuleType("src.tasks.common.obj_point_cloud")
    obj_point_cloud.sample_object_point_cloud = lambda *args, **kwargs: None
    sys.modules["src.tasks.common.obj_point_cloud"] = obj_point_cloud

    return previous_modules


def _restore_modules(previous_modules):
    for name, module in previous_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_observations_module():
    previous_modules = _install_isaaclab_stubs()
    spec = importlib.util.spec_from_file_location(
        "test_observations_module",
        REPO_ROOT / "src" / "tasks" / "common" / "mdps" / "observations.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        contact_spec = importlib.util.spec_from_file_location(
            "src.tasks.common.mdps.contacts",
            REPO_ROOT / "src/tasks/common/mdps/contacts.py",
        )
        contacts = importlib.util.module_from_spec(contact_spec)
        contact_spec.loader.exec_module(contacts)
        sys.modules[contact_spec.name] = contacts
        spec.loader.exec_module(module)
    finally:
        _restore_modules(previous_modules)
    return module


class BodyStateObservationTest(unittest.TestCase):
    def _make_env(self):
        num_envs = 2
        body_pos_w = torch.tensor(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
            ]
        )
        body_quat_w = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]],
            ]
        )
        body_lin_vel_w = torch.tensor(
            [
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                [[1.1, 1.2, 1.3], [1.4, 1.5, 1.6], [1.7, 1.8, 1.9]],
            ]
        )
        body_ang_vel_w = torch.tensor(
            [
                [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]],
                [[-1.1, -1.2, -1.3], [-1.4, -1.5, -1.6], [-1.7, -1.8, -1.9]],
            ]
        )
        root_link_pos_w = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

        robot = SimpleNamespace(
            data=SimpleNamespace(
                body_pos_w=body_pos_w,
                body_quat_w=body_quat_w,
                body_lin_vel_w=body_lin_vel_w,
                body_ang_vel_w=body_ang_vel_w,
                root_link_pos_w=root_link_pos_w,
                root_link_quat_w=root_link_quat_w,
            )
        )
        env = SimpleNamespace(num_envs=num_envs, scene={"robot": robot})
        cfg = SimpleNamespace(name="robot", body_ids=[0, 2])
        return env, cfg

    def test_split_body_state_observations_match_packed_state_slices(self):
        mdp = _load_observations_module()
        env, cfg = self._make_env()

        pos = mdp.body_pos_b(env, cfg, cfg)
        quat = mdp.body_quat_b(env, cfg, cfg)
        lin_vel = mdp.body_lin_vel_b(env, cfg, cfg)
        ang_vel = mdp.body_ang_vel_b(env, cfg, cfg)
        packed = mdp.body_state_b(env, cfg, cfg)

        self.assertEqual(pos.shape, (2, 6))
        self.assertEqual(quat.shape, (2, 8))
        self.assertEqual(lin_vel.shape, (2, 6))
        self.assertEqual(ang_vel.shape, (2, 6))
        self.assertEqual(packed.shape, (2, 26))

        state_by_body = packed.view(env.num_envs, 2, 13)
        torch.testing.assert_close(pos, state_by_body[:, :, 0:3].reshape(env.num_envs, -1))
        torch.testing.assert_close(quat, state_by_body[:, :, 3:7].reshape(env.num_envs, -1))
        torch.testing.assert_close(lin_vel, state_by_body[:, :, 7:10].reshape(env.num_envs, -1))
        torch.testing.assert_close(ang_vel, state_by_body[:, :, 10:13].reshape(env.num_envs, -1))

    def test_object_pose_b_concatenates_position_and_quaternion(self):
        mdp = _load_observations_module()
        self.assertTrue(hasattr(mdp, "object_pose_b"))
        robot = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2.0, 3.0]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            )
        )
        receptive_object = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[1.5, 2.5, 3.5]]),
                root_quat_w=torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
            )
        )
        env = SimpleNamespace(
            num_envs=1,
            scene={"robot": robot, "receptive_object": receptive_object},
        )

        pose = mdp.object_pose_b(
            env,
            object_cfg=SimpleNamespace(name="receptive_object"),
        )

        torch.testing.assert_close(
            pose,
            torch.tensor([[0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0]]),
        )

    def test_object_body_frame_observations_use_selected_body_frame(self):
        mdp = _load_observations_module()
        env, cfg = self._make_env()
        obj = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[5.0, 7.0, 9.0], [7.0, 9.0, 11.0]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
                root_lin_vel_w=torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
                root_ang_vel_w=torch.tensor([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]]),
            )
        )
        env.scene["object"] = obj
        base_cfg = SimpleNamespace(name="robot", body_ids=[1])

        pos = mdp.object_pos_body_b(env, body_asset_cfg=base_cfg)
        lin_vel = mdp.object_lin_vel_body_b(env, body_asset_cfg=base_cfg)

        torch.testing.assert_close(pos, torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]))
        torch.testing.assert_close(lin_vel, torch.tensor([[0.6, 1.5, 2.4], [1.6, 2.5, 3.4]]))

    def test_goal_pos_diff_body_b_uses_desired_in_hand_pose_for_object_and_hand_base_commands(self):
        mdp = _load_observations_module()
        env, _ = self._make_env()
        obj = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[5.0, 7.0, 9.0], [7.0, 9.0, 11.0]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            )
        )
        command = torch.zeros(2, 14)
        command[:, 3] = 1.0
        command[:, 10] = 1.0
        command[:, :3] = torch.tensor([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])
        # The hand-base target is deliberately far away. Low-level hand control
        # should use command[:7] as a desired in-hand pose and leave this transit
        # error to the arm IK action.
        command[:, 7:10] = torch.tensor([[0.5, 0.0, 0.0], [-0.5, 1.0, 0.0]])
        env.scene["object"] = obj
        env.command_manager = SimpleNamespace(get_command=lambda name: command)
        base_cfg = SimpleNamespace(name="robot", body_ids=[1])

        pos_diff = mdp.goal_pos_diff_body_b(
            env,
            asset_cfg=SimpleNamespace(name="object"),
            command_name="object_pose",
            body_asset_cfg=base_cfg,
        )

        torch.testing.assert_close(pos_diff, torch.tensor([[0.8, 1.7, 2.6], [1.5, 2.4, 3.3]]))


class ReorientObservationConfigTest(unittest.TestCase):
    def test_fingertip_pose_uses_packed_body_state(self):
        tree = ast.parse((REPO_ROOT / "src" / "tasks" / "reorient" / "env_cfg.py").read_text())
        assignments = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name) and target.id.startswith("fingertip")
        }

        self.assertEqual(self._obs_term_func_name(assignments["fingertip_pose"]), ("mdp", "body_state_b"))
        self.assertEqual(self._obs_term_func_name(assignments["fingertip_contact_force_b"]), ("mdp", "fingers_contact_force_b"))

    def test_observation_noise_matches_current_reorient_config(self):
        tree = ast.parse((REPO_ROOT / "src" / "tasks" / "reorient" / "env_cfg.py").read_text())
        assignments = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }

        expected_gaussian_noise_terms = {
            "joint_pos": 0.005,
            "joint_vel": 0.01,
            "object_pos": 0.002,
            "object_lin_vel": 0.002,
            "object_ang_vel": 0.002,
        }
        for term_name, expected_std in expected_gaussian_noise_terms.items():
            self.assertEqual(self._noise_std(assignments[term_name]), expected_std)

        self.assertIsNone(self._noise_std(assignments["fingertip_contact_force_b"]))
        self.assertIsNone(self._noise_std(assignments["contact_mask"]))
        self.assertIsNone(self._noise_std(assignments["contact_force_mag"]))

    def _obs_term_func_name(self, call):
        func_kw = next(kw for kw in call.keywords if kw.arg == "func")
        return func_kw.value.value.id, func_kw.value.attr

    def _noise_std(self, call):
        noise_kw = next((kw for kw in call.keywords if kw.arg == "noise"), None)
        if noise_kw is None:
            return None
        return next(kw.value.value for kw in noise_kw.value.keywords if kw.arg == "std")

    def _clip(self, call):
        clip_kw = next((kw for kw in call.keywords if kw.arg == "clip"), None)
        if clip_kw is None:
            return None
        return tuple(value.value for value in clip_kw.value.elts)

    def _uniform_noise_bounds(self, call):
        noise_kw = next((kw for kw in call.keywords if kw.arg == "noise"), None)
        if noise_kw is None:
            return None
        values = {kw.arg: kw.value.value for kw in noise_kw.value.keywords}
        return values["n_min"], values["n_max"]


class CleanTableObservationConfigTest(unittest.TestCase):
    def test_box_pose_observation_uses_receptive_object(self):
        tree = ast.parse((REPO_ROOT / "src" / "tasks" / "clean_table" / "env_cfg.py").read_text())
        assignments = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        target_pose = assignments["box_pose_b"]

        self.assertEqual(self._obs_term_func_name(target_pose), ("mdp", "box_pose_b"))
        self.assertEqual(
            self._scene_entity_param_name(target_pose, "box_cfg"),
            "receptive_object",
        )

    def test_receptive_object_position_is_randomized_on_reset(self):
        tree = ast.parse((REPO_ROOT / "src" / "tasks" / "clean_table" / "env_cfg.py").read_text())
        assignments = {
            target.id: node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("reset_receptive_object", assignments)
        reset_receptive_object = assignments["reset_receptive_object"]
        params = self._params(reset_receptive_object)

        self.assertEqual(self._obs_term_func_name(reset_receptive_object), ("mdp", "reset_root_state_uniform"))
        self.assertEqual(self._keyword_value(reset_receptive_object, "mode"), "reset")
        self.assertEqual(self._scene_entity_name(params["asset_cfg"]), "receptive_object")
        self.assertEqual(self._range_values(params["pose_range"], "x"), (-0.25, 0.15))
        self.assertEqual(self._range_values(params["pose_range"], "y"), (-0.25, 0.25))
        self.assertEqual(self._range_values(params["pose_range"], "z"), (0.0, 0.0))

    def _obs_term_func_name(self, call):
        func_kw = next(kw for kw in call.keywords if kw.arg == "func")
        return func_kw.value.value.id, func_kw.value.attr

    def _scene_entity_param_name(self, call, param_name):
        params_kw = next(kw for kw in call.keywords if kw.arg == "params")
        params = {
            key.value: value
            for key, value in zip(params_kw.value.keys, params_kw.value.values)
        }
        return params[param_name].args[0].value

    def _params(self, call):
        params_kw = next(kw for kw in call.keywords if kw.arg == "params")
        return {
            key.value: value
            for key, value in zip(params_kw.value.keys, params_kw.value.values)
        }

    def _keyword_value(self, call, keyword_name):
        return next(kw.value.value for kw in call.keywords if kw.arg == keyword_name)

    def _scene_entity_name(self, call):
        return call.args[0].value

    def _range_values(self, dict_node, key_name):
        values = {
            key.value: value
            for key, value in zip(dict_node.keys, dict_node.values)
        }
        return tuple(ast.literal_eval(values[key_name]))


if __name__ == "__main__":
    unittest.main()
