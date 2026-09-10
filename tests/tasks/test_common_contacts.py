"""CPU regression for shared fingertip contact aggregation and frame conversion."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def test_contact_metrics_preserve_filter_masks_force_frames_and_empty_sensors():
    envs = ModuleType("isaaclab.envs")
    envs.ManagerBasedRLEnv = object
    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.quat_apply_inverse = lambda quat, vec: torch.as_tensor(
        Rotation.from_quat(quat.numpy(), scalar_first=True).inv().apply(vec.numpy()),
        dtype=vec.dtype,
    )
    spec = importlib.util.spec_from_file_location(
        "common_contacts_under_test",
        Path(__file__).resolve().parents[2] / "src/tasks/common/mdps/contacts.py",
    )
    contacts = importlib.util.module_from_spec(spec)
    rotation = [np.sqrt(0.5), 0., 0., np.sqrt(0.5)]
    env = SimpleNamespace(num_envs=2, device="cpu", scene=SimpleNamespace(sensors={
        "tips": SimpleNamespace(data=SimpleNamespace(
            target_quat_w=torch.tensor([[rotation] * 2] * 2),
        )),
        "tip": SimpleNamespace(data=SimpleNamespace(force_matrix_w=torch.tensor([
            [[[0., 2., 0.], [0., 3., 0.], [0., -3., 0.]]],
            [[[float("nan"), 0., 0.], [2., 0., 0.], [0., 0., 0.]]],
        ]))),
        "empty": SimpleNamespace(data=SimpleNamespace(force_matrix_w=None)),
    }))
    with patch.dict(sys.modules, {"isaaclab.envs": envs, "isaaclab.utils.math": math_utils}):
        spec.loader.exec_module(contacts)
        object_contact = contacts._compute_contact_metrics(env, ["tip", "empty"], "tips", .25, 50.)
        external_contact = contacts._compute_contact_metrics(
            env, ["tip", "empty"], "tips", .25, 50., [1, 2, 99],
        )
    torch.testing.assert_close(object_contact["in_contact"], torch.tensor([[True, False], [False, False]]))
    torch.testing.assert_close(object_contact["force_mag"], torch.tensor([[2., 0.], [0., 0.]]))
    torch.testing.assert_close(object_contact["contact_pose"], torch.zeros(2, 2, 2), atol=1e-6, rtol=0.)
    # Opposing external forces retain contact even though their net force is zero.
    torch.testing.assert_close(external_contact["in_contact"], torch.tensor([[True, False], [True, False]]))
    torch.testing.assert_close(external_contact["force_mag"], torch.tensor([[0., 0.], [2., 0.]]))
    torch.testing.assert_close(external_contact["contact_pose"][1, 0], torch.tensor([-np.deg2rad(50.), 0.], dtype=torch.float32), atol=1e-6, rtol=0.)


def test_grasp_contact_requires_thumb_and_another_finger_on_object_filter():
    envs = ModuleType("isaaclab.envs")
    envs.ManagerBasedRLEnv = object
    spec = importlib.util.spec_from_file_location(
        "common_grasp_contacts_under_test",
        Path(__file__).resolve().parents[2] / "src/tasks/common/mdps/contacts.py",
    )
    contacts = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"isaaclab.envs": envs}):
        spec.loader.exec_module(contacts)
    names = ["thumb_fingertip_object_s", "fingertip_object_s", "fingertip_2_object_s", "fingertip_3_object_s"]
    forces = torch.zeros(3, 1, 2, 3)
    forces[:, :, 1, 0] = 100.0  # External contact never establishes an object grasp.
    thumb = forces.clone()
    thumb[:2, :, 0, 0] = 2.0
    index = forces.clone()
    index[[0, 2], :, 0, 0] = 2.0
    sensors = {name: SimpleNamespace(data=SimpleNamespace(force_matrix_w=value))
               for name, value in zip(names, [thumb, index, forces, None])}
    env = SimpleNamespace(num_envs=3, device="cpu", scene=SimpleNamespace(sensors=sensors))
    torch.testing.assert_close(contacts.contacts(env, 1.0), torch.tensor([True, False, False]))
