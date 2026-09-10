"""Small CPU checks for the recorded pouring input; physics checks live in check_pouring_runtime.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pouring_trajectory", ROOT / "src/tasks/pouring/trajectory.py")
trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory)


def test_real_recording_keeps_object_poses_and_grasp_pause():
    assert trajectory.DEFAULT_TRAJECTORY == ROOT / "src/tasks/pick_and_place/pouring_mano_isaac_trajectory.npz"
    poses, frames, segments = trajectory.load_pouring_trajectory()
    with np.load(trajectory.DEFAULT_TRAJECTORY, allow_pickle=False) as data:
        np.testing.assert_allclose(poses[:, :3], data["object_pos"][frames])
    assert frames[0] == 0 and frames[-1] == 1299
    assert (frames == 0).sum() == 2
    assert np.all(np.diff(frames) >= 0)
    assert frames[segments[0]] == frames[segments[0] + 1] == 0
    assert segments == (0, 1, 186)
    assert poses.dtype == np.float32
    assert segments[1] == 1 and sum(segments) + 1 == len(poses)
    np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1, atol=1e-6)


@pytest.mark.parametrize("bad_key,bad_value", [
    ("time", np.array([0., 0., 1.])),
    ("object_quat_wxyz", np.zeros((3, 4))),
    ("object_pos", np.zeros((3, 2))),
    ("object_pos", np.full((3, 3), np.nan)),
])
def test_rejects_invalid_recording(tmp_path, bad_key, bad_value):
    poses = np.zeros((3, 7)); poses[:, 3] = 1
    arrays = {"time": np.arange(3), "object_pos": poses[:, :3], "object_quat_wxyz": poses[:, 3:]}
    arrays[bad_key] = bad_value
    path = tmp_path / "invalid.npz"
    np.savez(path, **arrays)
    with pytest.raises(ValueError):
        trajectory.load_pouring_trajectory(path, grasp_frame=1)


def test_recording_does_not_require_hand_keypoints(tmp_path):
    poses = np.zeros((3, 7), dtype=np.float32)
    poses[:, 3] = 1
    path = tmp_path / "object_only.npz"
    np.savez(path, time=np.arange(3), qpos_obj_right=poses)
    sampled, frames, _ = trajectory.load_pouring_trajectory(path, stride=1, grasp_frame=1)
    np.testing.assert_array_equal(sampled, poses[frames])


def test_sampling_configuration_is_validated():
    for kwargs in ({"stride": 0}, {"grasp_frame": -1}, {"grasp_frame": 1299}):
        with pytest.raises(ValueError):
            trajectory.load_pouring_trajectory(**kwargs)


def test_pouring_registration_is_lazy():
    import gymnasium as gym
    import src.tasks  # noqa: F401

    spec = gym.spec("Pouring_HRL-v0")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "src.tasks.pouring.env_cfg:PouringEnvCfg"
