"""Recorded carry geometry and lazy task registration, without launching Isaac Sim."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.tasks.pick_and_place.trajectory import (
    DEFAULT_TRAJECTORY, build_pick_and_place_trajectory, load_carry_trajectory,
)


def test_original_recording_and_stage_layout():
    before = DEFAULT_TRAJECTORY.read_bytes()
    carry, frames, segments = load_carry_trajectory()
    with np.load(DEFAULT_TRAJECTORY) as data:
        source = np.c_[data["object_pos"], data["object_quat_wxyz"]]
        np.testing.assert_allclose(carry, source[frames[1:9]], atol=1e-7)
    assert len(carry) == 8
    assert frames[:2].tolist() == [320, 320]
    assert frames[8] == 511
    assert (frames[9:] == -1).all()
    assert segments == (0, 1, 7, 10, 1, 1)
    assert len(frames) == 1 + sum(segments) == 21
    assert DEFAULT_TRAJECTORY.read_bytes() == before


@pytest.mark.parametrize("stride", [7, 14, 28])
def test_carry_alignment_and_deposit_with_rotated_box(stride):
    carry, _, _ = load_carry_trajectory(stride=stride)
    start = np.r_[[0.55, 0.10, 0.32], Rotation.from_euler("z", 0.4).as_quat(scalar_first=True)]
    box = np.r_[[0.55, -0.35, 0.271], Rotation.from_euler("z", 0.7).as_quat(scalar_first=True)]
    above_offset = (0.02, -0.03, 0.25)
    deposit_offset = (0.02, -0.03, 0.10)
    path = build_pick_and_place_trajectory(carry, start, box, above_offset, deposit_offset)
    box_rotation = Rotation.from_quat(box[3:], scalar_first=True)
    above = box[:3] + box_rotation.apply(above_offset)
    deposit = box[:3] + box_rotation.apply(deposit_offset)
    np.testing.assert_allclose(path[:2], np.tile(start, (2, 1)), atol=1e-7)
    np.testing.assert_allclose(path[len(carry), :3], above, atol=1e-7)
    np.testing.assert_allclose(path[-3:, :3], np.tile(deposit, (3, 1)), atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(path[:, 3:], axis=1), 1.0)
    assert (path[2:len(carry) + 1, 2] >= start[2]).all()
    assert (np.diff(path[len(carry):-2, 2]) < 0).all()
    # Preserve the recorded rotation change, rather than replacing it with the box orientation.
    expected = (Rotation.from_quat(start[3:], scalar_first=True)
                * Rotation.from_quat(carry[0, 3:], scalar_first=True).inv()
                * Rotation.from_quat(carry[-1, 3:], scalar_first=True))
    actual = Rotation.from_quat(path[-1, 3:], scalar_first=True)
    assert (expected.inv() * actual).magnitude() < 1e-7
    shifted_start, shifted_box = start.copy(), box.copy()
    offset = [2.0, -3.0, 0.0]
    shifted_start[:3] += offset
    shifted_box[:3] += offset
    shifted = build_pick_and_place_trajectory(carry, shifted_start, shifted_box, above_offset, deposit_offset)
    np.testing.assert_allclose(shifted[:, :3], path[:, :3] + offset)
    np.testing.assert_allclose(shifted[:, 3:], path[:, 3:])


@pytest.mark.parametrize("kwargs", [
    {"stride": 0}, {"start_frame": -1}, {"end_frame": 800},
    {"start_frame": 511}, {"stride": 1.5}, {"start_frame": 0, "end_frame": 100},
])
def test_invalid_carry_interval(kwargs):
    with pytest.raises(ValueError):
        load_carry_trajectory(**kwargs)


def test_pick_and_place_registration():
    import gymnasium as gym
    import src.tasks  # noqa: F401

    spec = gym.spec("PickAndPlace_HRL-v0")
    assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert spec.kwargs["env_cfg_entry_point"] == "src.tasks.pick_and_place.env_cfg:PickAndPlaceEnvCfg"
