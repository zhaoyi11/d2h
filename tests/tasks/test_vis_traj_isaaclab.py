"""Run with: python tests/tasks/test_vis_traj_isaaclab.py (no simulator required)."""

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "src/tasks/pick_and_place/vis_traj_isaaclab.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("vis_traj_isaaclab", SCRIPT)
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)

from src.tasks.common.trajectory import load_object_trajectory


class ObjectReplayTest(unittest.TestCase):
    def test_mano_recordings(self):
        from vis_traj import load_trajectory, skeleton_edges

        for path, count in (
            (SCRIPT.with_name("pick_and_place_pig_mano_cropped_isaac.npz"), 800),
            (SCRIPT.parent.parent / "pouring/drink_0612_mano_isaac_trajectory.npz", 1000),
            (SCRIPT.parent.parent / "pouring/drink_0612_mano_cropped_isaac_trajectory.npz", 510),
        ):
            with self.subTest(path=path):
                original = path.read_bytes()
                times, poses = load_object_trajectory(path)
                hand_poses, points = load_trajectory(path)
                np.testing.assert_allclose(poses, hand_poses)
                np.testing.assert_allclose(times, np.arange(count) / 200)
                self.assertEqual(points.shape, (count, 21, 3))
                self.assertEqual(skeleton_edges(21).shape, (20, 2))
                with np.load(path, allow_pickle=False) as data:
                    np.testing.assert_array_equal(poses[:, :3], data["object_pos"])
                    np.testing.assert_array_equal(points[:, 0], data["hand_root_pos"])
                self.assertEqual(path.read_bytes(), original)

    def test_drink_crop_preserves_recorded_arrays(self):
        source = SCRIPT.parent.parent / "pouring/drink_0612_mano_isaac_trajectory.npz"
        cropped = source.with_name("drink_0612_mano_cropped_isaac_trajectory.npz")
        with np.load(source, allow_pickle=False) as original, np.load(cropped, allow_pickle=False) as data:
            self.assertEqual(set(data.files), set(original.files))
            metadata = json.loads(data["metadata_json"].item())
            crop = metadata.pop("crop")
            self.assertEqual(metadata, json.loads(original["metadata_json"].item()))
            self.assertEqual(crop["source_filename"], source.name)
            self.assertEqual(crop["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual((crop["start_frame"], crop["stop_frame"]), (490, 1000))
            self.assertEqual(crop["time_offset_seconds"], original["time"][490])
            for key in original.files:
                if key == "metadata_json":
                    continue
                value = original[key]
                expected = value[490:] if value.ndim and value.shape[0] == 1000 else value
                if key == "time":
                    expected = expected - original["time"][490]
                with self.subTest(key=key):
                    np.testing.assert_array_equal(data[key], expected)
                    self.assertEqual(data[key].dtype, value.dtype)

    def test_keypoint_recording(self):
        path = SCRIPT.with_name("pickandplace_trajectory_keypoints.npz")
        original = path.read_bytes()
        times, poses = load_object_trajectory(path)
        self.assertEqual(poses.shape, (91, 7))
        np.testing.assert_allclose(times, np.arange(91) / 30)
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(poses[:, :3], data["qpos_obj_right"][:, :3])
            np.testing.assert_allclose(poses[:, 3:], data["qpos_obj_right"][:, 3:], atol=1e-7)
        faster_times, faster_poses = load_object_trajectory(path, fps=60)
        np.testing.assert_array_equal(faster_poses, poses)
        np.testing.assert_allclose(faster_times, times / 2)
        self.assertEqual(path.read_bytes(), original)

    def test_original_recording(self):
        path = SCRIPT.with_name("23963_mano_isaac_trajectory.npz")
        original = path.read_bytes()
        times, poses = load_object_trajectory(path)
        self.assertEqual(poses.shape, (1300, 7))
        np.testing.assert_allclose(times[[0, -1]], [0, 6.495])
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(poses[:, :3], data["object_pos"])
            np.testing.assert_allclose(poses[:, 3:], data["object_quat_wxyz"], atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(poses[:, 3:], axis=1), 1)
        self.assertEqual(path.read_bytes(), original)

    def test_object_only_input_and_validation(self):
        arrays = dict(time=np.array([4., 4.125, 4.375]), object_pos=np.arange(9).reshape(3, 3),
                      object_quat_wxyz=np.tile([2., 0., 0., 0.], (3, 1)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object.npz"
            np.savez(path, **arrays)
            times, poses = load_object_trajectory(path)
            np.testing.assert_array_equal(times, [0, .125, .375])
            np.testing.assert_array_equal(poses[:, 3:], np.tile([1, 0, 0, 0], (3, 1)))
            invalid = [
                {k: v for k, v in arrays.items() if k != "object_pos"},
                {**arrays, "object_pos": np.zeros((3, 4))},
                {**arrays, "object_pos": np.zeros((2, 3))},
                {**arrays, "object_pos": np.full((3, 3), np.nan)},
                {**arrays, "object_pos": np.full((3, 3), "bad")},
                {**arrays, "object_quat_wxyz": np.zeros((3, 4))},
                {**arrays, "time": np.array([0, 0, 1])},
                {**arrays, "time": np.array([0, 2, 1])},
                {**arrays, "time": np.array([0, 1, np.inf])},
                {k: v[:0] for k, v in arrays.items()},
            ]
            for data in invalid:
                with self.subTest(data=data):
                    np.savez(path, **data)
                    with self.assertRaises(ValueError):
                        load_object_trajectory(path)

    def test_timestamp_selection_hold_and_loop(self):
        times = np.array([0, .125, .375])
        for elapsed, expected in [(0, 0), (.124, 0), (.125, 1), (.374, 1), (.375, 2), (10, 2)]:
            self.assertEqual(viewer.frame_index(times, elapsed), expected)
        for elapsed, expected in [(.624, 2), (.625, 0), (.75, 1), (1.25, 0)]:
            self.assertEqual(viewer.frame_index(times, elapsed, loop=True), expected)
        self.assertEqual(viewer.frame_index(np.array([0.]), 100, loop=True), 0)


if __name__ == "__main__":
    unittest.main()
