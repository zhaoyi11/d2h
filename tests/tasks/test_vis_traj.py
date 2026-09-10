"""Run with: python tests/tasks/test_vis_traj.py"""

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/tasks/pick_and_place/vis_traj.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("vis_traj", SCRIPT)
vis_traj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis_traj)


class TrajectoryTest(unittest.TestCase):
    def test_recorded_points_and_object_poses(self):
        path = SCRIPT.with_name("trajectory_keypoints.npz")
        poses, points = vis_traj.load_trajectory(path)
        self.assertEqual(poses.shape, (97, 7))
        self.assertEqual(points.shape, (97, 16, 3))
        segments = points[:, vis_traj.skeleton_edges(points.shape[1])]
        self.assertEqual(segments.shape, (97, 15, 2, 3))
        self.assertEqual(vis_traj.playback_fps(path), 30)
        self.assertEqual(vis_traj.playback_fps(path, 60), 60)
        with np.load(path, allow_pickle=False) as source:
            np.testing.assert_allclose(poses, source["qpos_obj_right"], atol=1e-15)
            np.testing.assert_array_equal(segments[:, :5, 0], np.repeat(source["qpos_wrist_right"][:, None, :3], 5, axis=1))
            np.testing.assert_array_equal(segments[:, :5, 1], source["qpos_pip_right"])
            np.testing.assert_array_equal(segments[:, 5:10, 0], source["qpos_pip_right"])
            np.testing.assert_array_equal(segments[:, 5:10, 1], source["qpos_dip_right"])
            np.testing.assert_array_equal(segments[:, 10:, 0], source["qpos_dip_right"])
            np.testing.assert_array_equal(segments[:, 10:, 1], source["qpos_finger_right"][..., :3])

    def test_validation_and_quaternion_normalization(self):
        pose = np.array([[1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0]])
        data = {
            "qpos_obj_right": pose.copy(),
            "qpos_wrist_right": pose.copy(),
            "qpos_pip_right": np.zeros((1, 5, 3)),
            "qpos_dip_right": np.zeros((1, 5, 3)),
            "qpos_finger_right": np.repeat(pose[:, None], 5, axis=1),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.npz"
            np.savez(path, **data)
            poses, points = vis_traj.load_trajectory(path)
            np.testing.assert_array_equal(poses, [[1, 2, 3, 1, 0, 0, 0]])
            self.assertEqual(points.shape, (1, 16, 3))

            invalid = [
                ({key: value for key, value in data.items() if key != "qpos_obj_right"}, "Missing"),
                ({**data, "qpos_obj_right": np.zeros((1, 6))}, "shape"),
                ({**data, "qpos_obj_right": np.zeros((0, 7))}, "shape"),
                ({**data, "qpos_obj_right": np.repeat(pose, 2, axis=0)}, "frame counts"),
                ({**data, "qpos_pip_right": np.full((1, 5, 3), np.nan)}, "finite"),
                ({**data, "qpos_obj_right": np.zeros((1, 7))}, "quaternions"),
                ({**data, "qpos_mcp_right": np.zeros((1, 4, 3))}, "shape"),
            ]
            for arrays, message in invalid:
                with self.subTest(message=message):
                    np.savez(path, **arrays)
                    with self.assertRaisesRegex(ValueError, message):
                        vis_traj.load_trajectory(path)

        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                vis_traj.positive_float(value)


    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "Joint-state loading requires MuJoCo")
    def test_export_maps_joint_names_and_preserves_world_coordinates(self):
        spec = importlib.util.spec_from_file_location("export_hand_keypoints", SCRIPT.with_name("export_hand_keypoints.py"))
        exporter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exporter)
        fingers = ("thumb", "index", "middle", "ring", "pinky")
        base_names = ["right_pos_x", "right_pos_y", "right_pos_z", "right_rot_z"]
        finger_names = [f"right_j_{finger}1{'x' if finger == 'thumb' else 'y'}" for finger in fingers]
        base_joints = ''.join(
            f'<joint name="{name}" type="slide" axis="{axis}"/>'
            for name, axis in zip(base_names[:3], ("1 0 0", "0 1 0", "0 0 1"))
        ) + '<joint name="right_rot_z" axis="0 0 1"/>'
        finger_bodies = ''.join(
            f'<body pos="{i * 0.01} 0 0"><joint name="{name}" axis="0 0 1"/>'
            '<geom type="sphere" size="0.01"/>'
            + ''.join(f'<site name="right_{finger}_{level}" pos="{offset} 0 0"/>'
                      for level, offset in (("pip", .02), ("dip", .04), ("tip", .06)))
            + '</body>'
            for i, (finger, name) in enumerate(zip(fingers, finger_names))
        )
        xml = '<mujoco><compiler angle="radian"/><worldbody><body name="right_palm">' + base_joints
        xml += '<geom type="sphere" size="0.01"/>' + finger_bodies + '</body></worldbody></mujoco>'
        qpos = np.zeros((2, 9))
        qpos[:, :3] = [[1, 2, 3], [4, 5, 6]]
        qpos[1, 3] = qpos[1, 5] = np.pi / 2
        arrays = {
            "robot_joint_names": np.array(base_names + finger_names)[::-1],
            "robot_joint_pos": qpos[:, ::-1],
            "hand_root_pos": qpos[:, :3],
            "hand_root_quat_wxyz": np.array([[1, 0, 0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)]]),
            "object_pos": np.zeros((2, 3)),
            "object_quat_wxyz": np.tile([1, 0, 0, 0], (2, 1)),
            "frequency": np.array(200.0),
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.npz"
            scene = Path(directory) / "hand.xml"
            output = Path(directory) / "keypoints.npz"
            scene.write_text(xml)
            np.savez(source, **arrays)
            original = source.read_bytes()
            direct_poses, direct_points = vis_traj.load_trajectory(source, scene)
            self.assertEqual(set(Path(directory).iterdir()), {source, scene})
            self.assertEqual(source.read_bytes(), original)
            exporter.export_keypoints(source, scene, output)
            poses, points = vis_traj.load_trajectory(output)
            np.testing.assert_array_equal(direct_poses, poses)
            np.testing.assert_array_equal(direct_points, points)
            self.assertEqual(points.shape, (2, 21, 3))
            np.testing.assert_allclose(points[:, 0], [[1, 2, 3], [4, 5, 6]])
            np.testing.assert_allclose(points[0, 17], [1.07, 2, 3], atol=1e-6)
            np.testing.assert_allclose(points[1, 17], [3.94, 5.01, 6], atol=1e-6)
            edges = vis_traj.skeleton_edges(21)
            self.assertEqual(edges.shape, (20, 2))
            np.testing.assert_array_equal(edges[[0, 5, 10, 15]], [[0, 1], [1, 6], [6, 11], [11, 16]])
            self.assertEqual(vis_traj.playback_fps(output), 200)
            with np.load(output) as exported:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(exported[key], value)
            saved_export = output.read_bytes()
            with self.assertRaises(FileExistsError):
                exporter.export_keypoints(source, scene, output)
            self.assertEqual(output.read_bytes(), saved_export)
            self.assertEqual(source.read_bytes(), original)
            arrays["hand_root_pos"] = arrays["hand_root_pos"] + 1
            np.savez(source, **arrays)
            with self.assertRaisesRegex(ValueError, "wrist pose"):
                exporter.export_keypoints(source, scene, Path(directory) / "invalid.npz")
            self.assertFalse((Path(directory) / "invalid.npz").exists())


if __name__ == "__main__":
    unittest.main()
