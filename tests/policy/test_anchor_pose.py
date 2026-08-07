from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "dev" / "anchor_pose.py"
SPEC = importlib.util.spec_from_file_location("anchor_pose", SCRIPT_PATH)
anchor_pose = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = anchor_pose
SPEC.loader.exec_module(anchor_pose)


def test_compute_anchor_pose_b_averages_all_base_frame_fingertips(tmp_path: Path) -> None:
    link_names = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
    (tmp_path / "metadata.json").write_text(json.dumps({"link_names": link_names}))
    link_pose_b = np.zeros((1, 4, 7), dtype=np.float32)
    link_pose_b[:, :, 3] = 1.0
    link_pose_b[0, :, :3] = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ]

    np.savez_compressed(tmp_path / "0000000000.npz", leap_link_pose_b=link_pose_b)

    pose = anchor_pose.compute_anchor_pose_b(tmp_path)

    np.testing.assert_allclose(
        pose,
        np.array([5.5, 6.46625, 7.5075, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert pose.dtype == np.float32


def test_compute_anchor_pose_b_requires_link_pose(tmp_path: Path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {"link_names": ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]}
        )
    )
    np.savez_compressed(
        tmp_path / "0000000000.npz",
        leap_fingertip_pose_b=np.zeros((1, 1, 7), dtype=np.float32),
    )

    with pytest.raises(KeyError, match="leap_link_pose_b"):
        anchor_pose.compute_anchor_pose_b(tmp_path)


def test_compute_anchor_pose_b_contact_only_averages_masked_fingertips(
    tmp_path: Path,
) -> None:
    link_names = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
    (tmp_path / "metadata.json").write_text(json.dumps({"link_names": link_names}))
    link_pose_b = np.zeros((2, 4, 7), dtype=np.float32)
    link_pose_b[:, :, 3] = 1.0
    contact_mask = np.array(
        [
            [True, False, False, False],
            [False, True, True, False],
        ],
        dtype=bool,
    )
    np.savez_compressed(
        tmp_path / "0000000000.npz",
        leap_link_pose_b=link_pose_b,
        fingertip_contact_mask=contact_mask,
    )

    pose = anchor_pose.compute_anchor_pose_b(tmp_path, contact_only=True)

    np.testing.assert_allclose(
        pose,
        np.array([0.0, -0.035, 0.005, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )


def test_compute_anchor_pose_b_contact_only_requires_contact_mask(tmp_path: Path) -> None:
    link_names = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
    (tmp_path / "metadata.json").write_text(json.dumps({"link_names": link_names}))
    link_pose_b = np.zeros((1, 4, 7), dtype=np.float32)
    link_pose_b[:, :, 3] = 1.0
    np.savez_compressed(tmp_path / "0000000000.npz", leap_link_pose_b=link_pose_b)

    with pytest.raises(KeyError, match="fingertip_contact_mask"):
        anchor_pose.compute_anchor_pose_b(tmp_path, contact_only=True)


def test_compute_anchor_pose_b_contact_only_validates_mask_shape(tmp_path: Path) -> None:
    link_names = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
    (tmp_path / "metadata.json").write_text(json.dumps({"link_names": link_names}))
    link_pose_b = np.zeros((1, 4, 7), dtype=np.float32)
    link_pose_b[:, :, 3] = 1.0
    np.savez_compressed(
        tmp_path / "0000000000.npz",
        leap_link_pose_b=link_pose_b,
        fingertip_contact_mask=np.zeros((1, 3), dtype=bool),
    )

    with pytest.raises(ValueError, match="fingertip_contact_mask"):
        anchor_pose.compute_anchor_pose_b(tmp_path, contact_only=True)


def test_parser_contact_only_defaults_off() -> None:
    parser = anchor_pose.build_parser()

    assert parser.parse_args([]).contact_only is False
    assert parser.parse_args(["--contact-only"]).contact_only is True


def test_offset_fingertip_positions_b_rotates_offsets_by_link_quaternion() -> None:
    link_pose_b = np.zeros((1, 1, 7), dtype=np.float32)
    link_pose_b[0, 0, :3] = [1.0, 2.0, 3.0]
    link_pose_b[0, 0, 3:7] = [
        np.cos(np.pi / 4.0),
        0.0,
        0.0,
        np.sin(np.pi / 4.0),
    ]

    positions = anchor_pose.offset_fingertip_positions_b(
        link_pose_b,
        ["tip"],
        offsets=(("tip", (1.0, 0.0, 0.0)),),
    )

    np.testing.assert_allclose(
        positions,
        np.array([[[1.0, 3.0, 3.0]]], dtype=np.float64),
        atol=1e-6,
    )


def test_make_anchor_axes_uses_identity_base_frame_orientation() -> None:
    pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    axes = anchor_pose.make_anchor_axes(pose, axis_length=0.5)

    expected = [
        np.array([[1.0, 2.0, 3.0], [1.5, 2.0, 3.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0], [1.0, 2.5, 3.0]], dtype=np.float32),
        np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.5]], dtype=np.float32),
    ]
    assert len(axes) == 3
    for actual, target in zip(axes, expected):
        np.testing.assert_allclose(actual, target)


def test_log_anchor_episode_to_rerun_logs_static_anchor_and_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_path = tmp_path / "0000000010.npz"
    rrd_path = tmp_path / "anchor.rrd"
    link_names = [
        "base",
        "mcp_joint",
        "thumb_temp_base",
        "mcp_joint_2",
        "mcp_joint_3",
        "pip",
        "thumb_pip",
        "pip_2",
        "pip_3",
        "dip",
        "thumb_dip",
        "dip_2",
        "dip_3",
        "fingertip",
        "thumb_fingertip",
        "fingertip_2",
        "fingertip_3",
    ]
    link_pose_b = np.zeros((2, len(link_names), 7), dtype=np.float32)
    link_pose_b[:, :, 3] = 1.0
    link_pose_b[:, :, 0] = np.arange(len(link_names), dtype=np.float32)
    np.savez_compressed(
        episode_path,
        leap_link_pose_b=link_pose_b,
        time=np.array([0.0, 0.1], dtype=np.float32),
    )

    calls = []

    class FakePoints3D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeLineStrips3D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeRerun:
        Points3D = FakePoints3D
        LineStrips3D = FakeLineStrips3D

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def save(path):
            Path(path).write_bytes(b"fake")

        @staticmethod
        def set_time(*args, **kwargs):
            return None

        @staticmethod
        def log(entity_path, entity, static=False):
            calls.append((entity_path, entity, static))

    monkeypatch.setitem(sys.modules, "rerun", FakeRerun)

    anchor_pose.log_anchor_episode_to_rerun(
        episode_path=episode_path,
        link_names=link_names,
        anchor_pose_b=np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rrd_path=rrd_path,
        spawn=False,
        show_labels=True,
        max_frames=None,
    )

    assert rrd_path.exists()
    assert len([call for call in calls if call[0] == "anchor/pose" and call[2]]) == 1
    assert len([call for call in calls if call[0] == "anchor/axes" and call[2]]) == 1
    assert len([call for call in calls if call[0] == "leap/links"]) == 2
    assert len([call for call in calls if call[0] == "leap/keypoints"]) == 2
    assert len([call for call in calls if call[0] == "leap/offset_fingertips"]) == 2


def test_log_anchor_episode_to_rerun_spawn_opens_saved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_path = tmp_path / "0000000010.npz"
    rrd_path = tmp_path / "anchor.rrd"
    link_names = ["base", "thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
    link_pose_b = np.zeros((1, len(link_names), 7), dtype=np.float32)
    link_pose_b[:, :, 3] = 1.0
    np.savez_compressed(episode_path, leap_link_pose_b=link_pose_b)

    events = []
    spawned_paths = []

    class FakePoints3D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeLineStrips3D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeRerun:
        Points3D = FakePoints3D
        LineStrips3D = FakeLineStrips3D

        @staticmethod
        def init(*args, **kwargs):
            events.append(("init", args, kwargs))

        @staticmethod
        def save(path):
            events.append(("save", Path(path)))
            Path(path).write_bytes(b"fake")

        @staticmethod
        def set_time(*args, **kwargs):
            return None

        @staticmethod
        def log(*args, **kwargs):
            return None

        @staticmethod
        def disconnect():
            events.append(("disconnect",))

    monkeypatch.setitem(sys.modules, "rerun", FakeRerun)
    monkeypatch.setattr(anchor_pose, "spawn_rerun_file_viewer", spawned_paths.append)

    anchor_pose.log_anchor_episode_to_rerun(
        episode_path=episode_path,
        link_names=link_names,
        anchor_pose_b=np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rrd_path=rrd_path,
        spawn=True,
    )

    assert events[0][0] == "init"
    assert events[0][2]["spawn"] is False
    assert ("disconnect",) in events
    assert spawned_paths == [rrd_path]


def test_log_anchor_episode_to_rerun_logs_mesh_assets_and_transforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episode_path = tmp_path / "0000000010.npz"
    rrd_path = tmp_path / "anchor_mesh.rrd"
    link_names = ["base", "fingertip"]
    link_pose_b = np.zeros((2, 2, 7), dtype=np.float32)
    link_pose_b[:, :, 3] = 1.0
    link_pose_b[:, 1, :3] = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    np.savez_compressed(episode_path, leap_link_pose_b=link_pose_b)

    calls = []

    class FakeMesh:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            dtype=np.float32,
        )
        triangles = np.array([[0, 1, 2]], dtype=np.uint32)
        vertex_colors = None

    class FakePoints3D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeLineStrips3D:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeMesh3D:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTransform3D:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeRerun:
        Points3D = FakePoints3D
        LineStrips3D = FakeLineStrips3D
        Mesh3D = FakeMesh3D
        Transform3D = FakeTransform3D

        @staticmethod
        def init(*args, **kwargs):
            return None

        @staticmethod
        def save(path):
            Path(path).write_bytes(b"fake")

        @staticmethod
        def set_time(*args, **kwargs):
            return None

        @staticmethod
        def log(entity_path, entity, static=False):
            calls.append((entity_path, entity, static))

    monkeypatch.setitem(sys.modules, "rerun", FakeRerun)

    anchor_pose.log_anchor_episode_to_rerun(
        episode_path=episode_path,
        link_names=link_names,
        anchor_pose_b=np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        rrd_path=rrd_path,
        robot_meshes={"base": FakeMesh(), "fingertip": FakeMesh()},
        show_offset_tips=False,
    )

    static_mesh_calls = [
        call for call in calls if call[0].endswith("/geometry") and call[2]
    ]
    transform_calls = [
        call for call in calls if call[0] in {"leap_mesh/base", "leap_mesh/fingertip"}
    ]
    assert len(static_mesh_calls) == 2
    assert all(isinstance(call[1], FakeMesh3D) for call in static_mesh_calls)
    assert len(transform_calls) == 4
    assert all(isinstance(call[1], FakeTransform3D) for call in transform_calls)
    assert not [call for call in calls if call[0] == "leap/links"]
