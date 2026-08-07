from __future__ import annotations

import pathlib
from types import SimpleNamespace

import numpy as np

from src.policy.data_collection_for_retargeting import RetargetingEpisodeWriter
from src.policy.data_collection_for_retargeting import _build_metadata_payload


def _batch(num_envs: int = 1, offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "leap_joint_pos": np.full((num_envs, 16), offset, dtype=np.float32),
        "leap_joint_vel": np.full((num_envs, 16), offset + 1.0, dtype=np.float32),
        "leap_root_pose_w": np.full((num_envs, 7), offset + 2.0, dtype=np.float32),
        "policy_action_clean": np.full((num_envs, 16), offset + 3.0, dtype=np.float32),
        "policy_action_executed": np.full((num_envs, 16), offset + 4.0, dtype=np.float32),
        "action_noise_std": np.full(num_envs, offset + 5.0, dtype=np.float32),
        "reward": np.full(num_envs, offset + 6.0, dtype=np.float32),
        "terminated": np.zeros(num_envs, dtype=bool),
        "truncated": np.zeros(num_envs, dtype=bool),
    }


def test_add_step_does_not_flush_when_done_is_false(tmp_path: pathlib.Path) -> None:
    writer = RetargetingEpisodeWriter(tmp_path, num_envs=1, max_workers=1)

    flushed = writer.add_step(
        step_data=_batch(),
        done=np.array([False]),
    )
    writer.close()

    assert flushed == 0
    assert writer.num_saved == 0
    assert list(tmp_path.glob("*.npz")) == []


def test_add_step_writes_state_arrays_without_trailing_post_reset_frame(tmp_path: pathlib.Path) -> None:
    writer = RetargetingEpisodeWriter(tmp_path, num_envs=1, max_workers=1)

    writer.add_step(step_data=_batch(offset=0.0), done=np.array([False]))
    flushed = writer.add_step(step_data=_batch(offset=10.0), done=np.array([True]))
    writer.close()

    assert flushed == 1
    assert writer.num_saved == 1
    with np.load(tmp_path / "0000000000.npz") as episode:
        assert episode["leap_joint_pos"].shape == (2, 16)
        assert episode["leap_root_pose_w"].shape == (2, 7)
        assert episode["policy_action_clean"].shape == (2, 16)
        assert "observation.policy" not in episode.files


def test_add_step_preserves_terminated_and_truncated_arrays(tmp_path: pathlib.Path) -> None:
    writer = RetargetingEpisodeWriter(tmp_path, num_envs=1, max_workers=1)
    first = _batch(offset=0.0)
    first["terminated"] = np.array([True])
    first["truncated"] = np.array([False])
    second = _batch(offset=1.0)
    second["terminated"] = np.array([False])
    second["truncated"] = np.array([True])

    writer.add_step(step_data=first, done=np.array([False]))
    writer.add_step(step_data=second, done=np.array([True]))
    writer.close()

    with np.load(tmp_path / "0000000000.npz") as episode:
        np.testing.assert_array_equal(episode["terminated"], np.array([True, False]))
        np.testing.assert_array_equal(episode["truncated"], np.array([False, True]))


def test_simultaneous_dones_do_not_write_past_target_count(tmp_path: pathlib.Path) -> None:
    writer = RetargetingEpisodeWriter(
        tmp_path,
        num_envs=3,
        max_workers=1,
        max_episodes=2,
    )

    flushed = writer.add_step(
        step_data=_batch(num_envs=3),
        done=np.array([True, True, True]),
    )
    writer.close()

    assert flushed == 2
    assert writer.num_saved == 2
    assert sorted(path.name for path in tmp_path.glob("*.npz")) == ["0000000000.npz", "0000000001.npz"]


def test_add_step_records_episode_success_when_available(tmp_path: pathlib.Path) -> None:
    writer = RetargetingEpisodeWriter(tmp_path, num_envs=1, max_workers=1)

    writer.add_step(
        step_data=_batch(),
        done=np.array([True]),
        success=np.array([True]),
    )
    writer.close()

    with np.load(tmp_path / "0000000000.npz") as episode:
        assert bool(episode["success"]) is True


def test_metadata_payload_records_retargeting_conventions() -> None:
    args = SimpleNamespace(
        action_noise_std_max=0.0,
        action_term_name="joint_pos",
        contact_force_threshold=0.25,
        deterministic=True,
        num_envs=4,
        num_episodes=100,
        seed=0,
        success_key=None,
        task="Reorient_Play-v0",
    )
    env_cfg = SimpleNamespace(sim=SimpleNamespace(dt=1.0 / 120.0), decimation=4, episode_length_s=15)
    robot = SimpleNamespace(
        joint_names=[f"a_{idx}" for idx in range(16)],
        body_names=["base", "thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"],
    )

    metadata = _build_metadata_payload(
        args_cli=args,
        env_cfg=env_cfg,
        robot=robot,
        action_dim=16,
        resume_path="/tmp/model.pt",
        git_sha="abc123",
        timestamp="2026-05-03T12:00:00",
        recorded_fields=["leap_joint_pos", "leap_fingertip_pose_w", "policy_action_clean"],
    )

    assert metadata["units"]["position"] == "meters"
    assert metadata["units"]["angle"] == "radians"
    assert metadata["quaternion_convention"] == "wxyz"
    assert metadata["frame_convention"]["_b"] == "LEAP base/root frame"
    assert metadata["human21_keypoint_names"][0] == "wrist"
    assert len(metadata["human21_keypoint_names"]) == 21
    assert "no pinky" in metadata["human21_pinky_policy"]
    assert metadata["fingertip_link_names"] == ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
