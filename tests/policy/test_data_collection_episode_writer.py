from __future__ import annotations

import io
import pathlib
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np


def _load_episode_writer() -> type:
    source_path = pathlib.Path(__file__).resolve().parents[2] / "src" / "policy" / "data_collection.py"
    source = source_path.read_text()
    start = source.index("# EpisodeWriter")
    end = source.index("# Helpers")
    namespace = {
        "Future": Future,
        "ThreadPoolExecutor": ThreadPoolExecutor,
        "io": io,
        "np": np,
        "pathlib": pathlib,
        "threading": threading,
    }
    exec(source[start:end], namespace)
    return namespace["EpisodeWriter"]


EpisodeWriter = _load_episode_writer()


def _batch(num_envs: int = 1, offset: float = 0.0) -> dict[str, np.ndarray]:
    return {
        "obs_groups": {
            "policy": (np.arange(num_envs * 2, dtype=np.float32).reshape(num_envs, 2) + offset),
        },
        "action": np.full((num_envs, 3), offset, dtype=np.float32),
        "reward": np.full(num_envs, offset, dtype=np.float32),
        "next_obs_groups": {
            "policy": (np.arange(num_envs * 2, dtype=np.float32).reshape(num_envs, 2) + offset + 100.0),
        },
    }


def test_add_step_does_not_flush_when_done_is_false(tmp_path: pathlib.Path) -> None:
    writer = EpisodeWriter(tmp_path, num_envs=1, obs_group_names=["policy"], max_workers=1)
    data = _batch()

    flushed = writer.add_step(
        **data,
        done=np.array([False]),
        terminated=np.array([False]),
        truncated=np.array([False]),
    )
    writer.close()

    assert flushed == 0
    assert writer.num_saved == 0
    assert list(tmp_path.glob("*.npz")) == []


def test_add_step_flushes_episode_when_done_is_true(tmp_path: pathlib.Path) -> None:
    writer = EpisodeWriter(tmp_path, num_envs=1, obs_group_names=["policy"], max_workers=1)

    writer.add_step(
        **_batch(offset=0.0),
        done=np.array([False]),
        terminated=np.array([False]),
        truncated=np.array([False]),
    )
    flushed = writer.add_step(
        **_batch(offset=10.0),
        done=np.array([True]),
        terminated=np.array([False]),
        truncated=np.array([False]),
    )
    writer.close()

    assert flushed == 1
    assert writer.num_saved == 1
    with np.load(tmp_path / "0000000000.npz") as episode:
        assert episode["action"].shape == (2, 3)
        assert episode["observation.policy"].shape == (3, 2)
        assert not episode["terminated"].any()
        assert not episode["truncated"].any()


def test_add_step_preserves_terminated_and_truncated_arrays(tmp_path: pathlib.Path) -> None:
    writer = EpisodeWriter(tmp_path, num_envs=1, obs_group_names=["policy"], max_workers=1)

    writer.add_step(
        **_batch(offset=0.0),
        done=np.array([False]),
        terminated=np.array([True]),
        truncated=np.array([False]),
    )
    writer.add_step(
        **_batch(offset=1.0),
        done=np.array([True]),
        terminated=np.array([False]),
        truncated=np.array([True]),
    )
    writer.close()

    with np.load(tmp_path / "0000000000.npz") as episode:
        np.testing.assert_array_equal(episode["terminated"], np.array([True, False]))
        np.testing.assert_array_equal(episode["truncated"], np.array([False, True]))


def test_simultaneous_dones_do_not_write_past_target_count(tmp_path: pathlib.Path) -> None:
    writer = EpisodeWriter(
        tmp_path,
        num_envs=3,
        obs_group_names=["policy"],
        max_workers=1,
        max_episodes=2,
    )

    flushed = writer.add_step(
        **_batch(num_envs=3),
        done=np.array([True, True, True]),
        terminated=np.array([True, True, True]),
        truncated=np.array([False, False, False]),
    )
    writer.close()

    assert flushed == 2
    assert writer.num_saved == 2
    assert sorted(path.name for path in tmp_path.glob("*.npz")) == ["0000000000.npz", "0000000001.npz"]
