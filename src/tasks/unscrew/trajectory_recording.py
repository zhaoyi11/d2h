"""NumPy schema helpers for physical screw-in calibration recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _pose(name: str, value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value)
    if pose.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {pose.shape}.")
    if not np.isfinite(pose).all():
        raise ValueError(f"{name} must contain only finite values.")
    return pose


def _pose_sequence(name: str, value: np.ndarray) -> np.ndarray:
    poses = np.asarray(value)
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] < 1:
        raise ValueError(f"{name} must have shape (N, 7) with N >= 1, got {poses.shape}.")
    if not np.isfinite(poses).all():
        raise ValueError(f"{name} must contain only finite values.")
    return poses


def _continuous_quaternions(poses: np.ndarray) -> np.ndarray:
    poses = poses.copy()
    for index in range(1, poses.shape[0]):
        if np.dot(poses[index - 1, 3:7], poses[index, 3:7]) < 0.0:
            poses[index, 3:7] *= -1.0
    return poses


def build_screw_in_recording(
    *,
    stable_pose_w: np.ndarray,
    stable_pose_receptive: np.ndarray,
    screw_targets_w: np.ndarray,
    screw_measured_w: np.ndarray,
    screw_measured_receptive: np.ndarray,
    dt: float,
    thread_pitch: float,
    commanded_screw_rotation: float,
    measured_screw_rotation: float,
    asset_scale: float,
) -> dict[str, np.ndarray]:
    """Build a pickle-free recording whose unscrew path reverses measured screw-in samples."""
    stable_w = _pose("stable_pose_w", stable_pose_w)
    stable_r = _pose("stable_pose_receptive", stable_pose_receptive)
    targets_w = _pose_sequence("screw_targets_w", screw_targets_w)
    measured_w = _pose_sequence("screw_measured_w", screw_measured_w)
    measured_r = _pose_sequence("screw_measured_receptive", screw_measured_receptive)
    if not (targets_w.shape[0] == measured_w.shape[0] == measured_r.shape[0]):
        raise ValueError("Screw target and measured trajectories must have the same number of samples.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")

    unscrew_w = _continuous_quaternions(
        np.concatenate((stable_w[None], measured_w[::-1]), axis=0)
    )
    unscrew_r = _continuous_quaternions(
        np.concatenate((stable_r[None], measured_r[::-1]), axis=0)
    )
    screw_time = np.arange(measured_w.shape[0], dtype=np.float64) * dt
    unscrew_time = np.arange(unscrew_w.shape[0], dtype=np.float64) * dt

    return {
        "schema_version": np.asarray(1, dtype=np.int64),
        "pose_format": np.asarray("xyz_qwqxqyqz"),
        "world_frame": np.asarray("simulation_world"),
        "receptive_frame": np.asarray("receptive_object_root"),
        "unscrew_source": np.asarray("reversed_measured_screw_in"),
        "dt": np.asarray(dt, dtype=np.float64),
        "thread_pitch": np.asarray(thread_pitch, dtype=np.float64),
        "commanded_screw_rotation": np.asarray(commanded_screw_rotation, dtype=np.float64),
        "measured_screw_rotation": np.asarray(measured_screw_rotation, dtype=np.float64),
        "asset_scale": np.asarray(asset_scale, dtype=np.float64),
        "stable_pose_w": stable_w,
        "stable_pose_receptive": stable_r,
        "screw_time_s": screw_time,
        "screw_targets_w": targets_w,
        "screw_measured_w": measured_w,
        "screw_measured_receptive": measured_r,
        "unscrew_time_s": unscrew_time,
        "unscrew_measured_w": unscrew_w,
        "unscrew_measured_receptive": unscrew_r,
    }


def save_screw_in_recording(path: str | Path, recording: dict[str, np.ndarray]) -> None:
    """Save a screw-in calibration recording as a compressed NumPy archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **recording)


__all__ = ["build_screw_in_recording", "save_screw_in_recording"]
