"""Read recorded object poses without loading the simulator or task code."""

import argparse

import numpy as np


def load_object_trajectory(path, fps=None):
    """Return relative timestamps and world-frame XYZ/WXYZ object poses."""
    arrays = {}
    with np.load(path, allow_pickle=False) as data:
        if "qpos_obj_right" in data:
            poses = data["qpos_obj_right"]
            if poses.ndim != 2 or poses.shape[1] != 7 or len(poses) == 0:
                raise ValueError("qpos_obj_right must have shape (N, 7) with N > 0")
            rate = positive_float(fps if fps is not None else data.get("frequency", 30.0))
            data = {
                "time": data.get("time", np.arange(len(poses)) / rate),
                "object_pos": poses[:, :3],
                "object_quat_wxyz": poses[:, 3:],
            }
        for key, shape in (("time", ()), ("object_pos", (3,)), ("object_quat_wxyz", (4,))):
            if key not in data:
                raise ValueError(f"Missing trajectory array: {key}")
            value = data[key]
            if value.ndim != len(shape) + 1 or value.shape[1:] != shape or len(value) == 0:
                raise ValueError(f"{key} must have shape (N, {shape}) with N > 0")
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain finite numeric values")
            arrays[key] = value.astype(np.float64)
    timestamps = arrays["time"]
    if any(len(value) != len(timestamps) for value in arrays.values()):
        raise ValueError("Trajectory arrays must have matching frame counts")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Trajectory timestamps must be strictly increasing")
    quaternions = arrays["object_quat_wxyz"]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms < 1e-12):
        raise ValueError("Object trajectory contains invalid quaternions")
    timestamps = timestamps - timestamps[0] if fps is None else np.arange(len(timestamps)) / positive_float(fps)
    return timestamps, np.concatenate((arrays["object_pos"], quaternions / norms), axis=1)



def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value

