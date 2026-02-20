from __future__ import annotations

import sys
import numpy as np

from dataclasses import dataclass


USD_PALM_LOWER_OFFSET = np.array([-0.1, 0.038, 0.098])
USD_PALM_LOWER_QUAT = np.array([0.0, -0.7071, 0.0, -0.7071])


def _quat_mul_wxyz(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Quaternion multiply (wxyz): q = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _rotate_pos_y_pos_90(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a position by +90deg about world +Y: (x,y,z) -> (z, y, -x)."""
    x, y, z = pos
    return (z, y, -x)


@dataclass
class GraspInitData:
    """Precomputed grasp initialization data."""

    object_pos: tuple[float, float, float]
    object_quat: tuple[float, float, float, float]
    object_scale: float
    object_asset_path: str
    robot_joint_pos: np.ndarray


def np_quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def np_quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute the inverse of a quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def np_quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def transform_object_to_robot_frame(
    mj_palm_pos: np.ndarray,
    mj_palm_quat: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform object pose so base is identity while preserving palm-to-object relationship."""
    R_palm_mj = np_quaternion_to_matrix(mj_palm_quat)
    R_palm_mj_inv = R_palm_mj.T
    obj_rel_pos_mj = R_palm_mj_inv @ (object_pos - mj_palm_pos)

    palm_quat_inv = np_quaternion_inverse(mj_palm_quat)
    obj_rel_quat = np_quaternion_multiply(palm_quat_inv, object_quat)

    new_palm_pos = USD_PALM_LOWER_OFFSET.copy()
    new_palm_quat = USD_PALM_LOWER_QUAT.copy()
    new_palm_rot = np_quaternion_to_matrix(new_palm_quat)

    obj_rel_pos_transformed = new_palm_rot @ obj_rel_pos_mj
    new_object_pos = new_palm_pos + obj_rel_pos_transformed
    new_object_quat = np_quaternion_multiply(new_palm_quat, obj_rel_quat)

    new_object_quat = new_object_quat / np.linalg.norm(new_object_quat)
    if new_object_quat[0] < 0:
        new_object_quat = -new_object_quat

    # example
    # pos  array([0.14013682, 0.02385659, 0.09118012])
    # quat array([0.02073345, 0.96638001, 0.24646428, 0.07025066])
    return new_object_pos, new_object_quat


def load_grasp_init(
    grasp_path: str,
    object_urdf_path: str,
    object_scale_override: float | None = None,
    base_pos: tuple[float, float, float] = (0, 0, 0),
    base_rot: tuple[float, float, float, float] = (1, 0, 0, 0),  # TODO: change based rot and pos
) -> GraspInitData:
    """Load grasp data and compute object/robot init state."""
    # Compatibility shim for numpy 2.x pickles
    import numpy.core

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric

    # sys.modules.setdefault("numpy._core", _np_core)

    grasp_data = np.load(grasp_path, allow_pickle=True).item()

    object_scale = (
        float(object_scale_override)
        if object_scale_override is not None
        else float(grasp_data["obj_scale"])
    )
    orig_object_pos = np.array(grasp_data["obj_pose"][:3])
    orig_object_quat = np.array(grasp_data["obj_pose"][3:7])
    mj_palm_pos = np.array(grasp_data["grasp_qpos"][:3])
    mj_palm_quat = np.array(grasp_data["grasp_qpos"][3:7])

    new_object_pos, new_object_quat = transform_object_to_robot_frame(
        mj_palm_pos, mj_palm_quat, orig_object_pos, orig_object_quat
    )

    object_pos = tuple(new_object_pos.tolist())
    object_quat = tuple(new_object_quat.tolist())

    object_asset_path = object_urdf_path  # may be None; caller can choose fallback

    return GraspInitData(
        object_pos=object_pos,
        object_quat=object_quat,
        object_scale=object_scale,
        object_asset_path=object_asset_path,
        robot_joint_pos=np.array(grasp_data["grasp_qpos"][7:]),
    )
