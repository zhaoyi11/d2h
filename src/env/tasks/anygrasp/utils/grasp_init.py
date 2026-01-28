from __future__ import annotations

import sys
import numpy as np

from src.env.tasks.anygrasp.anygrasp_env_cfg import GraspInitData, transform_object_to_robot_frame


def load_grasp_init(
    grasp_path: str,
    object_scale_override: float | None = None,
    object_urdf_path: str | None = None,
) -> GraspInitData:
    """Load grasp data and compute object/robot init state."""
    # Compatibility shim for numpy 2.x pickles
    import numpy.core

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric

    # sys.modules.setdefault("numpy._core", _np_core)

    grasp_data = np.load(grasp_path, allow_pickle=True).item()

    object_scale = float(object_scale_override) if object_scale_override is not None else float(grasp_data["obj_scale"])
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
