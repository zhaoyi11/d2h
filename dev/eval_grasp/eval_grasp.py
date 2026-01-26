"""Evaluate grasp poses from BODex/cuRobo in Isaac Lab."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import argparse


from isaaclab.app import AppLauncher


def torch_quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first, as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    quaternions = torch.as_tensor(quaternions)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def torch_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


# add argparse arguments
parser = argparse.ArgumentParser(description="Replay grasp poses in Isaac Lab.")
parser.add_argument("--grasp_path", type=str, 
                    # default="/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_mug_3d3e993f7baa4d7ef1ff24a8b1564a36/floating/scale010/0_grasp.npy",
                    # default="/home/yizhao/yi/DexGraspBench/output/debug_leap/graspdata/ddg_gd_jar_poisson_018/floating/scale010/13_grasp.npy",
                    # camera
                    default="/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_camera_fb3b5fae94f7b02a3b269928487f8a4c/floating/scale008/1_grasp.npy",
                    help="Path to grasp data file (.npy)")
parser.add_argument("--grasp_idx", type=int, default=0, help="Index of grasp in batch (for BODex format)")
parser.add_argument("--seed_idx", type=int, default=0, help="Index of seed (for BODex format)")
parser.add_argument("--rot_correction", type=str, default="none",
                    choices=["none", "z90", "z-90", "z180", "x90", "x180", "x-90", "y90", "y-90", "y180", "flip_quat"],
                    help="Rotation correction to apply to hand pose (default: flip_quat)")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets.rigid_object import RigidObject, RigidObjectCfg

##
# Pre-defined configs
##
from src.assets.leap_hand.leap import LEAP_HAND_CFG
# from src.assets.allegro_hand.allegro import ALLEGRO_HAND_CFG
# from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
# from whole_body_tracking.tasks.tracking.mdp import MotionLoader



# ============================================================================
# MuJoCo to IsaacLab Coordinate Conversion
# ============================================================================
# 
# The MuJoCo LEAP hand and IsaacLab USD LEAP hand have different coordinate conventions:
#
# MuJoCo (right_hand.xml):
#   - Palm body: pos="0 0 0.1", quat="0 1 0 0" (180° around X)
#   - Palm +Z points in world -Z direction (fingers point down)
#
# USD (leap_hand_right.usd):
#   - Base xform: identity
#   - palm_lower: translate=(0, 0.038, 0.098), orient=(0.7071, 0, -0.7071, 0) (-90° around Y)
#   - Combined frame transformation: 180° around Y (MuJoCo -Z to USD -X requires total 180°Y)
#
# To convert MuJoCo palm pose to IsaacLab base pose:
#   USD_base = MuJoCo_palm_world * inv(USD_palm_lower_local)

def np_quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def np_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m = matrix
    trace = np.trace(m)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


# Correction rotation to transform MuJoCo palm frame to USD base frame
# Rotation is correct with [0, -0.7071, 0, -0.7071]
# Offset: USD palm_lower [0, 0.038, 0.098] + MuJoCo palm Z offset (0.1) mapped to X
USD_PALM_LOWER_OFFSET = np.array([-0.1, 0.038, 0.098])  # Negative X offset
# USD_PALM_LOWER_OFFSET = np.array([-0.098, 0.038, 0.1])
USD_PALM_LOWER_QUAT = np.array([0.0, -0.7071, 0.0, -0.7071])
USD_PALM_LOWER_ROT = np_quaternion_to_matrix(USD_PALM_LOWER_QUAT)


def mujoco_palm_pose_to_isaaclab_base(
    mujoco_pos: np.ndarray,
    mujoco_quat: np.ndarray,
) -> tuple:
    """
    Convert MuJoCo LEAP hand palm world pose to IsaacLab/USD base pose.
    
    The MuJoCo palm and USD palm_lower have different local coordinate conventions.
    To match the physical palm pose, we compute:
        USD_base_pose = MuJoCo_palm_world_pose × inv(USD_palm_lower_local_transform)
    
    Args:
        mujoco_pos: Palm position in MuJoCo world frame [x, y, z]
        mujoco_quat: Palm quaternion in MuJoCo convention [w, x, y, z]
        
    Returns:
        isaaclab_pos: Base position for IsaacLab/USD [x, y, z]
        isaaclab_quat: Base quaternion for IsaacLab/USD [w, x, y, z]
    """
    # Get MuJoCo palm rotation matrix
    R_palm_mj = np_quaternion_to_matrix(mujoco_quat)
    
    # Compute inverse of the correction rotation (180°Y is self-inverse)
    R_usd_palm_inv = USD_PALM_LOWER_ROT.T
    
    # Compute USD base rotation: R_base = R_palm_mj * R_usd_palm_inv
    R_isaaclab_base = R_palm_mj @ R_usd_palm_inv
    
    # Compute USD base position
    # For transform composition: base_pos = palm_pos - R_base * USD_PALM_LOWER_OFFSET
    isaaclab_pos = mujoco_pos - R_isaaclab_base @ USD_PALM_LOWER_OFFSET
    
    isaaclab_quat = np_matrix_to_quaternion(R_isaaclab_base)
    
    return isaaclab_pos, isaaclab_quat


def convert_joint_order_mujoco_to_isaaclab(
    mujoco_joint_pos: np.ndarray,
    isaaclab_joint_names=None,
) -> np.ndarray:
    """
    Convert joint positions from MuJoCo/cuRobo order to IsaacLab order.
    
    MuJoCo/cuRobo joint order (from leap.yml):
    ['j1', 'j0', 'j2', 'j3', 'j5', 'j4', 'j6', 'j7', 'j9', 'j8', 'j10', 'j11', 'j12', 'j13', 'j14', 'j15']
    """
    curobo_joint_order = ['j1', 'j0', 'j2', 'j3', 'j5', 'j4', 'j6', 'j7', 
                          'j9', 'j8', 'j10', 'j11', 'j12', 'j13', 'j14', 'j15']
    
    if isaaclab_joint_names is not None:
        mapping = []
        for isaac_name in isaaclab_joint_names:
            match = re.search(r'(\d+)', isaac_name)
            if match:
                joint_num = int(match.group(1))
                target_joint = f'j{joint_num}'
                if target_joint in curobo_joint_order:
                    curobo_idx = curobo_joint_order.index(target_joint)
                    mapping.append(curobo_idx)
                else:
                    mapping.append(len(mapping))
            else:
                mapping.append(len(mapping))
    else:
        mapping = [curobo_joint_order.index(f'j{i}') for i in range(16)]
    
    return np.array(mujoco_joint_pos)[mapping]


def mujoco_to_isaaclab_full_state(
    mujoco_pos: np.ndarray,
    mujoco_quat: np.ndarray,
    mujoco_joint_pos: np.ndarray,
    isaaclab_joint_names=None,
) -> tuple:
    """
    Convert full robot state from MuJoCo to IsaacLab format.
    
    Args:
        mujoco_pos: Palm position in MuJoCo world frame [x, y, z]
        mujoco_quat: Palm quaternion in MuJoCo convention [w, x, y, z]
        mujoco_joint_pos: Joint positions in MuJoCo/cuRobo order (16 values)
        isaaclab_joint_names: Optional list of IsaacLab joint names for reordering
        
    Returns:
        isaaclab_pos: Base position for IsaacLab/USD [x, y, z]
        isaaclab_quat: Base quaternion for IsaacLab/USD [w, x, y, z]
        isaaclab_joint_pos: Joint positions in IsaacLab order (16 values)
    """
    # Convert base pose
    isaaclab_pos, isaaclab_quat = mujoco_palm_pose_to_isaaclab_base(mujoco_pos, mujoco_quat)
    
    # Convert joint positions (reorder if needed)
    isaaclab_joint_pos = convert_joint_order_mujoco_to_isaaclab(
        mujoco_joint_pos, isaaclab_joint_names
    )
    
    return isaaclab_pos, isaaclab_quat, isaaclab_joint_pos

def convert_bodex_to_dexgraspbench_format(bodex_data: dict, grasp_idx: int = 0, seed_idx: int = 0, pose_idx: int = 1) -> dict:
    """Convert BODex/plan_batch_env.py output format to DexGraspBench format.
    
    BODex format:
    - robot_pose: shape (batch, num_seeds, num_poses, 23)
    - world_cfg: list of dicts with mesh info
    - joint_names: list of joint names
    
    DexGraspBench format:
    - grasp_qpos: shape (23,) - [pos(3), quat(4), joints(16)]
    - obj_scale: float
    - obj_pose: shape (7,) - [pos(3), quat(4)]
    - obj_path: str
    
    Args:
        bodex_data: Data loaded from BODex .npy file
        grasp_idx: Index of grasp in batch (default: 0)
        seed_idx: Index of seed (default: 0)
        pose_idx: Index of pose (0=pregrasp, 1=grasp, 2=squeeze, default: 1)
    """
    # Extract robot pose
    robot_pose = bodex_data["robot_pose"]
    if len(robot_pose.shape) == 4:
        # Shape: (batch, num_seeds, num_poses, 23)
        grasp_qpos = robot_pose[grasp_idx, seed_idx, pose_idx]
    elif len(robot_pose.shape) == 3:
        # Shape: (num_seeds, num_poses, 23)
        grasp_qpos = robot_pose[seed_idx, pose_idx]
    else:
        grasp_qpos = robot_pose
    
    # Extract object info from world_cfg
    world_cfg = bodex_data["world_cfg"]
    if isinstance(world_cfg, list):
        world_cfg = world_cfg[grasp_idx] if grasp_idx < len(world_cfg) else world_cfg[0]
    
    # Get mesh info
    mesh_dict = world_cfg.get("mesh", {})
    if mesh_dict:
        # Get the first mesh (usually only one object)
        mesh_name = list(mesh_dict.keys())[0]
        mesh_info = mesh_dict[mesh_name]
        obj_scale = mesh_info["scale"]
        if isinstance(obj_scale, (list, np.ndarray)):
            obj_scale = float(obj_scale[0])  # Assume uniform scale
        obj_pose = np.array(mesh_info["pose"])
        obj_path = mesh_info.get("urdf_path", mesh_info.get("file_path", ""))
        # Extract object ID from path
        if "processed_data" in obj_path:
            parts = obj_path.split("processed_data/")
            if len(parts) > 1:
                obj_path = "assets/object/DGN_2k/processed_data/" + parts[1].split("/")[0]
    else:
        obj_scale = 1.0
        obj_pose = np.array([0, 0, 0, 1, 0, 0, 0])
        obj_path = ""
    
    return {
        "grasp_qpos": np.array(grasp_qpos),
        "pregrasp_qpos": np.array(robot_pose[grasp_idx, seed_idx, 0]) if len(robot_pose.shape) >= 3 else grasp_qpos,
        "squeeze_qpos": np.array(robot_pose[grasp_idx, seed_idx, 2]) if len(robot_pose.shape) >= 3 and robot_pose.shape[-2] > 2 else grasp_qpos,
        "obj_scale": obj_scale,
        "obj_pose": obj_pose,
        "obj_path": obj_path,
    }


def detect_and_load_grasp_data(path: str, grasp_idx: int = 0, seed_idx: int = 0) -> dict:
    """Load grasp data and convert to a unified format.
    
    Supports both:
    - DexGraspBench format (grasp_qpos, obj_scale, obj_pose)
    - BODex format (robot_pose, world_cfg)
    """
    data = np.load(path, allow_pickle=True).item()
    
    # Check format by looking at keys
    if "grasp_qpos" in data:
        print("Detected DexGraspBench format")
        return data
    elif "robot_pose" in data:
        print("Detected BODex format, converting...")
        return convert_bodex_to_dexgraspbench_format(data, grasp_idx, seed_idx)
    else:
        raise ValueError(f"Unknown data format. Keys: {list(data.keys())}")


def get_object_urdf_path(obj_path: str) -> str:
    """Convert obj_path from grasp data to full URDF path.
    
    The obj_path in grasp data is like:
    'assets/object/DGN_2k/scene_cfg/.../processed_data/<obj_id>'
    
    We need to construct the full URDF path.
    """
    # Handle relative path - the obj_path may be relative to some assets folder
    if obj_path.startswith("assets/"):
        # Try to find the base directory
        base_dirs = [
            "/home/yizhao/yi/DexGraspBench",
            "/home/yizhao/yi/BODex/src/curobo/content",
        ]
        for base_dir in base_dirs:
            full_path = os.path.join(base_dir, obj_path, "urdf/coacd.urdf")
            if os.path.exists(full_path):
                return full_path
    
    # If obj_path is already a full path
    urdf_path = os.path.join(obj_path, "urdf/coacd.urdf")
    if os.path.exists(urdf_path):
        return urdf_path
    
    # Fallback: try to extract object ID and construct path
    # The obj_path may end with the object ID
    obj_id = os.path.basename(obj_path.rstrip('/'))
    fallback_path = f"/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/{obj_id}/urdf/coacd.urdf"
    if os.path.exists(fallback_path):
        return fallback_path
    
    raise FileNotFoundError(f"Could not find URDF for object path: {obj_path}")


def create_scene_cfg(grasp_data: dict, flip_quat: bool = True) -> InteractiveSceneCfg:
    """Create scene configuration with object state from grasp data."""
    # # Extract object state from grasp data
    # object_scale = float(grasp_data["obj_scale"])
    # object_pos = tuple(grasp_data["obj_pose"][:3].tolist())
    
    # # Object quaternion - check format
    # # Note: Object pose seems to use [w,x,y,z] format (identity = [1,0,0,0])
    # # while robot pose uses [x,y,z,w] format
    # obj_quat_raw = grasp_data["obj_pose"][3:7]
    # print(f"Object quaternion (raw): {obj_quat_raw}")
    # import ipdb; ipdb.set_trace()
    # # Check if it looks like [w,x,y,z] format (w close to 1 for identity)
    # # or [x,y,z,w] format (w would be the 4th element)
    # # For identity quaternion [1,0,0,0] in [w,x,y,z], first element is 1
    # # For identity quaternion [0,0,0,1] in [x,y,z,w], last element is 1
    # if abs(obj_quat_raw[0]) > 0.9 and abs(obj_quat_raw[3]) < 0.1:
    #     # Looks like [w,x,y,z] format already (identity has w=1 as first element)
    #     object_quat = tuple(obj_quat_raw.tolist())
    #     print(f"Object quaternion (detected as wxyz, no flip): {object_quat}")
    # elif flip_quat:
    #     # Convert from [x, y, z, w] to [w, x, y, z] format for Isaac Lab
    #     object_quat = tuple([float(obj_quat_raw[3]), float(obj_quat_raw[0]), 
    #                        float(obj_quat_raw[1]), float(obj_quat_raw[2])])
    #     print(f"Object quaternion (flipped to wxyz): {object_quat}")
    # else:
    #     # Assume [w, x, y, z] format
    #     object_quat = tuple(obj_quat_raw.tolist())
    
    # # Get object URDF path
    # obj_path = grasp_data.get("obj_path", "")
    # try:
    #     # urdf_path = get_object_urdf_path(obj_path)
    #     urdf_path = "/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/ddg_gd_jar_poisson_018/urdf/coacd.urdf"
    # except FileNotFoundError as e:
    #     print(f"Warning: {e}")
    #     # Use a default path as fallback
    #     # urdf_path = "/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/core_mug_3d3e993f7baa4d7ef1ff24a8b1564a36/urdf/coacd.urdf"
    #     urdf_path = "/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/ddg_gd_jar_poisson_018/urdf/coacd.urdf"
    
    # print(f"\n=== Object Configuration ===")
    # print(f"Using object URDF: {urdf_path}")
    # print(f"Object scale: {object_scale}")
    # print(f"Object position: {object_pos}")
    # print(f"Object quaternion (wxyz): {object_quat}")
    
    # # Print robot grasp position for reference
    # robot_pos = grasp_data["grasp_qpos"][:3]
    # print(f"\n=== Robot Grasp Position (for reference) ===")
    # print(f"Robot position: {robot_pos}")
    # print(f"Distance from object: {np.linalg.norm(robot_pos - np.array(object_pos)):.4f}m")
    
    # obj_urdf_path = "/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/ddg_gd_jar_poisson_018/urdf/coacd.urdf
    obj_urdf_path = "/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/core_camera_fb3b5fae94f7b02a3b269928487f8a4c/urdf/coacd.urdf"
    # get object scale and position from grasp data
    object_scale = float(grasp_data["obj_scale"])
    object_pos = tuple(grasp_data["obj_pose"][:3].tolist())
    object_quat = tuple(grasp_data["obj_pose"][3:7].tolist())

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

        # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

        sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            ),
        )

        # articulation - use floating hand (fix_root_link=False) so we can set the root pose
        robot: ArticulationCfg = LEAP_HAND_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=LEAP_HAND_CFG.spawn.replace(
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0005,
                    fix_root_link=False,  # Allow floating hand
                ),
            ),
        )

        object = RigidObjectCfg(
            prim_path="/World/object",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=obj_urdf_path,
                scale=(object_scale, object_scale, object_scale),
                fix_base=True,  # Fix object in place for visualization
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,  # Disable articulation for rigid object
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=object_quat),
        )
    
    return ReplayMotionsSceneCfg


@dataclass
class Config:
    """Configuration for the grasp evaluation."""
    device: str = "cuda:0"



def curobo_to_isaaclab_qpos(curobo_qpos, isaaclab_joint_names=None):
    """Convert cuRobo LEAP hand joint positions to Isaac Lab joint order.
    
    cuRobo joint order: ['j1', 'j0', 'j2', 'j3', 'j5', 'j4', 'j6', 'j7', 'j9', 'j8', 'j10', 'j11', 'j12', 'j13', 'j14', 'j15']
    
    Isaac Lab joint order depends on the USD/URDF used. Common orderings:
    - dex-urdf: [0, 1, 2, ..., 15] (numeric names)
    - cuRobo simplified: [j0, j1, ..., j15]
    
    This function computes the mapping dynamically based on the Isaac Lab joint names.
    """
    # cuRobo joint order as defined in leap.yml
    curobo_joint_order = ['j1', 'j0', 'j2', 'j3', 'j5', 'j4', 'j6', 'j7', 'j9', 'j8', 'j10', 'j11', 'j12', 'j13', 'j14', 'j15']
    
    if isaaclab_joint_names is not None:
        # Compute mapping dynamically based on Isaac Lab joint names
        mapping = []
        for isaac_name in isaaclab_joint_names:
            # Extract the joint number from the name (handles formats like "0", "j0", "a_0", etc.)
            # Try to extract numeric part
            match = re.search(r'(\d+)', isaac_name)
            if match:
                joint_num = int(match.group(1))
                target_joint = f'j{joint_num}'
                if target_joint in curobo_joint_order:
                    curobo_idx = curobo_joint_order.index(target_joint)
                    mapping.append(curobo_idx)
                else:
                    print(f"Warning: Could not find {target_joint} in cuRobo joint order")
                    mapping.append(len(mapping))  # Identity mapping as fallback
            else:
                print(f"Warning: Could not extract joint number from '{isaac_name}'")
                mapping.append(len(mapping))  # Identity mapping as fallback
        
        print(f"Dynamic joint mapping: {mapping}")
    else:
        # Default mapping assuming Isaac Lab uses [0, 1, 2, ..., 15] order
        # isaaclab_qpos[i] = curobo_qpos[mapping[i]]
        # where mapping[i] is the index in curobo_joint_order where 'j{i}' is located
        mapping = [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]
    
    return np.array(curobo_qpos)[mapping]

def quaternion_multiply(q1, q2):
    """Multiply two quaternions in [w, x, y, z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotate_vector_by_quaternion(q, v):
    """Rotate a 3D vector by a quaternion in [w, x, y, z] format.
    
    This rotates the vector around the world origin.
    Formula: v' = q * v * q_conjugate (treating v as pure quaternion [0, vx, vy, vz])
    """
    # Convert vector to pure quaternion [0, vx, vy, vz]
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    # Quaternion conjugate: [w, -x, -y, -z]
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    # v' = q * v * q_conjugate
    result = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    # Return the vector part [x, y, z]
    return result[1:4]


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, grasp_data: dict):
    """Load scene, check robot state, and apply grasp pose."""
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Safely get object if it exists
    try:
        obj = scene["object"]
    except KeyError:
        obj = None
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    
    # Print Isaac Lab joint names for debugging
    print(f"\n=== Isaac Lab Robot Info ===")
    print(f"Joint names: {robot.joint_names}")
    print(f"Number of joints: {robot.num_joints}")
    
    # Extract robot state from grasp data
    # cuRobo/MuJoCo format: [x, y, z, qw, qx, qy, qz, j0, j1, ..., j15]
    mujoco_pos = grasp_data["grasp_qpos"][:3]
    mujoco_quat = grasp_data["grasp_qpos"][3:7]
    mujoco_joint_pos = grasp_data["grasp_qpos"][7:]
    
    print(f"\n=== Grasp Data (MuJoCo/cuRobo format) ===")
    print(f"MuJoCo palm position: {mujoco_pos}")
    print(f"MuJoCo palm quaternion: {mujoco_quat}")
    print(f"MuJoCo joint positions: {mujoco_joint_pos}")
    
    # The cuRobo LEAP hand URDF and Isaac Lab LEAP hand USD may have different base frame orientations.
    # cuRobo stores quaternion as [w, x, y, z].
    # We may need to apply a correction rotation to match Isaac Lab's LEAP hand coordinate frame.
    
    # Try interpreting as [w, x, y, z] (cuRobo convention)
    robot_quat = robot_quat_raw  # [w, x, y, z]
    
    # The Isaac Lab LEAP hand from dex-urdf has a different base orientation.
    # The default init rotation in LEAP_HAND_CFG is (0.5, 0.5, -0.5, 0.5)
    # This suggests a 120-degree rotation around (1, -1, 1) axis.
    # We need to find the transformation between cuRobo's and Isaac Lab's coordinate frames.
    
    # Based on the URDF comparison:
    # - cuRobo leap_hand_simplified.urdf: palm faces +X direction
    # - dex-urdf leap_hand_right: palm faces -Y direction (typically)
    # A rotation of 90 degrees around Z axis might be needed.
    
    # Correction rotation: Try rotating 90 degrees around Z axis
    # Quaternion for 90-deg rotation around Z: [cos(45), 0, 0, sin(45)] = [0.707, 0, 0, 0.707]
    # Or try 180 degrees: [0, 0, 0, 1]
    # Or try -90 degrees: [0.707, 0, 0, -0.707]
    
    # Apply rotation correction based on command-line argument
    # The cuRobo and Isaac Lab LEAP hand models may have different base frame orientations
    sqrt2_2 = 0.7071067811865476  # sqrt(2)/2
    correction_options = {
        "none": np.array([1.0, 0.0, 0.0, 0.0]),        # Identity
        "x90": np.array([sqrt2_2, sqrt2_2, 0.0, 0.0]), # 90-deg around X
        "x-90": np.array([sqrt2_2, -sqrt2_2, 0.0, 0.0]), # -90-deg around X
        "x180": np.array([0.0, 1.0, 0.0, 0.0]),        # 180-deg around X
        "y90": np.array([sqrt2_2, 0.0, sqrt2_2, 0.0]), # 90-deg around Y
        "y-90": np.array([sqrt2_2, 0.0, -sqrt2_2, 0.0]), # -90-deg around Y
        "y180": np.array([0.0, 0.0, 1.0, 0.0]),        # 180-deg around Y
        "z90": np.array([sqrt2_2, 0.0, 0.0, sqrt2_2]), # 90-deg around Z
        "z-90": np.array([sqrt2_2, 0.0, 0.0, -sqrt2_2]), # -90-deg around Z
        "z180": np.array([0.0, 0.0, 0.0, 1.0]),        # 180-deg around Z
        "flip_quat": None,  # Special case: swap quaternion format
    }
    
    rot_correction = args_cli.rot_correction

    
    # Convert joint positions from cuRobo order to Isaac Lab order
    robot_joint_pos = curobo_to_isaaclab_qpos(robot_joint_pos_raw, robot.joint_names)
    print(f"Robot joint positions (Isaac Lab order): {robot_joint_pos}")
    
    # Convert robot state to tensors
    root_pos = torch.tensor(robot_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
    root_quat = torch.tensor(robot_quat, device=sim.device, dtype=torch.float32).unsqueeze(0)
    joint_pos = torch.tensor(robot_joint_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
    
    # Set robot root pose (position and orientation)
    # Isaac Lab expects [x, y, z, w, x, y, z] format for pose
    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    print(f"\n=== Setting Robot Pose ===")
    print(f"Root pose tensor: {root_pose}")
    robot.write_root_pose_to_sim(root_pose)
    
    # Set robot joint positions (with zero velocities)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Force physics update to apply changes
    sim.step()
    scene.update(sim.get_physics_dt())
    
    # Verify the pose was applied
    print(f"\n=== Verifying Applied Pose ===")
    actual_pos = robot.data.root_pos_w[0].cpu().numpy()
    actual_quat = robot.data.root_quat_w[0].cpu().numpy()
    print(f"Requested position: {robot_pos}")
    print(f"Actual position:    {actual_pos}")
    print(f"Position error:     {np.linalg.norm(actual_pos - robot_pos):.6f}m")
    print(f"Requested quaternion: {robot_quat}")
    print(f"Actual quaternion:    {actual_quat}")
    
    # Check initial robot state
    print("\n=== Initial Robot State ===")
    print(f"Number of environments: {scene.num_envs}")
    print(f"Robot body names: {robot.body_names}")
    print(f"Robot joint names: {robot.joint_names}")
    print(f"Robot joint positions shape: {robot.data.joint_pos.shape}")
    print(f"Robot root position: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"Robot root orientation: {robot.data.root_quat_w[0].cpu().numpy()}")
    
    if obj is not None:
        print(f"\n=== Object State ===")
        print(f"Object root position: {obj.data.root_pos_w[0].cpu().numpy()}")
        print(f"Object root orientation: {obj.data.root_quat_w[0].cpu().numpy()}")
    
    # Get action space (joint positions)
    num_joints = robot.num_joints
    print(f"\n=== Action Space ===")
    print(f"Number of joints: {num_joints}")
    print(f"Actuator groups: {list(robot.actuators.keys())}")
    
    # Simulation loop
    step_count = 0
    while simulation_app.is_running():
        # Keep robot at initial pose (both root pose and joint positions)
        robot.write_root_pose_to_sim(root_pose)  # Maintain root position/orientation
        robot.set_joint_position_target(joint_pos)
        
        # Write data to simulation
        scene.write_data_to_sim()
        
        # Step simulation
        sim.step()
        
        # Update scene
        scene.update(sim_dt)
        
        # Render
        sim.render()
        
        # Update camera view
        root_pos = robot.data.root_pos_w[0].cpu().numpy()
        sim.set_camera_view(root_pos + np.array([2.0, 2.0, 0.5]), root_pos)
        
        # Print state every 100 steps
        step_count += 1
        if step_count % 100 == 0:
            print(f"\n=== Step {step_count} ===")
            print(f"Robot root position: {robot.data.root_pos_w[0].cpu().numpy()}")
            print(f"Robot joint positions: {robot.data.joint_pos[0].cpu().numpy()}")
            if obj is not None:
                print(f"Object root position: {obj.data.root_pos_w[0].cpu().numpy()}")

if __name__ == "__main__":
    config = Config()
    
    # Fix for numpy version compatibility (numpy 1.x <-> 2.x)
    import sys
    try:
        import numpy._core
        sys.modules['numpy.core'] = numpy._core
        sys.modules['numpy.core.multiarray'] = numpy._core.multiarray
        sys.modules['numpy.core.numeric'] = numpy._core.numeric
    except ImportError:
        import numpy.core
        sys.modules['numpy._core'] = numpy.core
        sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
        sys.modules['numpy._core.numeric'] = numpy.core.numeric
    # Example paths:
    # DexGraspBench format:
    # python eval_grasp.py --grasp_path "/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_mug_3d3e993f7baa4d7ef1ff24a8b1564a36/floating/scale010/0_grasp.npy"
    # BODex format:
    # python eval_grasp.py --grasp_path "/home/yizhao/yi/BODex/src/curobo/content/assets/output/sim_leap/fc/debug/graspdata/sem_TissueBox_ffb4e07d613b6a62bbfa57fc7493b378/floating/scale008_grasp.npy"
    
    grasp_data = detect_and_load_grasp_data(args_cli.grasp_path, args_cli.grasp_idx, args_cli.seed_idx)

    # import ipdb; ipdb.set_trace()
    sim_cfg = sim_utils.SimulationCfg(device=config.device)
    sim_cfg.dt = 0.02
    sim_cfg.gravity = (0.0, 0.0, 0.0)  # Disable gravity
    sim = SimulationContext(sim_cfg)

    # Create scene config with object state from grasp data
    flip_quat = (args_cli.rot_correction == "flip_quat")
    SceneCfg = create_scene_cfg(grasp_data, flip_quat=flip_quat)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene, grasp_data)
