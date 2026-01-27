"""Evaluate grasp poses from BODex/cuRobo in Isaac Lab."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
import torch
import argparse


from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="Replay grasp poses in Isaac Lab.")
parser.add_argument("--grasp_path", type=str, 
                    # default="/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_mug_3d3e993f7baa4d7ef1ff24a8b1564a36/floating/scale010/0_grasp.npy",
                    # default="/home/yizhao/yi/DexGraspBench/output/debug_leap/graspdata/ddg_gd_jar_poisson_018/floating/scale010/13_grasp.npy",
                    # camera
                    default="/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_camera_fb3b5fae94f7b02a3b269928487f8a4c/floating/scale008/1_grasp.npy",
                    help="Path to grasp data file (.npy)")
parser.add_argument("--obj_urdf_path", type=str, default="/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/core_camera_fb3b5fae94f7b02a3b269928487f8a4c/urdf/coacd.urdf", help="Path to object file (.urdf)")

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
    Convert joint positions from MuJoCo order to IsaacLab order.
    
    MuJoCo joint order (from leap.yml):
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


def create_scene_cfg(grasp_data: dict, obj_urdf_path: str) -> InteractiveSceneCfg:
    """Create scene configuration with object state from grasp data."""
    
    # get object scale and position from grasp data
    object_scale = float(grasp_data["obj_scale"])
    object_pos = tuple(grasp_data["obj_pose"][:3].tolist())
    object_quat = tuple(grasp_data["obj_pose"][3:7].tolist())

    @configclass
    class ReplayMotionsSceneCfg(InteractiveSceneCfg):
        """Configuration for a replay motions scene."""

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
                fix_base=False,  # Fix object in place for visualization
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    articulation_enabled=False,  # Disable articulation for rigid object
                ),
            ),
            # TODO: change this
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.0, 0.0), rot=object_quat),
            # init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=object_quat),
        )
    
    return ReplayMotionsSceneCfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, grasp_data: dict):
    """Load scene, check robot state, and apply grasp pose."""
    # Extract scene entities
    robot: Articulation = scene["robot"]
    obj = scene["object"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
        
    # Extract robot state from grasp data [x, y, z, qw, qx, qy, qz, j0, j1, ..., j15]
    robot_pos = grasp_data["grasp_qpos"][:3]
    robot_quat = grasp_data["grasp_qpos"][3:7]
    robot_joint_pos = grasp_data["grasp_qpos"][7:]
    
    print(f"\n=== Grasp Data (Original) ===")
    print(f"Robot position: {robot_pos}")
    print(f"Robot quaternion: {robot_quat}")
    print(f"Robot joint positions: {robot_joint_pos}")
    
    # Convert from MuJoCo to IsaacLab coordinate convention
    # This accounts for the MuJoCo palm body definition: pos="0 0 0.1" quat="0 1 0 0"
    robot_pos, robot_quat, robot_joint_pos = mujoco_to_isaaclab_full_state(
        robot_pos, robot_quat, robot_joint_pos, robot.joint_names
    )
    
    print(f"\n=== Converted to IsaacLab format ===")
    print(f"IsaacLab base position: {robot_pos}")
    print(f"IsaacLab base quaternion: {robot_quat}")
    print(f"IsaacLab joint positions: {robot_joint_pos}")  
    
    # Convert robot state to tensors
    root_pos = torch.tensor(robot_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
    root_quat = torch.tensor(robot_quat, device=sim.device, dtype=torch.float32).unsqueeze(0)
    joint_pos = torch.tensor(robot_joint_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
    
    # Set robot root pose (position and orientation)
    # Isaac Lab expects [x, y, z, w, x, y, z] format for pose
    root_pose = torch.cat([root_pos, root_quat], dim=-1)
    robot.write_root_pose_to_sim(root_pose)
    # Set robot joint positions (with zero velocities)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Force physics update to apply changes
    sim.step()
    scene.update(sim.get_physics_dt())
    
    print(f"\n=== Object State ===")
    print(f"Object root position: {obj.data.root_pos_w[0].cpu().numpy()}")
    print(f"Object root orientation: {obj.data.root_quat_w[0].cpu().numpy()}")
    

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
            print(f"Object root position: {obj.data.root_pos_w[0].cpu().numpy()}")

if __name__ == "__main__":

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

    # Load grasp data
    grasp_data = np.load(args_cli.grasp_path, allow_pickle=True).item()

    sim_cfg = sim_utils.SimulationCfg(device="cuda:0")
    sim_cfg.dt = 0.02
    sim_cfg.gravity = (0.0, 0.0, 0.0)  # Disable gravity
    sim = SimulationContext(sim_cfg)

    # Create scene config with object state from grasp data
    SceneCfg = create_scene_cfg(grasp_data=grasp_data, 
                                obj_urdf_path=args_cli.obj_urdf_path)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    run_simulator(sim, scene, grasp_data)
