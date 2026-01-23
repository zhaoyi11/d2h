""" Play motion files for debugging. """

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from dataclasses import dataclass

import loguru
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
# parser.add_argument("--registry_name", type=str, required=True, help="The name of the wand registry.")

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


def create_scene_cfg(grasp_data: dict) -> InteractiveSceneCfg:
    """Create scene configuration with object state from grasp data."""
    # Extract object state from grasp data
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

        # articulation
        robot: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        object = RigidObjectCfg(
            prim_path="/World/object",
            spawn=sim_utils.UrdfFileCfg(
                asset_path="/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/core_mug_3d3e993f7baa4d7ef1ff24a8b1564a36/urdf/coacd.urdf",
                scale=(object_scale, object_scale, object_scale),
                fix_base=False,  # Object should be free-floating
                joint_drive=None,  # No joints for rigid object
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=object_pos, rot=object_quat),
        )
    
    return ReplayMotionsSceneCfg


@dataclass
class Config:
    # === TASK CONFIGURATION ===
    embodiment_type: str = "bimanual"  # "left", "right", "bimanual", "CMU"

    # === SIMULATOR CONFIGURATION ===
    device: str = "cuda:0"
    # Simulation timing
    sim_dt: float = 0.01  # simulation timestep
    ref_dt: float = 1 / 50.0  # reference data timestep

    # === AUTOMATICALLY SET PROPERTIES ===
    # Computed timesteps
    horizon_steps: int = 160
    ref_steps: int = 2
    ctrl_steps: int = 40
    # Model dimensions
    nq_obj: int = 14 # object DOF 


def interp(src: torch.Tensor, n: int, order: int = 1) -> torch.Tensor:
    """Interpolate the source tensor using zeroth, first, or second-order hold.

    This function uses torch.nn.functional.interpolate for an efficient implementation
    of all interpolation methods.

    - order=0: Zeroth-order hold (Nearest Neighbor). Steps between values.
    - order=1: First-order hold (Linear Interpolation). Smooth lines between values.
    - order=2: Second-order hold (Quadratic Interpolation). Smooth curves between values.

    Args:
        src: Source tensor, shape (N, H, D).
        n: The integer upsampling factor.
        order: The order of interpolation. Must be 0, 1, or 2.

    Returns:
        Interpolated tensor, shape (N, H * n, D).
    """
    if order not in [0, 1, 2]:
        raise ValueError("Order must be an integer: 0, 1, or 2.")

    N, H, D = src.shape

    # If there's only one time step, interpolation is not meaningful.
    # The only possible behavior is to repeat the value (zero-order hold).
    if H <= 1:
        return src.repeat(1, n, 1)

    # Determine the interpolation mode string for the backend function.
    # Also handle cases where the input is too short for the chosen order.
    if order == 0:
        mode = "nearest"
    elif order == 1:
        mode = "linear"
    elif order == 2:
        # Quadratic interpolation requires at least 3 points to define a curve.
        if H < 3:
            # Gracefully fall back to linear if we can't do quadratic.
            print(
                f"Warning: Source tensor has H={H} < 3 time steps. "
                "Falling back to linear interpolation for order=2."
            )
            mode = "linear"
        else:
            mode = "quadratic"

    # Ensure the input tensor is a floating-point type for interpolation,
    # as linear and quadratic modes require it.
    if not src.is_floating_point():
        src = src.to(torch.float32)

    # `F.interpolate` expects the dimension to be interpolated as the last one.
    # The input shape should be (N, Channels, Length).
    # We treat our D dimension as "channels" and H as "length".
    # So, we permute the tensor from (N, H, D) to (N, D, H).
    src_permuted = src.permute(0, 2, 1)

    # Calculate the desired output length.
    # We subtract 1 from H, multiply by n, then add 1 to ensure that the
    # total number of points is correct after upsampling.
    # However, for simplicity and direct control, setting size=H*n works well.
    dst_len = H * n

    # align_corners=True is important for signal-like data. It ensures that the
    # endpoint values of the input and output sequences match perfectly.
    # It does not apply to 'nearest' mode.
    align = mode != "nearest"

    # Perform the 1D interpolation
    dst_permuted = F.interpolate(
        src_permuted, size=dst_len, mode=mode, align_corners=align
    )

    # Permute the dimensions back to the desired output shape: (N, D, H*n) -> (N, H*n, D)
    dst = dst_permuted.permute(0, 2, 1)

    return dst


# def load_data(
#     config: Config,
#     data_path: str = "/home/yizhao/yi/D2H/trajectory_kinematic.npz",
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectory data from NPZ file."""
    raw_data = np.load(data_path)
    qpos_ref = raw_data["qpos"]
    qvel_ref = raw_data["qvel"]
    try:
        contact = raw_data["contact"]
    except:
        contact = np.zeros((qpos_ref.shape[0], 10))
        loguru.logger.warning("contact data not found")
    try:
        contact_pos = raw_data["contact_pos"]
    except:
        contact_pos = np.zeros((qpos_ref.shape[0], 10, 3))
        loguru.logger.warning("contact_pos data not found")
    if "ctrl" in raw_data:
        ctrl_ref = raw_data["ctrl"]
    else:
        # TODO: disable automatic reference control generation, instead, move it to preprocess
        # Fallback if 'ctrl' is not in the data file
        loguru.logger.warning(
            "ctrl data not found, using 'qpos' as a initial guess for control."
        )
        if config.embodiment_type in ["bimanual", "right", "left"]:
            ctrl_ref = qpos_ref[:, : -config.nq_obj]
        elif config.embodiment_type in ["CMU", "DanceDB"]:
            ctrl_ref = qpos_ref[:, 7:]
        else:
            raise ValueError(f"Invalid embodiment_type: {config.embodiment_type}")
    # move to device
    qpos_ref_torch = torch.from_numpy(qpos_ref).to(config.device).to(torch.float32)
    qvel_ref_torch = torch.from_numpy(qvel_ref).to(config.device).to(torch.float32)
    ctrl_ref_torch = torch.from_numpy(ctrl_ref).to(config.device).to(torch.float32)
    contact_ref_torch = torch.from_numpy(contact).to(config.device).to(torch.float32)
    contact_pos_ref_torch = (
        torch.from_numpy(contact_pos).to(config.device).to(torch.float32)
    )
    # interpolate to match sim_dt
    if config.ref_dt > config.sim_dt:
        qpos_ref_interp = interp(qpos_ref_torch.unsqueeze(0), config.ref_steps).squeeze(
            0
        )
        qvel_ref_interp = interp(qvel_ref_torch.unsqueeze(0), config.ref_steps).squeeze(
            0
        )
        ctrl_ref_interp = interp(ctrl_ref_torch.unsqueeze(0), config.ref_steps).squeeze(
            0
        )
        contact_ref_interp = interp(
            contact_ref_torch.unsqueeze(0), config.ref_steps
        ).squeeze(0)
        H, Nc, D = contact_pos_ref_torch.shape
        contact_pos_ref_flat = contact_pos_ref_torch.view(H, Nc * D)
        contact_pos_ref_flat_interp = interp(
            contact_pos_ref_flat.unsqueeze(0), config.ref_steps
        ).squeeze(0)
        contact_pos_ref_interp = contact_pos_ref_flat_interp.view(-1, Nc, D)
    else:
        # downsample
        downsample_factor = int(config.sim_dt / config.ref_dt)
        qpos_ref_interp = qpos_ref_torch[::downsample_factor]
        qvel_ref_interp = qvel_ref_torch[::downsample_factor]
        ctrl_ref_interp = ctrl_ref_torch[::downsample_factor]
        contact_ref_interp = contact_ref_torch[::downsample_factor]
        contact_pos_ref_interp = contact_pos_ref_torch[::downsample_factor]
    # repeat the last frame with extra config.horizon_steps
    for _ in range(config.horizon_steps + config.ctrl_steps):
        qpos_ref_interp = torch.cat([qpos_ref_interp, qpos_ref_interp[-1:]], dim=0)
        qvel_ref_interp = torch.cat([qvel_ref_interp, qvel_ref_interp[-1:]], dim=0)
        ctrl_ref_interp = torch.cat([ctrl_ref_interp, ctrl_ref_interp[-1:]], dim=0)
        contact_ref_interp = torch.cat(
            [contact_ref_interp, contact_ref_interp[-1:]], dim=0
        )
        contact_pos_ref_interp = torch.cat(
            [contact_pos_ref_interp, contact_pos_ref_interp[-1:]], dim=0
        )

    return (
        qpos_ref_interp,
        qvel_ref_interp,
        ctrl_ref_interp,
        contact_ref_interp,
        contact_pos_ref_interp,
    )

def mujoco_to_urdf_qpos(mujoco_qpos):
    """Convert MuJoCo LEAP hand qpos to URDF joint order."""
    # Index mapping: urdf_qpos[i] = mujoco_qpos[mapping[i]]
    # mapping = [1, 0, 2, 3,    # Index finger: swap MCP(0) ↔ ROT(1)
    #            5, 4, 6, 7,    # Middle finger: swap MCP(4) ↔ ROT(5)
    #            9, 8, 10, 11,  # Ring finger: swap MCP(8) ↔ ROT(9)
    #            12, 13, 14, 15]  # Thumb: no change
    # mapping = [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    mapping =  [0, 12, 4, 8, 1, 13, 5, 9, 2, 14, 6, 10, 3, 15, 7, 11]
    return np.array(mujoco_qpos)[mapping]

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, grasp_data: dict):
    """Load scene, check robot state, and apply random actions."""
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Safely get object if it exists
    try:
        object = scene["object"]
    except KeyError:
        object = None
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    
    
    # Extract robot state from grasp data
    robot_pos = grasp_data["grasp_qpos"][:3]
    robot_quat = grasp_data["grasp_qpos"][3:7]
    robot_joint_pos = grasp_data["grasp_qpos"][7:]
    robot_joint_pos = mujoco_to_urdf_qpos(robot_joint_pos)
    
    # Convert robot state to tensors
    root_pos = torch.tensor(robot_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
    root_quat = torch.tensor(robot_quat, device=sim.device, dtype=torch.float32).unsqueeze(0)
    joint_pos = torch.tensor(robot_joint_pos, device=sim.device, dtype=torch.float32).unsqueeze(0)
    
    # Set robot root pose (position and orientation)
    robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1))
    
    # Set robot joint positions (with zero velocities)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    
    # Check initial robot state
    print("=== Initial Robot State ===")
    print(f"Number of environments: {scene.num_envs}")
    print(f"Robot body names: {robot.body_names}")
    print(f"Robot joint names: {robot.joint_names}")
    print(f"Robot joint positions shape: {robot.data.joint_pos.shape}")
    print(f"Robot root position: {robot.data.root_pos_w[0].cpu().numpy()}")
    print(f"Robot root orientation: {robot.data.root_quat_w[0].cpu().numpy()}")
    
    if object is not None:
        print(f"\n=== Object State ===")
        print(f"Object root position: {object.data.root_pos_w[0].cpu().numpy()}")
        print(f"Object root orientation: {object.data.root_quat_w[0].cpu().numpy()}")
    
    # Get action space (joint positions)
    num_joints = robot.num_joints
    print(f"\n=== Action Space ===")
    print(f"Number of joints: {num_joints}")
    print(f"Actuator groups: {list(robot.actuators.keys())}")
    
    # Simulation loop
    step_count = 0
    while simulation_app.is_running():
        # Keep robot at initial pose
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
            if object is not None:
                print(f"Object root position: {object.data.root_pos_w[0].cpu().numpy()}")

if __name__ == "__main__":
    config = Config()
    # qpos_ref_interp, qvel_ref_interp, ctrl_ref_interp, contact_ref_interp, contact_pos_ref_interp = load_data(config)
    # motion = {
    #     "qpos_ref": qpos_ref_interp,
    #     "qvel_ref": qvel_ref_interp,
    #     "ctrl_ref": ctrl_ref_interp,
    #     "contact_ref": contact_ref_interp,
    #     "contact_pos_ref": contact_pos_ref_interp,
    # }
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
    # path = "/home/yizhao/yi/DexGraspBench/output/example_shadow/graspdata/core_bottle_523cddb320608c09a37f3fc191551700/scale006_pose000/0.npy"
    # path = "/home/yizhao/yi/D2H/dev/eval_grasp/test_obj/ddg_gd_camera_poisson_006/floating/scale008_grasp.npy"
    # path = "/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_bottle_44dae93d7b7701e1eb986aac871fa4e5/floating/scale012/0_grasp.npy"
    # path = "/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_bottle_d655a217ad7d8974ce60bdf271ddc452/floating/scale012/0_grasp.npy"
    # path = "/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/mujoco_Star_Wars_Rogue_Squadron_Nintendo_64/floating/scale006/8_grasp.npy"
    path = "/home/yizhao/yi/DexGraspBench/output/debug_leap/succgrasp/core_mug_3d3e993f7baa4d7ef1ff24a8b1564a36/floating/scale010/0_grasp.npy"
    grasp_data = np.load(path, allow_pickle=True).item()

    # import ipdb; ipdb.set_trace()
    sim_cfg = sim_utils.SimulationCfg(device=config.device)
    sim_cfg.dt = 0.02
    sim_cfg.gravity = (0.0, 0.0, 0.0)  # Disable gravity
    sim = SimulationContext(sim_cfg)

    # Create scene config with object state from grasp data
    SceneCfg = create_scene_cfg(grasp_data)
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene, grasp_data)
