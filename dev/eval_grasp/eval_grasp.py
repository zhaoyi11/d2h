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


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

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
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/yizhao/yi/D2H/dev/eval_grasp/usd_20260122_144238_1527/coacd.usd",
            scale=(0.1, 0.1, 0.1),
        ),
    )


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


def load_data(
    config: Config,
    data_path: str = "/home/yizhao/yi/D2H/trajectory_kinematic.npz",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
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
    
    # Initialize random actions (normalized to [-1, 1])
    # Actions are joint position targets, matching the number of joints
    # These will be scaled by the action scale in the robot's actuator configuration
    random_actions = 2.0 * torch.rand(scene.num_envs, num_joints, device=sim.device) - 1.0
    
    # Simulation loop
    step_count = 0
    while simulation_app.is_running():
        # Generate new random actions periodically (every 50 steps)
        if step_count % 50 == 0:
            random_actions = 2.0 * torch.rand(scene.num_envs, num_joints, device=sim.device) - 1.0
        
        # Apply random actions
        robot.set_joint_position_target(random_actions)
        
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

    sim_cfg = sim_utils.SimulationCfg(device=config.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    run_simulator(sim, scene)
