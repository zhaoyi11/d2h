"""
LEAP hand configs file for IsaacLab.

Modified template from https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/allegro.py
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pathlib import Path

FRANKA_LEAP_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{Path(__file__).parent}/franka_leap_hand_v1_right/franka_leap_hand_right.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            # angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0, 
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, 
            solver_position_iteration_count=16, 
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={"panda_joint1": 0.0, 
                    "panda_joint2": -0.569, 
                    "panda_joint3": 0.0, 
                    "panda_joint4": -2.810, 
                    "panda_joint5": 0.0, 
                    "panda_joint6": 3.037, 
                    "panda_joint7": 0.741,  
                    "a_.*": 0.0},
    ),
    actuators={
        "joints": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-7]"],
            stiffness=400.0,
            damping=80.0,
            friction=0.01,
        ),
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["a_.*"],
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
            effort_limit=0.5,
            velocity_limit=100.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)