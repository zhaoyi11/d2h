# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg

import src.tasks.pick_insert.mdps as mdp
import src.tasks.reorient.mdps as task_mdps
from src.tasks.pick_insert.mdps.contact_filters import (
    contact_filter_prim_paths,
    external_indices,
    object_indices,
)

from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG

UWLAB_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main"

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""

    # robot
    robot = FRANKA_LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # insertive_object: 
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
                kinematic_enabled=False,
                # Gently correct any residual overlap instead of flinging the ~0.02 kg peg out of the
                # workspace (which reads as an env reset).
                max_depenetration_velocity=0.1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.2, 0.30), rot=(0.7071068, 0.0, 0.7071068, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(2.0, 2.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
                kinematic_enabled=True,
                max_depenetration_velocity=1.0,
            ),
            # Speculative-contact margin on the hole surfaces (its convexHull collision is rewritten
            # to SDF by the receptacle_collision_sdf prestartup event so the cavity is collidable).
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, 
                contact_offset=0.01, rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.275), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # trick: we let visualizer's color to show the table with success coloring
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # static obstacle: a fixed pillar the MPC must route around. Visual-only (no collision_props):
    # it exists physically only in the cuRobo planning world, keeping the demo clean if the planner
    # ever clips it. Mirrored as a static "static_obstacle" cuboid in the arm action.
    static_obstacle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/StaticObstacle",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.25),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.85)),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, -0.25, 0.38), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # ###### DYNAMIC OBSTACLE ######
    # # dynamic obstacle: a box driven on a scripted Y-sinusoid each step (EventCfg.move_dynamic_obstacle).
    # # The arm action mirrors its live pose into the cuRobo world (dynamic_obstacle_assets). Visual-only.
    # dynamic_obstacle: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/DynamicObstacle",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.06, 0.06, 0.06),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.2)),
    #         visible=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.4, 0.05, 0.45), rot=(1.0, 0.0, 0.0, 0.0)
    #     ),
    # )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.PickInsertTrajectoryObjectAndHandBasePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        success_vis_asset_name="table",
        # Reach-to-grasp is the trajectory's first stage (pregrasp): the object goal is held at the
        # peg's spawn pose so the arm reaches down to grasp before advancing into the
        # pick->insert trajectory (see build_pick_insert_object_pose_sequence).
        # Actively nudge the hand-base anchor so the object reaches its goal pose when the
        # in-hand policy alone cannot; cost-regularized (rotation costs more than position),
        # complementing the MPC-driven arm.
        correction=mdp.AnchorCorrectionCfg(
            enable=True,
            # Bounded but memoryless: clamped pure-P correction toward the CURRENT command anchor.
            slew_pos=1.0,  # large -> slew never limits, correction is fresh each step
            slew_rot=10.0,
            max_pos=0.05,  # keep the 5 cm bound (default, made explicit)
            max_rot=0.2,  # keep the ~10 deg bound (default, made explicit)
            # Gate: only activate correction when arm is settled AND object has stalled.
            anchor_achieved_pos=0.01,  # hand-base position tolerance (m)
            anchor_achieved_rot=0.05,  # hand-base orientation tolerance (rad)
            stall_window=5,  # steps for object to stall before arm corrects
            stall_delta_pos=0.003,  # position improvement threshold (m)
            stall_delta_rot=0.01,  # orientation improvement threshold (rad)
        ),
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        object_pos_b = ObsTerm(
            func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0)
        )

        object_quat_b = ObsTerm(
            func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0)
        )
        target_object_pose_b = ObsTerm(
            func=mdp.command_object_pose_b, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):

        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),  # clamp between -2 m to 2 m
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    @configclass
    class LowLevelObsCfg(ObsGroup):
        """Reorient-style current-step state used by the frozen low-level VAE."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            noise=Gnoise(std=0.005),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.2,
            noise=Gnoise(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        fingertip_pose = ObsTerm(
            func=mdp.body_state_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=".*fingertip.*"),
                "base_body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
        # -- object contact: raw 3D force in hand-base frame (object filter only) 
        # TODO: remove this in the latest policy
        fingertip_contact_force_b = ObsTerm(
            func=mdp.fingers_contact_force_body_b,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "base_body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "filter_indices": object_indices(),
            },
        )
        contact_mask = ObsTerm(
            func=task_mdps.tip_contact_mask_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "filter_indices": object_indices(),
            },
        )
        contact_force_mag = ObsTerm(
            func=task_mdps.tip_contact_force_mag_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "filter_indices": object_indices(),
            },
        )
        contact_pose = ObsTerm(
            func=task_mdps.tip_contact_pose_flat,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "contact_pose_range_deg": 45.0,
                "filter_indices": object_indices(),
            },
        )
        # -- external contact: receptacle + table, vector-summed per fingertip (the
        #    "external force sensing" channel). mask (4) + magnitude (4) + pose (8) = 16 dims.
        #    These 16 dims turn the 151-dim object-only group into the 167-dim layout the
        #    dex_reorient external-force checkpoint was trained on.
        external_contact_mask = ObsTerm(
            func=task_mdps.tip_contact_mask_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "filter_indices": external_indices(),
            },
        )
        external_contact_force_mag = ObsTerm(
            func=task_mdps.tip_contact_force_mag_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "filter_indices": external_indices(),
            },
        )
        external_contact_pose = ObsTerm(
            func=task_mdps.tip_contact_pose_flat,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "contact_pose_range_deg": 45.0,
                "filter_indices": external_indices(),
            },
        )
        object_pos = ObsTerm(
            func=mdp.object_pos_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_quat = ObsTerm(
            func=mdp.object_quat_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_lin_vel = ObsTerm(
            func=mdp.object_lin_vel_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_ang_vel = ObsTerm(
            func=mdp.object_ang_vel_body_b,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        gravity_dir = ObsTerm(
            func=mdp.gravity_dir_body_b,
            params={"body_asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        goal_pos_diff = ObsTerm(
            func=mdp.goal_pos_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "make_quat_unique": False,
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "hand_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    # # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()
    low_level: LowLevelObsCfg | None = None


@configclass
class EventCfg:
    """Configuration for randomization."""
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.5, 2.0],
            "damping_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    # Rest the peg on the table each reset (its init_state is an on-table lying pose). Small x/y
    # jitter keeps it on the table; velocity zeroed so it settles gently under the reduced-gravity
    # curriculum. This replaces the in-hand teleport (removed below) so there is no hand/object
    # contact at t=0. Orientation is fixed (no yaw jitter): the reach-to-grasp command derives a
    # grasp pose with a fixed anchor orientation, so the peg must spawn at a consistent orientation
    # for the grasp to align (x/y variation is fine -- the reach reads the peg's live pose).
    reset_object: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03], "yaw": [0.0, 0.0]},
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": [-0.50, 0.50],
    #         "velocity_range": [0.0, 0.0],
    #     },
    # )
    # Note (Octi): This is a deliberate trick in Remake to accelerate learning.
    # By scheduling gravity as a curriculum — starting with no gravity (easy)
    # and gradually introducing full gravity (hard) — the agent learns more smoothly.
    # This removes the need for a special "Lift" reward (often required to push the
    # agent to counter gravity), which has bonus effect of simplifying reward composition overall.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.81], [0.0, 0.0, -1.81]), # TODO: change to [0.0, 0.0, -9.81] for full gravity
            "operation": "abs",
        },
    )

    # ##### DYNAMIC OBSTACLE #####
    # # Drive the dynamic_obstacle prop along a scripted Y-sinusoid every step (kinematic). The arm
    # # action mirrors its live pose into the cuRobo collision world (dynamic_obstacle_assets), so the
    # # MPC re-plans around the moving box. Global-time phased so all envs stay in sync.
    # move_dynamic_obstacle = EventTerm(
    #     func=mdp.move_dynamic_obstacle,
    #     mode="interval",
    #     interval_range_s=(0.0, 0.0),
    #     params={
    #         "asset_cfg": SceneEntityCfg("dynamic_obstacle"),
    #         "center": (0.4, 0.05, 0.45),
    #         "axis": (0.0, 1.0, 0.0),
    #         "amplitude": 0.2,
    #         "freq": 0.25,
    #     },
    # )


@configclass
class ActionsCfg:

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.1
    )


@configclass
class HrlActionsCfg:
    """Low-level action interface consumed by the HRL chunk wrapper."""

    # cuRobo reactive MPC: drives the hand `base` to the command anchor while the full Franka+LEAP
    # collision model avoids the table. The receptacle is intentionally NOT an obstacle so the peg
    # can still reach the bore (MPC is position-controlled, no compliance — insertion is stiff).
    arm_action = mdp.CommandHandBaseCuroboMpcActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="base",
        command_name="object_pose",
        robot_config_file=(
            f"{Path(__file__).resolve().parents[2]}/assets/franka_leap_hand/curobo/franka_leap.yml"
        ),
        # Obstacle cuboids in the robot base frame (poses match the scene props' world poses; the
        # robot base sits at the world origin in this env). The receptacle is intentionally excluded
        # so the peg can still reach the bore.
        obstacle_cuboids={
            # "table": {"dims": [0.8, 1.5, 0.04], "pose": [0.55, 0.0, 0.235, 1, 0, 0, 0]},
            "static_obstacle": {"dims": [0.05, 0.05, 0.25], "pose": [0.5, -0.25, 0.38, 1, 0, 0, 0]},
            #### DYNAMIC OBSTACLE ####
            # "dynamic_obstacle": {"dims": [0.06, 0.06, 0.06], "pose": [0.4, 0.05, 0.45, 1, 0, 0, 0]},
        },

        # ###### DYNAMIC OBSTACLE TRACKING ######
        # # Track the moving scene prop: its live pose is mirrored into the cuRobo world each step.
        # dynamic_obstacle_assets={"dynamic_obstacle": "dynamic_obstacle"},
        # # Avoidance tuning: react only when ~8cm from an obstacle and let the goal compete more with
        # # avoidance (weight 5000 vs the 10000 default) so the arm deviates less when near obstacles.
        # collision_activation_distance=0.08,
        # collision_weight=8000.0,
        # use_cuda_graph=True,
    )
    hand_action = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.002)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.25}, weight=1.0
    )

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=1.0,
        params={"threshold": 1.0},
    )

    object_lifted = RewTerm(
        func=mdp.object_lifted_above_table,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "table_cfg": SceneEntityCfg("table"),
            "height": 0.06,
            "table_half_height": 0.02,
        },
    )

    pre_insert_position = RewTerm(
        func=mdp.object_to_hole_xy_tanh,
        weight=4.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "hole_cfg": SceneEntityCfg("receptive_object"),
            "table_cfg": SceneEntityCfg("table"),
            "std": 0.08,
            "contact_threshold": 1.0,
            "lift_height": 0.06,
            "lift_gate": 0.5,
        },
    )

    insertion_orientation = RewTerm(
        func=mdp.peg_hole_axis_alignment,
        weight=3.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "hole_cfg": SceneEntityCfg("receptive_object"),
            "std": 0.35,
        },
    )

    insertion_depth = RewTerm(
        func=mdp.peg_insertion_depth,
        weight=8.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "hole_cfg": SceneEntityCfg("receptive_object"),
            "table_cfg": SceneEntityCfg("table"),
            "target_depth": 0.015,
            "approach_height": 0.08,
            "xy_tolerance": 0.04,
            "axis_tolerance": 0.25,
            "contact_threshold": 1.0,
            "lift_height": 0.06,
            "lift_gate": 0.5,
        },
    )

    success = RewTerm(
        func=mdp.peg_inserted_success,
        weight=20.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "hole_cfg": SceneEntityCfg("receptive_object"),
            "pos_tol": 0.015,
            "axis_tol": 0.15,
            "depth": 0.015,
        },
    )

    # early_termination = RewTerm(
    #     func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"}
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-0.5, 1.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    # abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class DexsuiteInsertPegEnvCfg(ManagerBasedRLEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
    )
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(gravity=(0.0, 0.0, -9.81))
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: mdp.CurriculumCfg | None = mdp.CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # 50 Hz

        # *single-goal setup
        self.commands.object_pose.resampling_time_range = (10.0, 10.0)
        self.commands.object_pose.position_only = False
        self.commands.object_pose.success_visualizer_cfg.markers["failure"] = (
            self.scene.table.spawn.replace(
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.15, 0.15), roughness=0.25
                ),
                visible=True,
            )
        )
        self.commands.object_pose.success_visualizer_cfg.markers["success"] = (
            self.scene.table.spawn.replace(
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.25, 0.15), roughness=0.25
                ),
                visible=True,
            )
        )

        self.episode_length_s = 10.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**30

        # # Render settings
        # self.sim.render.enable_dlssg = True
        # self.sim.render.enable_ambient_occlusion = True
        # self.sim.render.enable_reflections = True
        # self.sim.render.enable_dl_denoiser = True
        
        
        ''' Note: the following settings are used previously. Compare whether this is the cause of penetration. '''
        # self.sim.render_interval = self.decimation
        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        # self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
        # self.sim.physx.gpu_collision_stack_size = 2**28

        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = self.rewards.success.params["pos_tol"]
            self.curriculum.adr.params["rot_tol"] = None


class DexsuiteInsertEnvCfg(DexsuiteInsertPegEnvCfg):
    """Dexsuite lift task definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.position_only = False


class DexsuiteInsertEnvCfg_PLAY(DexsuiteInsertPegEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = False
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]


# class DexsuiteInsertEnvCfg_PLAY(DexsuiteInsertEnvCfg):
#     """Dexsuite lift task evaluation environment definition"""

#     def __post_init__(self):
#         super().__post_init__()
#         self.commands.object_pose.resampling_time_range = (2.0, 3.0)
#         self.commands.object_pose.debug_vis = True
#         self.commands.object_pose.position_only = True
#         self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
#             "max_difficulty"
#         ]


##########
#  Tasks
##########


@configclass
class FrankaLeapMixinCfg:

    def __post_init__(self: DexsuiteInsertPegEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "base"  # TODO: check this !!
        finger_tip_body_list = [
            "thumb_fingertip",
            "fingertip",
            "fingertip_2",
            "fingertip_3",
        ]
        self.scene.fingertip_transforms = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/thumb_fingertip",
                    offset=OffsetCfg(pos=(0.0, -0.045, -0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/fingertip",
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/fingertip_2",
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/fingertip_3",
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
            ],
            debug_vis=False,
        )
        # Filter against the full target registry [Object, ReceptiveObject, Table] so
        # force_matrix_w column k == CONTACT_FILTER_TARGETS[k]; the low-level obs then splits
        # object (index 0) vs external (receptacle + table) contact via filter_indices.
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/" + link_name,
                    filter_prim_paths_expr=contact_filter_prim_paths(),
                    debug_vis=False,
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={
                "contact_sensor_names": [
                    f"{link}_object_s" for link in finger_tip_body_list
                ],
                "filter_indices": object_indices(),  # keep proprio object-only with 3 filters
            },
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )


# @configclass
# class DexsuiteFrankaLeapInsertEnvCfg(FrankaLeapMixinCfg, DexsuiteInsertEnvCfg):
#     pass


# @configclass
# class DexsuiteFrankaLeapInsertEnvCfg_PLAY(
#     FrankaLeapMixinCfg, DexsuiteInsertEnvCfg_PLAY
# ):
#     pass


@configclass
class DexsuiteFrankaLeapInsertEnvCfg(FrankaLeapMixinCfg, DexsuiteInsertEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapInsertEnvCfg_PLAY(FrankaLeapMixinCfg, DexsuiteInsertEnvCfg_PLAY):
    pass


@configclass
class DexsuiteFrankaLeapInsertHrlEnvCfg(FrankaLeapMixinCfg, DexsuiteInsertEnvCfg):
    """Pick-insert HRL env driven by the EXTERNAL-FORCE-SENSING low-level hand policy.

    cuRobo-MPC arm + frozen low-level LEAP-hand policy. The peg spawns on the table and the
    reach-to-grasp command drives the pick before the pick->insert trajectory. The ``low_level``
    observation group is the 167-dim reorient layout (object + external fingertip-contact
    sensing), matching the dex_reorient ``reorient`` checkpoint. The 7-DOF arm is purely
    MPC-driven (0 external action dims).
    """

    actions: HrlActionsCfg = HrlActionsCfg()

    def __post_init__(self):
        self.observations.low_level = ObservationsCfg.LowLevelObsCfg()
        super().__post_init__()
        # cuRobo MPC owns the arm: zero the arm position gains (the MPC action restores them on
        # reset); the LEAP hand ("fingers") stays position-controlled.
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
