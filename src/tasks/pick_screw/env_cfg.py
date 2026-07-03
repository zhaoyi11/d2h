# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg

import src.tasks.pick_anyrotate.mdps as mdp
import src.tasks.pick_insert.mdps as insert_mdp
import src.tasks.reorient.mdps as task_mdps
from src.tasks.pick_screw.mdps.contacts import (
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

    # object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd",
            scale=(1.5, 1.5, 1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.1, 0.34), 
                                                rot=(1.0, 0.0, 0.0, 0.0)),
    )
    
    receptive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd",
            scale=(2.0, 2.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.271), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # # table base
    # object_table = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/ObjectTable",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/home/yizhao/yi/D2H/src/assets/square_table_leg/square_table_top_convex.usd",
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=True,
    #             disable_gravity=True,
    #             max_depenetration_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=1000.0,
    #         ),
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #             articulation_enabled=False,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    #         scale=(1.5, 1.5, 1.5),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.55, 0.0, 0.271],
    #         rot=[0.7071068, 0.7071068, 0.0, 0.0],
    #     ),
    # )

    # # table leg
    # object = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"/home/yizhao/yi/D2H/src/assets/square_table_leg/square_table_leg1_convex.usd",
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=False,
    #             max_depenetration_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_angular_velocity=1000.0,
    #         ),
    #         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #             articulation_enabled=False,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    #         scale=(1.5, 1.5, 1.5),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=[0.55, 0.1, 0.34],
    #         rot=[1.0, 0.0, 0.0, 0.0],
    #     ),
    # )    
    
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

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7),
            pos_y=(-0.25, 0.25),
            pos_z=(0.55, 0.95),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # object_pos_b = ObsTerm(
        #     func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0)
        # )

        object_quat_b = ObsTerm(
            func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0)
        )
        target_object_pose_b = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
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
        """Reorient-style current-step state consumed by the frozen low-level hand policy.

        Reproduces the dex_reorient ``reorient`` policy observation EXACTLY (167 dims, in order),
        so its checkpoint loads against this env. Everything is expressed relative to the LEAP
        hand ``base`` body (the floating-hand policy's root) via the ``*_body_b`` helpers. The
        three external-contact terms (mask/mag/pose, ``external_indices()``) are the
        "external force sensing" channel — fingertips sense the receptacle and table, vector
        summed, on top of the object-contact channel.
        """

        # -- hand joints (16 LEAP DOF)
        joint_pos = ObsTerm(
            func=insert_mdp.joint_pos_limit_normalized,
            noise=Gnoise(std=0.005),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        joint_vel = ObsTerm(
            func=insert_mdp.joint_vel_rel,
            scale=0.2,
            noise=Gnoise(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        # -- fingertip full state (pos/quat/lin_vel/ang_vel) in hand-base frame (4 x 13 = 52)
        fingertip_pose = ObsTerm(
            func=insert_mdp.body_state_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=".*fingertip.*"),
                "base_body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
        # -- object contact: raw 3D force in hand-base frame (object filter only)
        fingertip_contact_force_b = ObsTerm(
            func=insert_mdp.fingers_contact_force_body_b,
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
        # -- external contact: receptacle + table, vector-summed per fingertip (the new
        #    "external force sensing" channel). mask (4) + magnitude (4) + pose (8) = 16 dims.
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
        # -- object state in hand-base frame
        object_pos = ObsTerm(
            func=insert_mdp.object_pos_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_quat = ObsTerm(
            func=insert_mdp.object_quat_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_lin_vel = ObsTerm(
            func=insert_mdp.object_lin_vel_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_ang_vel = ObsTerm(
            func=insert_mdp.object_ang_vel_body_b,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        gravity_dir = ObsTerm(
            func=insert_mdp.gravity_dir_body_b,
            params={"body_asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        goal_pos_diff = ObsTerm(
            func=insert_mdp.goal_pos_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        goal_quat_diff = ObsTerm(
            func=insert_mdp.goal_quat_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "make_quat_unique": False,
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        last_action = ObsTerm(func=insert_mdp.last_action, params={"action_name": "hand_action"})

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

    # # -- pre-startup
    # randomize_object_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},
    # )
     
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

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.2, 0.2],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.4],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    # Note (Octi): This is a deliberate trick in Remake to accelerate learning.
    # By scheduling gravity as a curriculum — starting with no gravity (easy)
    # and gradually introducing full gravity (hard) — the agent learns more smoothly.
    # This removes the need for a special "Lift" reward (often required to push the
    # agent to counter gravity), which has bonus effect of simplifying reward composition overall.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )


@configclass
class ActionsCfg:

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.1
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    # success_position = RewTerm(
    #     func=mdp.success_reward,
    #     weight=5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "pos_std": 0.1,
    #         "rot_std": None,
    #         "command_name": "object_pose",
    #         "align_asset_cfg": SceneEntityCfg("object"),
    #     },
    # )

    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    early_termination = RewTerm(
        func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"}
    )


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

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class DexsuiteScrewEnvCfg(ManagerBasedRLEnvCfg):
    """Dexsuite screw task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
    )
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
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

        self.episode_length_s = 4.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2**28

        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = (
                self.rewards.success.params["pos_std"] / 2
            )

            self.curriculum.adr.params["rot_tol"] = (
                self.rewards.success.params["rot_std"] / 2
            )


class DexsuiteScrewLiftEnvCfg(DexsuiteScrewEnvCfg):
    """Dexsuite screw lift task definition"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None  # no orientation reward
        self.commands.object_pose.position_only = True
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = (
                None  # make success reward not consider orientation
            )
            self.curriculum.adr.params["rot_tol"] = (
                None  # make adr not tracking orientation
            )


class DexsuiteScrewEnvCfg_PLAY(DexsuiteScrewEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]


class DexsuiteScrewLiftEnvCfg_PLAY(DexsuiteScrewLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.position_only = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]


##########
#  Tasks
##########


@configclass
class FrankaLeapMixinCfg:

    def __post_init__(self: DexsuiteScrewEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "base"  # TODO: check this !!
        finger_tip_body_list = [
            "thumb_fingertip",
            "fingertip",
            "fingertip_2",
            "fingertip_3",
        ]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/square_table_leg1"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={
                "contact_sensor_names": [
                    f"{link}_object_s" for link in finger_tip_body_list
                ]
            },
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )

        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )


@configclass
class DexsuiteFrankaLeapScrewEnvCfg(FrankaLeapMixinCfg, DexsuiteScrewEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapScrewEnvCfg_PLAY(
    FrankaLeapMixinCfg, DexsuiteScrewEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteFrankaLeapScrewLiftEnvCfg(FrankaLeapMixinCfg, DexsuiteScrewLiftEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapScrewLiftEnvCfg_PLAY(FrankaLeapMixinCfg, DexsuiteScrewLiftEnvCfg_PLAY):
    pass


##########
#  HRL + cuRobo-MPC variant
##########


@configclass
class HrlCommandsCfg:
    """Command that emits BOTH the object goal pose and the hand-base anchor pose.

    ``command[:, 7:14]`` is the hand-base anchor the cuRobo MPC arm action tracks; the in-hand
    low-level policy reads the object goal from ``command[:, :7]``.
    """

    object_pose = insert_mdp.PickInsertTrajectoryObjectAndHandBasePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        ranges=insert_mdp.PickInsertTrajectoryObjectAndHandBasePoseCommandCfg.Ranges(
            pos_x=(0.35, 0.35),
            pos_y=(0.0, 0.0),
            pos_z=(0.30, 0.30),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
        # receptacle (table top) the screw trajectory targets — matches the scene's ReceptiveObject.
        receptive_pose=(0.55, 0.0, 0.271, 1.0, 0.0, 0.0, 0.0),
        # Nudge the hand-base anchor so the object reaches its goal when the in-hand policy alone
        # cannot; bounded, memoryless, gated on a settled arm + stalled object (as in pick_insert).
        correction=insert_mdp.AnchorCorrectionCfg(
            enable=True,
            slew_pos=1.0,
            slew_rot=10.0,
            max_pos=0.05,
            max_rot=0.2,
            anchor_achieved_pos=0.01,
            anchor_achieved_rot=0.05,
            stall_window=5,
            stall_delta_pos=0.003,
            stall_delta_rot=0.01,
        ),
    )


@configclass
class HrlActionsCfg:
    """Low-level action interface: cuRobo-MPC arm (0 dims) + 16-DOF LEAP hand."""

    # cuRobo reactive MPC drives the hand `base` to the command anchor while the full Franka+LEAP
    # collision model avoids the table. Consumes ZERO external action dims (goal comes from the
    # command), so the env action is the 16-DOF hand action only.
    arm_action = insert_mdp.CommandHandBaseCuroboMpcActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="base",
        command_name="object_pose",
        robot_config_file=(
            f"{Path(__file__).resolve().parents[2]}/assets/franka_leap_hand/curobo/franka_leap.yml"
        ),
        # Obstacle cuboids in the robot base frame (the robot base sits at the world origin here).
        # The receptacle is intentionally excluded so the leg can still reach the bore.
        obstacle_cuboids={
            "table": {"dims": [0.8, 1.5, 0.04], "pose": [0.55, 0.0, 0.235, 1, 0, 0, 0]},
        },
    )
    hand_action = insert_mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class DexsuiteFrankaLeapScrewHrlEnvCfg(FrankaLeapMixinCfg, DexsuiteScrewEnvCfg):
    """Pick-screw HRL variant: cuRobo-MPC arm + frozen low-level LEAP-hand policy.

    The 7-DOF arm tracks the command hand-base anchor via cuRobo reactive MPC (0 external action
    dims); the 16-DOF LEAP hand is driven by a frozen reorient policy reading the ``low_level``
    observation group (167 dims, incl. external-contact force sensing). No high-level RL policy —
    a smoke script feeds the hand policy's 16 actions while the MPC drives the arm.
    """

    commands: HrlCommandsCfg = HrlCommandsCfg()
    actions: HrlActionsCfg = HrlActionsCfg()

    def __post_init__(self):
        # Enable the reorient-style 167-dim low-level observation group.
        self.observations.low_level = ObservationsCfg.LowLevelObsCfg()
        super().__post_init__()  # FrankaLeapMixinCfg -> DexsuiteScrewEnvCfg

        # FrameTransformer over the fingertips: tip_contact_* read its target_quat_w to rotate
        # contact forces into each fingertip frame.
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

        # Rebuild the fingertip contact sensors to filter against the full target registry
        # [Object, ReceptiveObject, Table] so force_matrix_w column k == CONTACT_FILTER_TARGETS[k].
        # (Replaces the object-only / stale single-target filter from FrankaLeapMixinCfg.)
        for link_name in ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/" + link_name,
                    filter_prim_paths_expr=contact_filter_prim_paths(),
                    debug_vis=False,
                ),
            )
        # Keep the proprio object-contact channel object-only now that sensors carry 3 filters.
        self.observations.proprio.contact.params["filter_indices"] = object_indices()

        # cuRobo MPC owns the arm: zero the arm's position gains so set_joint_position_target has
        # no PD authority until the MPC action restores them on reset (mirrors pick_insert demo).
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
