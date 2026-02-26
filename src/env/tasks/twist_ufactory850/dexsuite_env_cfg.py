# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.env.tasks.twist.mdps as mdp
from src.env.tasks.twist.adr_curriculum import CurriculumCfg

JOINT_LOWER_LIMIT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    # jif1, jmf1, jpf1, jth1
                    -0.05, -0.05, -0.570, 0.364,
                    # jif2, jmf2, jpf2, jth2
                    -0.296, -0.296, -0.296, -0.205,
                    # jif3, jmf3, jpf3, jth3
                    -0.274, -0.274, -0.274, -0.290,
                    # jif4, jmf4, jpf4, jth4
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    # jif1, jmf1, jpf1, jth1
                    0.570, 0.05, 0.05, 1.497,
                    # jif2, jmf2, jpf2, jth2
                    1.710, 1.710, 1.710, 1.130, 
                    # jif3, jmf3, jpf3, jth3
                    1.809, 1.809, 1.809, 1.633, 
                    # jif4, jmf4, jpf4, jth4
                    1.718, 1.718, 1.718, 1.820]
JOINT_LOWER_LIMIT_LEFT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    # jif1, jmf1, jpf1, jth1
                    -0.570, -0.05, -0.05, 0.364,
                    # jif2, jmf2, jpf2, jth2
                    -0.296, -0.296, -0.296, -0.205,
                    # jif3, jmf3, jpf3, jth3
                    -0.274, -0.274, -0.274, -0.290,
                    # jif4, jmf4, jpf4, jth4
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT_LEFT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    # jif1, jmf1, jpf1, jth1
                    0.05, 0.05, 0.570, 1.497,
                    # jif2, jmf2, jpf2, jth2
                    1.710, 1.710, 1.710, 1.130, 
                    # jif3, jmf3, jpf3, jth3
                    1.809, 1.809, 1.809, 1.633, 
                    # jif4, jmf4, jpf4, jth4
                    1.718, 1.718, 1.718, 1.820]

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""

    # robot
        # robots
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"/home/yizhao/yi/D2H/src/assets/ufactory850/uf850_allegro_right_colored.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": -0.05,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": 1.4,
                "joint5": -1.0,
                "joint6": -3.14,
                # hand 
                "jif1": 0.0,
                "jif2": 0.4,
                "jif3": 0.4,
                "jif4": 0.0,
                "jmf1": 0.0,
                "jmf2": 0.4,
                "jmf3": 0.4,
                "jmf4": 0.0,
                "jpf1": 0.0,
                "jpf2": 0.4,
                "jpf3": 0.4,
                "jpf4": 0.0,
                "jth1": 0.364,
                "jth2": 0.0,
                "jth3": 0.2,
                "jth4": 0.0,
            },
            # pos=(-0.274, -0.475, 0.01),
            pos=(0.0, 0.0, 0.0),
        ),
        actuators={
            "xArm_1-6": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=2000.0,
                damping=16.0,
            ),
            "allegro_hand_1": ImplicitActuatorCfg(
                joint_names_expr=["j.*f1"],
                stiffness=325.0,
                damping=20.0,
            ),
            "allegro_hand_2": ImplicitActuatorCfg(
                joint_names_expr=["j.*f2"],
                stiffness=425.0,
                damping=25.0,
            ),
            "allegro_hand_3": ImplicitActuatorCfg(
                joint_names_expr=["j.*f3"],
                stiffness=245.0,
                damping=15.0,
            ),
            "allegro_hand_4": ImplicitActuatorCfg(
                joint_names_expr=["j.*f4"],
                stiffness=1050.0,
                damping=65.0,
            ),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(
                joint_names_expr=["jth1"],
                stiffness=100.0,
                damping=5.0,
            ),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(
                joint_names_expr=["jth2"],
                stiffness=300.0,
                damping=15.0,
            ),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(
                joint_names_expr=["jth3"],
                stiffness=1270.0,
                damping=100.0,
            ),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(
                joint_names_expr=["jth4"],
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )


    # table base
    object_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ObjectTable",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/yizhao/yi/D2H/src/assets/furniture_bench/urdf/square_table/square_table_top.urdf",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
            ),
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                ),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            # Keep this asset visual-only to avoid overlapping support collisions with the task table cuboid.
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            scale=(1.5, 1.5, 1.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.55, 0.0, 0.271],
            rot=[0.7071068, 0.7071068, 0.0, 0.0],
        ),
    )

    # table leg
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/yizhao/yi/D2H/src/assets/furniture_bench/urdf/square_table/square_table_leg2.urdf",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                ),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            scale=(1.5, 1.5, 1.5),
            # physics_material=RigidBodyMaterialCfg(static_friction=0.5), # TODO: check how the friction defined in the urdf file
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.55, 0.1, 0.34],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )

    # # object
    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[
    #             CuboidCfg(
    #                 size=(0.05, 0.1, 0.1),
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CuboidCfg(
    #                 size=(0.05, 0.05, 0.1),
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CuboidCfg(
    #                 size=(0.025, 0.1, 0.1),
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CuboidCfg(
    #                 size=(0.025, 0.05, 0.1),
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CuboidCfg(
    #                 size=(0.025, 0.025, 0.1),
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CuboidCfg(
    #                 size=(0.01, 0.1, 0.1),
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             SphereCfg(
    #                 radius=0.05,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             SphereCfg(
    #                 radius=0.025,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CapsuleCfg(
    #                 radius=0.04,
    #                 height=0.025,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CapsuleCfg(
    #                 radius=0.04,
    #                 height=0.01,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CapsuleCfg(
    #                 radius=0.04,
    #                 height=0.1,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CapsuleCfg(
    #                 radius=0.025,
    #                 height=0.1,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CapsuleCfg(
    #                 radius=0.025,
    #                 height=0.2,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             CapsuleCfg(
    #                 radius=0.01,
    #                 height=0.2,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             ConeCfg(
    #                 radius=0.05,
    #                 height=0.1,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #             ConeCfg(
    #                 radius=0.025,
    #                 height=0.1,
    #                 physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    #             ),
    #         ],
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=0,
    #             disable_gravity=False,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    # )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # trick: we let visualizer's color to show the table with success coloring
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)
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


    # # lights
    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=750.0,
    #         texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    #     ),
    # )


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            # Fixed target at 3 cm above rear-left hole of object_table.
            pos_x=(-0.634375, -0.634375),
            pos_y=(0.084375, 0.084375),
            # pos_z=(0.3243475, 0.3243475),
            pos_z=(0.4, 0.4), # when 3 cm above, the agent learn to hack the task by pushing the object, change it higher to avoid this
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
        # contact: ObsTerm = MISSING

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

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


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

    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="joint6"),
            "position_range": [-3, 3],
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
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    # fingers_to_object = RewTerm(
    #     func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0
    # )

    # position_tracking = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=2.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "std": 0.2,
    #         "command_name": "object_pose",
    #         "align_asset_cfg": SceneEntityCfg("object"),
    #     },
    # )

    # orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error_tanh,
    #     weight=4.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "std": 1.5,
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
            "in_bound_range": {"x": (-1.5, 0.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class DexsuiteReorientEnvCfg(ManagerBasedRLEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(-2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
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
    curriculum: CurriculumCfg | None = CurriculumCfg()

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
        self.sim.gravity = (0.0, 0.0, 0.0)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = (
                self.rewards.success.params["pos_std"] / 2
            )

            self.curriculum.adr.params["rot_tol"] = (
                self.rewards.success.params["rot_std"] / 2
            )


class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Dexsuite lift task definition"""

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


class DexsuiteReorientEnvCfg_PLAY(DexsuiteReorientEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]


class DexsuiteLiftEnvCfg_PLAY(DexsuiteLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.position_only = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]


# Tasks

from isaaclab_assets.robots import KUKA_ALLEGRO_CFG

# from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg

# from isaaclab.utils import configclass

# from ... import dexsuite_env_cfg as dexsuite


@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.1
    )


@configclass
class KukaAllegroReorientRewardCfg(RewardsCfg):

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )


@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        # self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = [
            "index_link_3",
            "middle_link_3",
            "ring_link_3",
            "thumb_link_3",
        ]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    # filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/square_table_leg2"],
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
        ].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=["palm_link", ".*_tip"]
        )


@configclass
class DexsuiteKukaAllegroReorientEnvCfg(KukaAllegroMixinCfg, DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroReorientEnvCfg_PLAY(
    KukaAllegroMixinCfg, DexsuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg(KukaAllegroMixinCfg, DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteKukaAllegroLiftEnvCfg_PLAY(KukaAllegroMixinCfg, DexsuiteLiftEnvCfg_PLAY):
    pass
