# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotate-object task: rotate an upright object about its fixed world-Z axis.

Mirrors the ``pick_insert`` HRL stack: a scripted object-pose trajectory command drives a cuRobo-MPC
arm (hand ``base``) and a frozen ``dex_reorient`` low-level LEAP-hand policy. The base ``-v0`` env is
the RL/joint-control variant; ``...HrlEnvCfg`` is the HRL variant consumed by ``scripts/hrl/play.py``.
"""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg

import src.tasks.rotate_knob.mdps as mdp
import src.tasks.reorient.mdps as task_mdps
from src.tasks.rotate_knob.mdps.contact_filters import (
    contact_filter_prim_paths,
    external_indices,
    object_indices,
)
from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite scene: Franka+LEAP, a Z-axis-constrained object, a plate, and a table."""

    # robot
    robot = FRANKA_LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # The ARIA knob handle is connected to the world by an unbounded revolute Z joint. Its authored
    # baseline is z=0, so the spawn Z matches the table's top surface at z=0.255.
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "aria/knob1/knob1_handle.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
                kinematic_enabled=False,
                enable_gyroscopic_forces=True,
            ),
            scale=(0.65, 0.65, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.20, 0.255), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # receptive_object: the plate (placement target). Kinematic (fixed placement surface).
    receptive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "uwlab/plate.usd"),
            scale=(0.6, 0.6, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=False,
                kinematic_enabled=True,
                max_depenetration_velocity=1.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.01, rest_offset=0.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -0.20, 0.255), rot=(1.0, 0.0, 0.0, 0.0)),
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
            pos=(0.55, 0.0, 0.135), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # static obstacle: a fixed pillar the MPC must route around, off to the side (clear of the
    # object/plate/arm path). Visual-only (no collision_props): it exists physically only in the
    # cuRobo planning world. Mirrored as a static "static_obstacle" cuboid in the arm action.
    # static_obstacle: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/StaticObstacle",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.05, 0.05, 0.25),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.85)),
    #         visible=True,
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.5, -0.25, 0.38), rot=(1.0, 0.0, 0.0, 0.0)
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

    object_pose = mdp.RotateObjectTrajectoryObjectAndHandBasePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        success_vis_asset_name="table",
        object_to_anchor_pose=(0.0, 0.0, 0.01, 0.70710678, 0.0, 0.70710678, 0.0),
        enable_drop_recovery=False,
        # Keep the hand open for the initial reach, then close while following yaw goals.
        hand_open_until_stage=0,
        # Hand-base follows each object goal so command[:, :7] stays a stable grasp offset while the
        # arm and hand execute the requested yaw.
        hand_base_hold_until_stage=-1,
        correction=mdp.AnchorCorrectionCfg(
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
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
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
            clip=(-2.0, 2.0),
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
        """Reorient-style current-step state used by the frozen low-level VAE (155-dim layout)."""

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
                "contact_pose_range_deg": 90.0,
                "filter_indices": object_indices(),
            },
        )
        # -- external contact: plate + table, vector-summed per fingertip (16 dims) -> 155-dim layout.
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

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()
    low_level: LowLevelObsCfg | None = None


@configclass
class EventCfg:
    """Configuration for randomization."""

    anchor_object_z_axis_joint = EventTerm(
        func=mdp.anchor_object_z_axis_joint,
        mode="prestartup",
        params={"asset_name": "object", "joint_name": "z_axis_joint"},
    )

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

    # Keep the object at its joint anchor with a deterministic upright pose on every reset.
    reset_object: EventTerm | None = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0]},
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

    # Reduced-gravity curriculum (matches pick_insert / the frozen dex_reorient policy's regime).
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.81], [0.0, 0.0, -1.81]),
            "operation": "abs",
        },
    )


@configclass
class ActionsCfg:
    """Base (non-HRL) joint-control action."""

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.1
    )


@configclass
class HrlActionsCfg:
    """Low-level action interface consumed by the HRL chunk wrapper."""

    # cuRobo reactive MPC: drives the hand `base` to the command anchor. The full Franka+LEAP collision
    # model plans around the static_obstacle pillar (the table/plate are NOT obstacles so the hand can
    # reach down to grasp the object and lower it onto the plate).
    arm_action = mdp.CommandHandBaseCuroboMpcActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="base",
        command_name="object_pose",
        robot_config_file=(
            f"{Path(__file__).resolve().parents[2]}/assets/franka_leap_hand/curobo/franka_leap.yml"
        ),
        obstacle_cuboids={
            # "static_obstacle": {"dims": [0.05, 0.05, 0.25], "pose": [0.5, -0.25, 0.38, 1, 0, 0, 0]},
        },
    )
    hand_action = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    The yaw term reads the trajectory command term's current object goal directly rather than the
    14-D public hand-relative command representation.
    """

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.002)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.25}, weight=1.0)

    # bool award if >=2 finger tips (one being the thumb) contact the object.
    good_finger_contact = RewTerm(func=mdp.contacts, weight=1.0, params={"threshold": 1.0})

    yaw_tracking = RewTerm(
        func=mdp.trajectory_yaw_tracking,
        weight=4.0,
        params={"command_name": "object_pose", "std": 0.5},
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


@configclass
class DexsuiteRotateObjectEnvCfg(ManagerBasedRLEnvCfg):
    """Base rotate-object env: scripted trajectory command + joint-control actions."""

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
    # Scripted-rollout task: no ADR curriculum (the trajectory/hand policy drive the motion).
    curriculum: mdp.CurriculumCfg | None = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # 60 Hz control

        # Goal-driven resampling; the command term replaces each achieved yaw target immediately.
        self.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)
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

        self.episode_length_s = 20.0
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


class DexsuiteRotateObjectEnvCfg_PLAY(DexsuiteRotateObjectEnvCfg):
    """Rotate-object evaluation environment definition."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.debug_vis = True


##########
#  Tasks
##########


@configclass
class FrankaLeapMixinCfg:

    def __post_init__(self: DexsuiteRotateObjectEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "base"
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
        # object (index 0) vs external (plate + table) contact via filter_indices.
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
            clip=(-20.0, 20.0),
        )
        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )


@configclass
class DexsuiteFrankaLeapRotateObjectEnvCfg(FrankaLeapMixinCfg, DexsuiteRotateObjectEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapRotateObjectEnvCfg_PLAY(
    FrankaLeapMixinCfg, DexsuiteRotateObjectEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteFrankaLeapRotateObjectHrlEnvCfg(FrankaLeapMixinCfg, DexsuiteRotateObjectEnvCfg):
    """HRL variant that reaches the fixed object and follows successive world-Z yaw goals."""

    actions: HrlActionsCfg = HrlActionsCfg()

    def __post_init__(self):
        self.observations.low_level = ObservationsCfg.LowLevelObsCfg()
        super().__post_init__()
        # cuRobo MPC owns the arm: zero the arm position gains (the MPC action restores them on
        # reset); the LEAP hand ("fingers") stays position-controlled.
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
