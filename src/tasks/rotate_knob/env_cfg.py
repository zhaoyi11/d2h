# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotate-object task: rotate an upright object about its fixed world-Z axis.

Mirrors the ``pick_insert`` HRL stack: a scripted object-pose trajectory command drives a cuRobo-MPC
arm (hand ``base``) and a frozen ``dex_reorient`` low-level LEAP-hand policy. The base ``-v0`` env is
the RL/joint-control variant; ``...HrlEnvCfg`` is the HRL variant consumed by ``scripts/instant_dexterity.py``.
"""

from src.tasks.common.observations_cfg import (
    ProprioObsCfg,
    PerceptionObsCfg,
    LowLevelObsCfg,
)
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from src.tasks.common.env_cfg import (
    HrlActionsCfg,
    JointActionsCfg,
    ObjectTerminationsCfg,
    configure_contact_physics,
    configure_fingertip_contacts,
    configure_success_visualization,
    fingertip_contact_observation,
    fingertip_transforms_cfg,
    ground_plane_cfg,
    light_cfg,
    table_cfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.rotate_knob.mdps as mdp
from src.tasks.rotate_knob.mdps.contact_filters import contact_filter_prim_paths, object_indices
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
    table: RigidObjectCfg = table_cfg(pos=(0.55, 0.0, 0.135))

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
    plane = ground_plane_cfg()

    light = light_cfg()


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
    actions: JointActionsCfg = JointActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: ObjectTerminationsCfg = ObjectTerminationsCfg()
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
        configure_success_visualization(self.commands.object_pose, self.scene.table)


        self.episode_length_s = 20.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120

        # Contact and solver settings
        configure_contact_physics(self.sim, found_lost_pairs=2**22, collision_stack_size=2**30)

        self.commands.object_pose.body_name = "base"

        self.scene.fingertip_transforms = fingertip_transforms_cfg()
        # Filter against the full target registry [Object, ReceptiveObject, Table] so
        # force_matrix_w column k == CONTACT_FILTER_TARGETS[k]; the low-level obs then splits
        # object (index 0) vs external (plate + table) contact via filter_indices.
        configure_fingertip_contacts(self.scene, contact_filter_prim_paths())
        self.observations.proprio.contact = fingertip_contact_observation(filter_indices=object_indices())
        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )


class DexsuiteRotateObjectEnvCfg_PLAY(DexsuiteRotateObjectEnvCfg):
    """Rotate-object evaluation environment definition."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.debug_vis = True


##########
#  Tasks
##########


@configclass
class DexsuiteFrankaLeapRotateObjectEnvCfg(DexsuiteRotateObjectEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapRotateObjectEnvCfg_PLAY(
    DexsuiteRotateObjectEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteFrankaLeapRotateObjectHrlEnvCfg(DexsuiteRotateObjectEnvCfg):
    """HRL variant that reaches the fixed object and follows successive world-Z yaw goals."""

    actions: HrlActionsCfg = HrlActionsCfg()

    def __post_init__(self):
        self.observations.low_level = LowLevelObsCfg()
        super().__post_init__()
        # cuRobo MPC owns the arm: zero the arm position gains (the MPC action restores them on
        # reset); the LEAP hand ("fingers") stays position-controlled.
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
