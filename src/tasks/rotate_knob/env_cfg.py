# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rotate an upright knob about its fixed world-Z axis using the HRL controller."""

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
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from src.tasks.common.env_cfg import HrlActionsCfg, ObjectTerminationsCfg, configure_contact_physics, configure_fingertip_contacts, configure_success_visualization, fingertip_contact_observation, fingertip_transforms_cfg, ground_plane_cfg, light_cfg, table_cfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.rotate_knob.mdps as mdp
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
    low_level: LowLevelObsCfg = LowLevelObsCfg()


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

    # Reduced gravity matches the frozen dex_reorient policy's regime.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.81], [0.0, 0.0, -1.81]),
            "operation": "abs",
        },
    )

@configclass
class DexsuiteFrankaLeapRotateObjectHrlEnvCfg(ManagerBasedRLEnvCfg):
    """Reach the fixed knob and follow successive world-Z yaw goals."""

    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
    )
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    sim: SimulationCfg = SimulationCfg(gravity=(0.0, 0.0, -9.81))
    observations: ObservationsCfg = ObservationsCfg()
    actions: HrlActionsCfg = HrlActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards = None
    terminations: ObjectTerminationsCfg = ObjectTerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 20.0
        self.is_finite_horizon = True
        self.sim.dt = 1 / 120
        configure_contact_physics(self.sim, found_lost_pairs=2**22, collision_stack_size=2**30)
        configure_success_visualization(self.commands.object_pose, self.scene.table)
        self.commands.object_pose.body_name = "base"
        self.commands.object_pose.position_only = False

        self.scene.fingertip_transforms = fingertip_transforms_cfg()
        # Object is filter zero; remaining filters are external contact surfaces.
        configure_fingertip_contacts(self.scene, [
            "{ENV_REGEX_NS}/Object/handle",
            "{ENV_REGEX_NS}/ReceptiveObject",
            "{ENV_REGEX_NS}/Table",
        ])
        self.observations.proprio.contact = fingertip_contact_observation(filter_indices=[0])
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [".*fingertip.*"]

        # The MPC action restores arm gains on reset; the hand stays position-controlled.
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
