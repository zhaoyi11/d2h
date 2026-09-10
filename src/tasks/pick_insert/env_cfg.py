# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from src.tasks.common.observations_cfg import (
    ProprioObsCfg,
    PerceptionObsCfg,
    LowLevelObsCfg,
)

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
from src.tasks.common.env_cfg import HrlActionsCfg as CommonHrlActionsCfg, ObjectTerminationsCfg, configure_contact_physics, configure_fingertip_contacts, configure_success_visualization, fingertip_contact_observation, fingertip_transforms_cfg, ground_plane_cfg, light_cfg, table_cfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.pick_insert.mdps as mdp
from src.policy.high_level.trajectory_stepper import StageObjTol

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
                # max_depenetration_velocity=0.1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.2, 0.30), rot=(0.5, -0.5, 0.5, 0.5)),
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
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.275), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # table
    table: RigidObjectCfg = table_cfg()

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

    # plane
    plane = ground_plane_cfg()

    light = light_cfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.PickInsertTrajectoryObjectAndHandBasePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(10.0, 10.0),
        debug_vis=False,
        success_vis_asset_name="table",
        # Fail recovery: if the peg leaves the hand mid-episode, wait for it to come to rest and
        # regenerate the whole reach->insert trajectory from its new pose so the arm re-grasps it.
        enable_drop_recovery=True,
        recovery_settle_speed=0.05,
        recovery_settle_steps=5,
        # Hand<->object gap that counts as a drop. Grasped transport keeps this ~0.01-0.02 m; after a
        # drop the stalled arm stays fairly near the fallen peg (~0.14 m observed), so 0.20 was too
        # high to trigger. 0.10 sits well clear of normal transport and catches the drop.
        drop_object_hand_distance=0.10,
        # Arm recovery only after reach(0)+lift(1); a hand<->object gap while the grip is still forming
        # is not a drop. Capture the reach goal from the peg's settled pose.
        recovery_arm_after_stage=1,
        capture_goal_after_settle=True,
        # Lift-stall recovery (complements the distance drop detector, which can't see a lift failure:
        # the arm is held over the peg so the hand<->object gap never reaches drop_object_hand_distance).
        # If the peg sits off its goal and at rest through reach+lift (stage <= recovery_arm_after_stage)
        # for this many steps, the grasp failed -> replan from the peg's settled pose (re-open, re-grasp).
        # A normal lift dwells ~25 at-rest steps while the bounded correction ramps up before it raises
        # the peg, so this must sit well above that (sim: a successful lift peaks ~26); 60 (~2.3x) keeps
        # a working grasp from being aborted while still catching a genuinely stuck lift in ~1 s.
        grasp_stall_steps=60,
        # Grasp sequencing: keep the hand open through the reach stage (0), then close at lift.
        hand_open_until_stage=0,
        # Hand-base hold DISABLED (-1): the hand-base must stay derived from the object goal, because the
        # frozen low-level policy chases command[:,:7] (the in-hand object target) via goal_pos_diff. If
        # the hand-base is held while the object goal lifts +lift_height, command[:,:7] drifts +lift_height
        # from the actual grip, so the policy shoves the peg up through the fingers and drops it (lift
        # success regression). With -1 the hand-base follows the object goal -> command[:,:7] is a constant
        # grasp offset (goal_pos_diff ~= 0, stable grip) and the arm (MPC) lifts the peg directly.
        hand_base_hold_until_stage=-1,
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


    # # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()
    low_level: LowLevelObsCfg = LowLevelObsCfg()


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
            "mass_distribution_params": [0.010, 0.100],
            "operation": "abs",
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
    # jitter keeps it on the table; velocity zeroed so it settles gently under reduced gravity.
    # There is no hand/object
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
    # Fixed reduced gravity for the frozen hand policy.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.81], [0.0, 0.0, -1.81]), # TODO: change to [0.0, 0.0, -9.81] for full gravity
            "operation": "abs",
        },
    )


@configclass
class HrlActionsCfg(CommonHrlActionsCfg):
    arm_action = CommonHrlActionsCfg().arm_action.replace(
        obstacle_cuboids={"static_obstacle": {"dims": [0.05, 0.05, 0.25], "pose": [0.5, -0.25, 0.38, 1, 0, 0, 0]}},
    )


@configclass
class DexsuiteFrankaLeapInsertHrlEnvCfg(ManagerBasedRLEnvCfg):
    """Reach, establish a grasp, and insert the peg with a frozen hand policy."""

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
        self.decimation = 4
        self.episode_length_s = 20.0
        self.is_finite_horizon = True
        self.sim.dt = 1 / 120
        configure_contact_physics(self.sim, collision_stack_size=2**30)
        configure_success_visualization(self.commands.object_pose, self.scene.table)
        self.commands.object_pose.body_name = "base"
        self.commands.object_pose.position_only = False

        self.scene.fingertip_transforms = fingertip_transforms_cfg()
        # Object is filter zero; remaining filters are external contact surfaces.
        configure_fingertip_contacts(self.scene, [
            "{ENV_REGEX_NS}/Object",
            "{ENV_REGEX_NS}/ReceptiveObject",
            "{ENV_REGEX_NS}/Table",
        ])
        self.observations.proprio.contact = fingertip_contact_observation(filter_indices=[0])
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [".*fingertip.*"]

        self.commands.object_pose.enable_grasp_establish = True
        self.commands.object_pose.trajectory_segment_steps = (0, 1, 1, 2, 1, 1, 1, 1)
        self.commands.object_pose.stage_object_tolerances = (
            StageObjTol(0.02, 0.3),  # reach
            StageObjTol(0.02, 0.3),  # establish grasp
            StageObjTol(0.02, 0.1),  # lift
            StageObjTol(0.02, 0.3),  # move
            StageObjTol(0.02, 0.2),  # align
            StageObjTol(0.01, 0.2),  # approach
            StageObjTol(0.005, 0.1),  # insert
            StageObjTol(0.005, 0.1),  # hold
        )
        self.commands.object_pose.recovery_arm_after_stage = 2
        self.commands.object_pose.hand_base_hold_until_stage = 1

        # The MPC action restores arm gains on reset; the hand stays position-controlled.
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
