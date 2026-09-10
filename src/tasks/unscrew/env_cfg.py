# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from src.tasks.common.observations_cfg import (
    ProprioObsCfg,
    PerceptionObsCfg,
    LowLevelObsCfg as SharedLowLevelObsCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
from src.tasks.common.env_cfg import (
    HrlActionsCfg as CommonHrlActionsCfg,
    JointActionsCfg,
    configure_contact_physics,
    configure_fingertip_contacts,
    configure_success_visualization,
    fingertip_contact_observation,
    fingertip_transforms_cfg,
    ground_plane_cfg,
    light_cfg,
    table_cfg,
)
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.common.mdps as mdp
import src.tasks.unscrew.mdps as unscrew_mdp
from src.tasks.unscrew.mdps.contacts import contact_filter_prim_paths, object_indices
from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG

UWLAB_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main"

ASSET_SCALE = 1.5
RECEPTIVE_OBJECT_POS = (0.55, 0.0, 0.271)
# INSTALLED_OBJECT_POS = (0.465625, 0.084375, 0.341834)
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

INSTALLED_OBJECT_POS = (0.4659913182258606, 0.08350233733654022, 0.3423238694667816)
INSTALLED_OBJECT_QUAT = (0.97566819190979, 0.00011654444824671373, -0.0003023587341886014, 0.21925236284732819)

EXTRACTED_OBJECT_POS = (0.465625, 0.084375, 0.416834)
TABLE_POS = (0.55, 0.0, 0.235)


@configclass
class LowLevelObsCfg(SharedLowLevelObsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.contact_pose.params["contact_pose_range_deg"] = 45.0


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Square-table scene with one leg initialized in a threaded socket."""

    # robot
    robot = FRANKA_LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd",
            scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=1.0555,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=INSTALLED_OBJECT_POS, rot=INSTALLED_OBJECT_QUAT),
    )

    receptive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd",
            scale=(ASSET_SCALE, ASSET_SCALE, ASSET_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                # contact_offset=0.0005,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=RECEPTIVE_OBJECT_POS, rot=IDENTITY_QUAT),
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
    #     spawn=sim_utils.UsdFileCfg(https://github.com/mattpocock/skills/tree/main
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
    table: RigidObjectCfg = table_cfg(pos=TABLE_POS, rot=IDENTITY_QUAT)

    # plane
    plane = ground_plane_cfg()

    light = light_cfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(20.0, 20.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(EXTRACTED_OBJECT_POS[0], EXTRACTED_OBJECT_POS[0]),
            pos_y=(EXTRACTED_OBJECT_POS[1], EXTRACTED_OBJECT_POS[1]),
            pos_z=(EXTRACTED_OBJECT_POS[2], EXTRACTED_OBJECT_POS[2]),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
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
            "mass_distribution_params": [0.05, 0.2],
            "operation": "abs",
        },
    )

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
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
            "position_range": [0.0, 0.0],
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

    orientation_tracking = None

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
            "pos_std": 0.01,
            "rot_std": None,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    completion = RewTerm(
        func=mdp.is_terminated_term,
        weight=10.0,
        params={"term_keys": "unscrew_success"},
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

    unscrew_success = DoneTerm(
        func=unscrew_mdp.StableUnscrewSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "clearance_margin": 0.030,
            "force_threshold": 1.0,
        },
    )


@configclass
class DexsuiteUnscrewEnvCfg(ManagerBasedRLEnvCfg):
    """Flat RL task for extracting a threaded square-table leg."""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
    )
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: JointActionsCfg = JointActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: mdp.CurriculumCfg | None = mdp.CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 50 Hz

        # *single-goal setup
        self.commands.object_pose.resampling_time_range = (20.0, 20.0)
        self.commands.object_pose.position_only = True
        configure_success_visualization(self.commands.object_pose, self.scene.table)


        self.episode_length_s = 20.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        # Contact and solver settings
        configure_contact_physics(self.sim)
        # self.sim.physx.contact_offset = 10


        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.bounce_threshold_velocity = 0.01
        # self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
        # self.sim.physx.gpu_collision_stack_size = 2**28

        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = (
                self.rewards.success.params["pos_std"] / 2
            )

            self.curriculum.adr.params["rot_tol"] = None

        self.commands.object_pose.body_name = "base"  # TODO: check this !!

        configure_fingertip_contacts(self.scene, ['{ENV_REGEX_NS}/Object'])
        self.observations.proprio.contact = fingertip_contact_observation()

        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )


class DexsuiteUnscrewEnvCfg_PLAY(DexsuiteUnscrewEnvCfg):
    """Unscrew evaluation environment definition."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (20.0, 20.0)
        self.commands.object_pose.debug_vis = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]

##########
#  Tasks
##########


@configclass
class DexsuiteFrankaLeapUnscrewEnvCfg(DexsuiteUnscrewEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapUnscrewEnvCfg_PLAY(
    DexsuiteUnscrewEnvCfg_PLAY
):
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

    object_pose = unscrew_mdp.UnscrewTrajectoryObjectAndHandBasePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(20.0, 20.0),
        debug_vis=False,
        success_vis_asset_name="table",
        # Rotate positive yaw by pi in 30-degree stages while rising 5 mm, then lift 10 cm along +Z.
        # The frozen hand policy supplies in-hand rotation; the hand-base orientation stays fixed.
        twist_total_angle=2.0 * math.pi,
        twist_segments=6,
        thread_pitch=0.02,
        extraction_height=0.100,
        enable_clearance_transition=True,
        receptive_name="receptive_object",
        clearance_transition_margin=0.005,
        clearance_transition_stable_steps=1,
        # Fail recovery: if the leg leaves the hand mid-episode, wait for it to come to rest and
        # regenerate the reach->unscrew trajectory from its new pose so the arm re-grasps it.
        enable_drop_recovery=True,
        recovery_settle_speed=0.05,
        recovery_settle_steps=5,
        drop_object_hand_distance=0.10,
        # Arm recovery only after reach(0)+grasp(1); capture the goal from the settled threaded pose.
        recovery_arm_after_stage=1,
        capture_goal_after_settle=True,
        # Keep the hand open during reach, close during contact confirmation, then begin turning.
        hand_open_until_stage=0,
        hand_base_hold_until_stage=1,
        # Nudge the hand-base anchor so the object reaches its goal when the in-hand policy alone
        # cannot; bounded, memoryless, gated on a settled arm + stalled object (as in pick_insert).
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
class HrlActionsCfg(CommonHrlActionsCfg):
    arm_action = CommonHrlActionsCfg().arm_action.replace(
        obstacle_cuboids={"table": {"dims": [0.8, 1.5, 0.04], "pose": [0.55, 0.0, 0.235, 1, 0, 0, 0]}},
    )


@configclass
class DexsuiteFrankaLeapUnscrewHrlEnvCfg(DexsuiteUnscrewEnvCfg):
    """Unscrew HRL variant: cuRobo-MPC arm + frozen low-level LEAP-hand policy.

    The 7-DOF arm tracks the command hand-base anchor via cuRobo reactive MPC (0 external action
    dims); the 16-DOF LEAP hand is driven by a frozen reorient policy reading the ``low_level``
    observation group (167 dims, incl. external-contact force sensing). No high-level RL policy —
    a smoke script feeds the hand policy's 16 actions while the MPC drives the arm.
    """

    commands: HrlCommandsCfg = HrlCommandsCfg()
    actions: HrlActionsCfg = HrlActionsCfg()

    def __post_init__(self):
        # Enable the reorient-style 167-dim low-level observation group.
        self.observations.low_level = LowLevelObsCfg()
        super().__post_init__()
        # The flat task's goal is position-only; the scripted HRL stepper must still enforce each
        # 30-degree orientation target.
        self.commands.object_pose.position_only = False

        # FrameTransformer over the fingertips: tip_contact_* read its target_quat_w to rotate
        # contact forces into each fingertip frame.
        self.scene.fingertip_transforms = fingertip_transforms_cfg()

        # Rebuild the fingertip contact sensors to filter against the full target registry
        # [Object, ReceptiveObject, Table] so force_matrix_w column k == CONTACT_FILTER_TARGETS[k].
        # (Replaces the base task's object-only contact filter.)
        configure_fingertip_contacts(self.scene, contact_filter_prim_paths())
        # Keep the proprio object-contact channel object-only now that sensors carry 3 filters.
        self.observations.proprio.contact.params["filter_indices"] = object_indices()

        # cuRobo MPC owns the arm: zero the arm's position gains so set_joint_position_target has
        # no PD authority until the MPC action restores them on reset (mirrors pick_insert demo).
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0

        # Reduced gravity (-1.81) is the regime the frozen reorient hand policy was trained under
        # (matches pick_insert/cupcake); the flat env retains its gravity curriculum.
        self.events.variable_gravity.params["gravity_distribution_params"] = (
            [0.0, 0.0, -1.81],
            [0.0, 0.0, -1.81],
        )
