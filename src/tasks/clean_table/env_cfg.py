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
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from src.tasks.common.env_cfg import TabletopSceneCfg, TabletopEventsCfg, PlacementRewardsCfg, tote_cfg, HrlActionsCfg, JointActionsCfg, ObjectTerminationsCfg, configure_contact_physics, configure_fingertip_contacts, configure_success_visualization, fingertip_contact_observation, fingertip_transforms_cfg, get_visdex_usd_paths
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.clean_table.mdps as mdp
from src.tasks.clean_table.mdps.contact_filters import contact_filter_prim_paths, object_indices
from src.policy.high_level.trajectory_stepper import StageObjTol

# UWLAB_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main"


@configclass
class SceneCfg(TabletopSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""

    # robot

    # object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=get_visdex_usd_paths(),
            random_choice=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            scale=(0.8, 0.8, 0.8),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.55, 0.10, 0.34],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
    )

    receptive_object = tote_cfg()

    # table

    # plane


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.CleanTableTrajectoryObjectAndHandBasePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        box_name="receptive_object",
        table_name="table",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        success_vis_asset_name="table",
        position_only=False,
        enable_drop_recovery=True,
        recovery_settle_speed=0.05,
        recovery_settle_steps=5,
        drop_object_hand_distance=0.10,
        recovery_arm_after_stage=1,
        capture_goal_after_settle=True,
        grasp_stall_steps=60,
        hand_open_until_stage=0,
        hand_base_hold_until_stage=-1,
        lift_height=0.08,
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

        object_pos_b = ObsTerm(
            func=mdp.object_pos_b, noise=Unoise(n_min=-0.0, n_max=0.0)
        )

        object_quat_b = ObsTerm(
            func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0)
        )
        box_pose_b = ObsTerm(
            func=mdp.box_pose_b,
            params={"box_cfg": SceneEntityCfg("receptive_object")},
        )
        object_pos_box = ObsTerm(
            func=mdp.object_pos_box,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "box_cfg": SceneEntityCfg("receptive_object"),
            },
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


    # early_termination = RewTerm(
    #     func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"}
    # )


    # abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class DexsuiteReorientEnvCfg(ManagerBasedRLEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
    )
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    sim: SimulationCfg = SimulationCfg(gravity=(0.0, 0.0, -9.81))
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: JointActionsCfg = JointActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: PlacementRewardsCfg = PlacementRewardsCfg()
    terminations: ObjectTerminationsCfg = ObjectTerminationsCfg()
    events: TabletopEventsCfg = TabletopEventsCfg()
    curriculum: mdp.CurriculumCfg | None = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2  # 60 Hz

        configure_success_visualization(self.commands.object_pose, self.scene.table)


        self.episode_length_s = 20.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120
        configure_contact_physics(self.sim, found_lost_pairs=2**22, collision_stack_size=2**30)


        self.scene.fingertip_transforms = fingertip_transforms_cfg()
        configure_fingertip_contacts(self.scene, contact_filter_prim_paths())
        self.observations.proprio.contact = fingertip_contact_observation(filter_indices=object_indices())

        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )


class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Dexsuite lift task definition"""

    def __post_init__(self):
        super().__post_init__()


class DexsuiteReorientEnvCfg_PLAY(DexsuiteReorientEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()


class DexsuiteLiftEnvCfg_PLAY(DexsuiteLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()


##########
#  Tasks
##########


@configclass
class DexsuiteFrankaLeapCleanTableEnvCfg(DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapCleanTableEnvCfg_PLAY(
    DexsuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteFrankaLeapCleanTableHrlEnvCfg(
    DexsuiteReorientEnvCfg
):
    """Autonomous clean-table environment driven by cuRobo and a frozen hand policy."""

    actions: HrlActionsCfg = HrlActionsCfg()

    def __post_init__(self):
        self.observations.low_level = LowLevelObsCfg()
        super().__post_init__()
        self.commands.object_pose.hand_base_hold_until_stage = 1
        self.commands.object_pose.drop_object_hand_distance = 0.12
        self.commands.object_pose.lift_height = 0.10
        self.commands.object_pose.stage_object_tolerances = (
            StageObjTol(0.02, 0.3),
            StageObjTol(0.02, 0.3),
            *self.commands.object_pose.stage_object_tolerances[2:],
        )
        self.decimation = 4  # 30 Hz, matching the frozen hand policy.
        self.episode_length_s = 30.0
        self.sim.render_interval = self.decimation
        configure_contact_physics(self.sim)


        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
