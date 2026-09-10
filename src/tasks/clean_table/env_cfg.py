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
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from src.tasks.common.env_cfg import TabletopSceneCfg, TabletopEventsCfg, tote_cfg, HrlActionsCfg, ObjectTerminationsCfg, configure_contact_physics, configure_fingertip_contacts, configure_success_visualization, fingertip_contact_observation, fingertip_transforms_cfg, get_visdex_usd_paths
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.clean_table.mdps as mdp
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
    low_level: LowLevelObsCfg = LowLevelObsCfg()


@configclass
class DexsuiteFrankaLeapCleanTableHrlEnvCfg(ManagerBasedRLEnvCfg):
    """Place objects in the tote with cuRobo and a frozen hand policy."""

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
    events: TabletopEventsCfg = TabletopEventsCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.is_finite_horizon = True
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        configure_contact_physics(self.sim)
        configure_success_visualization(self.commands.object_pose, self.scene.table)

        self.scene.fingertip_transforms = fingertip_transforms_cfg()
        # Object is filter zero; remaining filters are external contact surfaces.
        configure_fingertip_contacts(self.scene, [
            "{ENV_REGEX_NS}/Object/baseLink*",
            "{ENV_REGEX_NS}/ReceptiveObject",
            "{ENV_REGEX_NS}/Table",
        ])
        self.observations.proprio.contact = fingertip_contact_observation(filter_indices=[0])
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [".*fingertip.*"]

        self.commands.object_pose.hand_base_hold_until_stage = 1
        self.commands.object_pose.drop_object_hand_distance = 0.12
        self.commands.object_pose.lift_height = 0.10
        self.commands.object_pose.stage_object_tolerances = (
            StageObjTol(0.02, 0.3),
            StageObjTol(0.02, 0.3),
            *self.commands.object_pose.stage_object_tolerances[2:],
        )

        # The MPC action restores arm gains on reset; the hand stays position-controlled.
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
