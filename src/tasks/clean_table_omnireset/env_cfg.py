# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.common.mdps as task_mdps
from src.tasks.clean_table_omnireset.mdps import task_mdps as mdp
from src.tasks.clean_table_omnireset.mdps.events import ResetSceneFromInstantDexterity
from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG
from src.tasks.clean_table_omnireset.mdps.contact_filters import (
    contact_filter_prim_paths,
    object_indices,
)


_SRC_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RESET_DATASET_DIR = Path(__file__).resolve().parents[3] / "dataets/clean_table_bc"


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = FRANKA_LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FRANKA_LEAP_HAND_CFG.spawn.replace(
            articulation_props=FRANKA_LEAP_HAND_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=False,
            ),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_SRC_ROOT / "assets/visdex_objects/USD/104738/104738.usd"),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
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
            pos=(0.55, 0.10, 0.34),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    receptive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(_SRC_ROOT / "assets/symdex/tote_collision.usd"),
            scale=(0.6, 0.6, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, -0.35, 0.271),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
        ),
    )
    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.235),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
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
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        object_pos_b = ObsTerm(func=task_mdps.object_pos_b, noise=Unoise(n_min=0.0, n_max=0.0))
        object_quat_b = ObsTerm(func=task_mdps.object_quat_b, noise=Unoise(n_min=0.0, n_max=0.0))
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
        actions = ObsTerm(func=task_mdps.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=task_mdps.joint_pos, noise=Unoise(n_min=0.0, n_max=0.0))
        joint_vel = ObsTerm(func=task_mdps.joint_vel, noise=Unoise(n_min=0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=task_mdps.body_state_b,
            noise=Unoise(n_min=0.0, n_max=0.0),
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
            func=task_mdps.object_point_cloud_b,
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class ActionsCfg:
    arm_action = task_mdps.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.1
    )
    hand_action = task_mdps.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class EventCfg:
    robot_physics_material = EventTerm(
        func=task_mdps.randomize_rigid_body_material,
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
        func=task_mdps.randomize_rigid_body_material,
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
        func=task_mdps.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.5, 2.0],
            "damping_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )
    joint_friction = EventTerm(
        func=task_mdps.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )
    object_scale_mass = EventTerm(
        func=task_mdps.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )
    variable_gravity = EventTerm(
        func=task_mdps.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.81], [0.0, 0.0, -1.81]),
            "operation": "abs",
        },
    )
    reset_from_dataset = EventTerm(
        func=ResetSceneFromInstantDexterity,
        mode="reset",
        params={},
    )


@configclass
class RewardsCfg:
    action_l2 = RewTerm(func=task_mdps.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=task_mdps.action_rate_l2_clamped, weight=-0.005)
    fingers_to_object = RewTerm(
        func=task_mdps.object_ee_distance, params={"std": 0.4}, weight=1.0
    )
    good_finger_contact = RewTerm(
        func=task_mdps.contacts, params={"threshold": 1.0}, weight=0.5
    )
    lift = RewTerm(func=mdp.lift_reward, weight=2.0)
    transport = RewTerm(func=mdp.transport_reward, weight=3.0)
    inside_box = RewTerm(func=mdp.inside_box_reward, weight=8.0)
    success = RewTerm(
        func=mdp.success_reward,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "box_cfg": SceneEntityCfg("receptive_object"),
            "hand_base_cfg": SceneEntityCfg("robot", body_names="base"),
        },
        weight=10.0,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=task_mdps.time_out, time_out=True)
    object_outside_table = DoneTerm(
        func=mdp.object_outside_table,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "table_cfg": SceneEntityCfg("table"),
        },
    )
    success = DoneTerm(
        func=mdp.placed_and_released,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "box_cfg": SceneEntityCfg("receptive_object"),
            "hand_base_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class DexsuiteFrankaLeapCleanTableOmniResetEnvCfg(ManagerBasedRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
    )
    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=3, replicate_physics=False)
    sim: SimulationCfg = SimulationCfg(gravity=(0.0, 0.0, -9.81))
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None
    reset_dataset_dir: str = str(_DEFAULT_RESET_DATASET_DIR)

    def __post_init__(self):
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
            func=task_mdps.fingers_contact_force_b,
            params={
                "contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list],
                "filter_indices": object_indices(),
            },
            clip=(-20.0, 20.0),
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = [
            ".*fingertip.*"
        ]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )

        self.decimation = 2
        self.episode_length_s = 20.0
        self.is_finite_horizon = True
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**22
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**30


__all__ = ["DexsuiteFrankaLeapCleanTableOmniResetEnvCfg"]
