# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from src.tasks.common.observations_cfg import (
    ProprioObsCfg,
    PerceptionObsCfg,
)
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
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from src.tasks.common.env_cfg import (
    ArmHandActionsCfg,
    configure_contact_physics,
    configure_fingertip_contacts,
    fingertip_contact_observation,
    fingertip_transforms_cfg,
    light_cfg,
)
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.common.mdps as task_mdps
from src.tasks.clean_table_omnireset.mdps import task_mdps as mdp
from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG
from src.tasks.clean_table_omnireset.mdps.contact_filters import (
    contact_filter_prim_paths,
    object_indices,
)


_SRC_ROOT = Path(__file__).resolve().parents[2]


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
    light = light_cfg()


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


    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


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
    reset_scene_to_default = EventTerm(
        func=task_mdps.reset_scene_to_default,
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
    # lift = RewTerm(func=mdp.lift_reward, weight=2.0)
    # transport = RewTerm(func=mdp.transport_reward, weight=3.0)
    # inside_box = RewTerm(func=mdp.inside_box_reward, weight=8.0)
    success = RewTerm(
        func=mdp.success_reward,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "box_cfg": SceneEntityCfg("receptive_object"),
            "hand_base_cfg": SceneEntityCfg("robot", body_names="base"),
        },
        weight=250.0,
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
    actions: ArmHandActionsCfg = ArmHandActionsCfg()
    commands = None
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):

        self.scene.fingertip_transforms = fingertip_transforms_cfg()
        configure_fingertip_contacts(self.scene, contact_filter_prim_paths())
        self.observations.proprio.contact = fingertip_contact_observation(filter_indices=object_indices())
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
        configure_contact_physics(self.sim, found_lost_pairs=2**22, collision_stack_size=2**30)


__all__ = ["DexsuiteFrankaLeapCleanTableOmniResetEnvCfg"]
