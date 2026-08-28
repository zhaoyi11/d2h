# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint arm-and-hand rotate-object task with recorded OmniReset initial states."""

from dataclasses import MISSING
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.common.mdps as task_mdps
from src.tasks.rotate_object.env_cfg import SceneCfg
from src.tasks.rotate_object.mdps.contact_filters import (
    contact_filter_prim_paths,
    object_indices,
)
from src.tasks.rotate_object.mdps.task_mdps import anchor_object_z_axis_joint
from src.tasks.rotate_object_omnireset.mdps import commands as command_mdp
from src.tasks.rotate_object_omnireset.mdps import events as event_mdp
from src.tasks.rotate_object_omnireset.mdps import task_mdps as mdp
from src.tasks.rotate_object_omnireset.mdps.curriculums import CurriculumCfg


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FINGERTIP_NAMES = [
    "thumb_fingertip",
    "fingertip",
    "fingertip_2",
    "fingertip_3",
]


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        object_pos_b = ObsTerm(
            func=task_mdps.object_pos_b,
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        object_quat_b = ObsTerm(
            func=task_mdps.object_quat_b,
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        object_lin_vel_b = ObsTerm(func=mdp.object_lin_vel_robot_b)
        object_ang_vel_b = ObsTerm(func=mdp.object_ang_vel_robot_b)
        target_object_pose_b = ObsTerm(
            func=task_mdps.generated_commands,
            params={"command_name": "object_pose"},
        )
        actions = ObsTerm(func=task_mdps.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=task_mdps.joint_pos,
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        joint_vel = ObsTerm(
            func=task_mdps.joint_vel,
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
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

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()


@configclass
class CommandsCfg:
    object_pose = command_mdp.RecordedObjectPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    arm_action = task_mdps.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.1,
    )
    hand_action = task_mdps.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class EventCfg:
    anchor_object_z_axis_joint = EventTerm(
        func=anchor_object_z_axis_joint,
        mode="prestartup",
        params={"asset_name": "object", "joint_name": "z_axis_joint"},
    )
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
            "gravity_distribution_params": (
                [0.0, 0.0, -9.81],
                [0.0, 0.0, -9.81],
            ),
            "operation": "abs",
        },
    )
    reset_from_dataset = EventTerm(
        func=event_mdp.ResetSceneFromInstantDexterity,
        mode="reset",
        params={},
    )


@configclass
class RewardsCfg:
    action_l2 = RewTerm(func=task_mdps.action_l2_clamped, weight=-0.002)
    action_rate_l2 = RewTerm(
        func=task_mdps.action_rate_l2_clamped,
        weight=-0.005,
    )
    fingers_to_object = RewTerm(
        func=task_mdps.object_ee_distance,
        params={"std": 0.25},
        weight=0.1,
    )
    good_finger_contact = RewTerm(
        func=task_mdps.contacts,
        params={"threshold": 1.0},
        weight=0.1,
    )
    yaw_tracking = RewTerm(
        func=mdp.recorded_goal_yaw_tracking,
        params={"command_name": "object_pose", "std": 0.5},
        weight=0.1,
    )
    success = RewTerm(
        func=mdp.RecordedGoalSuccess,
        params={"command_name": "object_pose", "tolerance": 0.2},
        weight=250.0,
    )
    abnormal_robot = RewTerm(
        func=task_mdps.abnormal_robot_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=-100.0,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=task_mdps.time_out, time_out=True)
    object_out_of_bound = DoneTerm(
        func=task_mdps.out_of_bound,
        params={
            "in_bound_range": {
                "x": (-0.5, 1.5),
                "y": (-2.0, 2.0),
                "z": (0.0, 2.0),
            },
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    abnormal_robot = DoneTerm(
        func=task_mdps.abnormal_robot_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    success = DoneTerm(
        func=mdp.RecordedGoalSuccess,
        params={"command_name": "object_pose", "tolerance": 0.2},
    )


@configclass
class DexsuiteFrankaLeapRotateObjectOmniResetEnvCfg(ManagerBasedRLEnvCfg):
    reset_dataset_dir: str = str(_REPO_ROOT / "dataets" / "rotate_object_bc")
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75),
        lookat=(0.0, 0.0, 0.45),
        origin_type="env",
    )
    scene: SceneCfg = SceneCfg(
        num_envs=4096,
        env_spacing=3,
        replicate_physics=False,
    )
    sim: SimulationCfg = SimulationCfg(gravity=(0.0, 0.0, -9.81))
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()

    def __post_init__(self):
        for link_name in _FINGERTIP_NAMES:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/"
                        f"leap_hand_right/{link_name}"
                    ),
                    filter_prim_paths_expr=contact_filter_prim_paths(),
                    debug_vis=True,
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=task_mdps.fingers_contact_force_b,
            params={
                "contact_sensor_names": [
                    f"{link_name}_object_s" for link_name in _FINGERTIP_NAMES
                ],
                "filter_indices": object_indices(),
            },
            clip=(-20.0, 20.0),
        )
        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
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


@configclass
class DexsuiteFrankaLeapRotateObjectOmniResetEnvCfg_PLAY(
    DexsuiteFrankaLeapRotateObjectOmniResetEnvCfg
):
    """Rotate-object OmniReset evaluation with recorded goals and default scene reset."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.reset_state.params["initial_stage"] = (
            self.curriculum.reset_state.params["num_stages"] - 1
        )
        self.events.reset_scene_to_default = EventTerm(
            func=task_mdps.reset_scene_to_default,
            mode="reset",
            params={},
        )


__all__ = [
    "DexsuiteFrankaLeapRotateObjectOmniResetEnvCfg",
    "DexsuiteFrankaLeapRotateObjectOmniResetEnvCfg_PLAY",
]
