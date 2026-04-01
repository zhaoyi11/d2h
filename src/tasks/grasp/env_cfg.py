# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, dataclass
from pathlib import Path
from re import I

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.markers.config import FRAME_MARKER_CFG

import src.tasks.common.mdps as mdp
import src.tasks.grasp.mdps as task_mdp
from src.assets.franka_leap_hand.leap import LEAP_HAND_CFG

##
# Scene definition
##


def _get_visdex_usd_paths() -> list[str]:
    """Return sorted visdex USD asset paths bundled with this repo."""
    usd_root = Path(__file__).resolve().parents[2] / "assets" / "visdex_objects" / "USD"
    if not usd_root.is_dir():
        raise FileNotFoundError(f"visdex USD asset directory does not exist: {usd_root}")

    usd_paths: list[str] = []
    for object_dir in sorted(path for path in usd_root.iterdir() if path.is_dir()):
        usd_path = object_dir / f"{object_dir.name}.usd"
        if usd_path.is_file():
            usd_paths.append(str(usd_path))

    if not usd_paths:
        raise ValueError(f"No visdex USD assets found in: {usd_root}")
    return usd_paths


@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robots
    robot: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=_get_visdex_usd_paths(),
            random_choice=True, # TODO: check this later. what is the difference of using random choice or not.
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            scale=(0.8, 0.8, 0.8),
        ),
        # 12 cm above the hand
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.03, 0.62), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )
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


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    object_pose = task_mdp.InHandReOrientationCommandCfg(
        asset_name="object",
        random_range=0,
        init_pos_offset=(0.0, 0.0, 0.0),
        update_goal_on_success=False,
        orientation_success_threshold=0.3,
        make_quat_unique=False,
        marker_pos_offset=(-0.2, -0.06, 0.08),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # TODO: check which action manager to use.
    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        alpha=0.95,
        rescale_to_limits=True,
    )

    # joint_pos = mdp.RelativeJointPositionActionCfg(
    #   asset_name="robot",
    #    joint_names=[".*"],
    #    # debug_vis=True,
    #    use_zero_offset=False,
    #    # scale=0.3,
    # )

    # joint_pos = mdp.EMACumulativeRelativeJointPositionActionCfg(
    # asset_name="robot",
    # joint_names=[".*"],
    # alpha=0.95,
    # )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class KinematicObsGroupCfg(ObsGroup):
        """Observations with full-kinematic state information.

        This does not include acceleration or force information.
        """

        # observation terms (order preserved)
        # -- robot terms
        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized, noise=Gnoise(std=0.005)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, noise=Gnoise(std=0.01))

        # fingertip pose
        fingertip_pose = ObsTerm(
            func=mdp.body_state_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=".*fingertip.*"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # fingertip contact (net force)
        fingertip_contact_force_b = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={
                "contact_sensor_names": [
                    "thumb_tip_object_s",
                    "index_tip_object_s",
                    "middle_tip_object_s",
                    "ring_tip_object_s",
                ],
            },
        )

        # -- object terms
        object_pos = ObsTerm(
            func=mdp.root_pos_w,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w,
            params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False},
        )
        object_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )

        # -- action terms
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoVelocityKinematicObsGroupCfg(KinematicObsGroupCfg):
        """Observations with partial kinematic state information.

        In contrast to the full-kinematic state group, this group does not include velocity information
        about the robot joints and the object root frame. This is useful for tasks where velocity information
        is not available or has a lot of noise.
        """

        def __post_init__(self):
            # call parent post init
            super().__post_init__()
            # set unused terms to None
            self.joint_vel = None
            self.object_lin_vel = None
            self.object_ang_vel = None

    @configclass
    class PerceptionObsCfg(ObsGroup):
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),  # clamp between -2 m to 2 m
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    # observation groups
    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
            "damping_distribution_params": (0.75, 1.5),  # default: 0.1
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.4, 1.6),
            "operation": "scale",
        },
    )

    randomize_hand_object_default_pose = EventTerm(
        func=task_mdp.randomize_hand_object_default_pose,
        mode="startup",
        params={
            "base_asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "object_asset_cfg": SceneEntityCfg("object"),
            "roll_range": (-torch.pi, torch.pi),
            "pitch_range": (-torch.pi, torch.pi),
            "yaw_range": (-torch.pi, torch.pi),
        },
    )

    # reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], # TODO: disable this for now. later can change this when generating grasp data.
            "pose_range": {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                #  "z": [-0.01, 0.01]
                "z": [0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )

    reset_robot_joints = EventTerm(
        func=task_mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            # "position_range": {".*": [0.2, 0.2]},
            "position_range": {".*": [0.0, 0.0]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )

    # reset gravity to zero
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )

    # variable_gravity = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="interval",
    #     interval_range_s=(0.8, 1.2),
    #     params={
    #         "gravity_distribution_params": ([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]),
    #         "operation": "abs",
    #     },
    # )

@configclass
class GraspGenEventCfg(EventCfg):
    """Configuration for randomization for grasp generation."""

    # startup
    # -- object
    save_grasp_data = EventTerm(
        func=task_mdp.collect_stable_grasp_states,
        mode="interval",
        interval_range_s=(1.5, 1.5),
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "object_asset_cfg": SceneEntityCfg("object"),
            "cache_path": "/home/yizha/yi/D2H/grasp_data",
            "max_cached_grasp_size": 100,
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # -- task
    # track_pos_l2 = RewTerm(
    #     func=task_mdp.track_pos_l2,
    #     weight=-1.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "command_name": "object_pose",
    #         "max_pos_error": 3.0,
    #         "use_gravity_gate": True,
    #     },
    # )
    # track_orientation_inv_l2 = RewTerm(
    #     func=task_mdp.track_orientation_inv_l2,
    #     weight=1.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "rot_eps": 0.1,
    #         "command_name": "object_pose",
    #         "use_gravity_gate": True,
    #     },
    # )

    fingertip_obj_dist = RewTerm(
        func=task_mdp.neg_fingertip_object_distance,
        weight=3.0,
    )

    # TODO: add contact ralated info later.
    fingertip_contact = RewTerm(
        func=task_mdp.fingertip_object_contacts,
        weight=3,
        params={
            "contact_sensor_names": [
                "thumb_tip_object_s",
                "index_tip_object_s",
                "middle_tip_object_s",
                "ring_tip_object_s",
            ],
        },
    )

    # success_bonus = RewTerm(
    #     func=task_mdp.success_bonus,
    #     weight=5.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    # )

    # penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    joint_pos_default_l2 = RewTerm(func=task_mdp.joint_pos_default_l2, weight=-1e-3)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # object_away_penalty = RewTerm(
    #     func=task_mdp.object_away_from_robot,
    #     weight=-1,
    #     params={"threshold": 0.3},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # max_consecutive_success = DoneTerm(
    #     func=task_mdp.max_consecutive_success,
    #     params={"num_success": 6, "command_name": "object_pose"},
    # )

    object_out_of_reach = DoneTerm(
        func=task_mdp.object_away_from_robot, params={"threshold": 0.3}
    )


##
# Environment configuration
##


@configclass
class InHandObjectEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the in hand reorientation environment."""

    # Scene settings
    scene: InHandObjectSceneCfg = InHandObjectSceneCfg(
        num_envs=8192, env_spacing=0.6, replicate_physics=False
    )
    # # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        gravity=(0.0, 0.0, -9.81),
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: task_mdp.CurriculumCfg | None = task_mdp.CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 25 Hz
        self.episode_length_s = 3  # 10 seconds
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)
        if self.curriculum is not None:
            self.curriculum.adr.params["rot_tol"] = (
                self.commands.object_pose.orientation_success_threshold
            )


##
# Pre-defined configs
##
# from src.assets.leap_hand.leap import LEAP_HAND_CFG

_BASE_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)  # face +y axis


@configclass
class LeapObjectEnvCfg(InHandObjectEnvCfg):
    object_urdf_path: str | None = None
    grasp_path: str | None = None
    object_scale_override: float | None = None

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to leap hand
        self.scene.robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # attach transform sensors to fingertip links for contact-based rewards/observations
        self.scene.fingertip_transforms = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/thumb_fingertip",
                    offset=OffsetCfg(pos=(0.0, -0.045, -0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fingertip",
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fingertip_2",
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/fingertip_3",
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
            ],
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/FrameTransformer",
                markers={
                    "frame": FRAME_MARKER_CFG.markers["frame"].replace(
                        scale=(0.05, 0.05, 0.05)
                    ),
                    "connecting_line": FRAME_MARKER_CFG.markers[
                        "connecting_line"
                    ].replace(radius=0.0005),
                },
            ),
        )

        # TODO: change the urdf obj, the current urdf file can't be filtered properly.
        # attach contact sensors to fingertip links for contact-based rewards/observations
        # Note: use net force here, all forces are considered.
        fingertip_prim_paths = {
            "thumb_tip_object_s": "{ENV_REGEX_NS}/Robot/thumb_fingertip",
            "index_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip",
            "middle_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip_2",
            "ring_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip_3",
        }
        for sensor_name, prim_path in fingertip_prim_paths.items():
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg( 
                    prim_path=prim_path,
                    filter_prim_paths_expr=[
                        "{ENV_REGEX_NS}/Object/baseLink",
                    ],
                    update_period=0.0,
                    history_length=6,
                    debug_vis=False,
                ),
            )



@configclass
class LeapObjectCollectGraspEnvCfg(LeapObjectEnvCfg):
    """ Save grasp data to a file. """
    events: GraspGenEventCfg = GraspGenEventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 4096
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        self.events.save_grasp_data.params["cache_path"] = "grasp_data_visdex"
        self.events.save_grasp_data.params["max_cached_grasp_size"] = 1000



@configclass
class LeapObjectReplayEnvCfg(LeapObjectEnvCfg):
    """ Replay grasp data from a checkpoint. """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 2048
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # enable gravity
        # self.events.reset_gravity.params["gravity_distribution_params"] = (
        #     [0.0, 0.0, -9.81],
        #     [0.0, 0.0, -9.81],
        # )
