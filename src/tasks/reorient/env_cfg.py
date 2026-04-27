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
import src.tasks.reorient.mdps as task_mdp
from src.tasks.reorient.curriculum import CurriculumCfg
from src.assets.franka_leap_hand.leap import LEAP_HAND_CFG

##
# Scene definition
##
UWLAB_CLOUD_ASSETS_DIR = "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main"

OBJ_CONTACT_SENSOR_FILTER_PRIM_PATHS_EXPR = "{ENV_REGEX_NS}/Object/baseLink"

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

    # # object
    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
    #         scale=(1.5, 1.5, 1.5),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=4,
    #             solver_velocity_iteration_count=0,
    #             disable_gravity=False,
    #             kinematic_enabled=False,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
    #     ),
    #     # 12 cm above the hand 
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.08, 0.12), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # all visdex objects
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=_get_visdex_usd_paths(),
            # random_choice=True,
            random_choice=False,
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
            pos=(0.0, -0.08, 0.12), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
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
        random_range=(0.3 * torch.pi, 0.5 * torch.pi),  # will increase to (0.3\pi, 0.5\pi) with curriculum
        init_pos_offset=(0.0, 0.0, 0.0),
        resample_on="success",  # can be ["success", "time"]
        hold_steps_on_success=20,  # timestep count to update goal once success
        resampling_time_range=(3.0, 5.0),  # if resample based on time, the sample time range.
        orientation_success_threshold=0.3,  # reduce to 0.2 with curriculum
        make_quat_unique=False,
        marker_pos_offset=(-0.2, -0.06, 0.08),
        debug_vis=True,
        use_position_success=True,  # also consider position success
        position_success_threshold=0.1,  # reduce to 0.05 with curriculum
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # TODO: check which action manager to use.
    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        # alpha=0.95
        alpha=0.5,  # TODO: can I add this to curriculum?
        rescale_to_limits=True,
    )


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

        # fingertip contact — raw 3D forces in robot base frame (12 dims)
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

        # binary contact mask per fingertip (4 dims)
        contact_mask = ObsTerm(
            func=task_mdp.tip_contact_mask_obs,
            params={
                "contact_sensor_names": [
                    "thumb_tip_object_s",
                    "index_tip_object_s",
                    "middle_tip_object_s",
                    "ring_tip_object_s",
                ],
                "force_threshold": 0.25,
            },
        )

        # per-fingertip contact force magnitude (4 dims)
        contact_force_mag = ObsTerm(
            func=task_mdp.tip_contact_force_mag_obs,
            params={
                "contact_sensor_names": [
                    "thumb_tip_object_s",
                    "index_tip_object_s",
                    "middle_tip_object_s",
                    "ring_tip_object_s",
                ],
                "force_threshold": 0.25,
            },
        )

        # per-fingertip contact pose (theta, phi) in fingertip frame, flattened (8 dims)
        # TOOD: check this
        contact_pose = ObsTerm(
            func=task_mdp.tip_contact_pose_flat,
            params={
                "contact_sensor_names": [
                    "thumb_tip_object_s",
                    "index_tip_object_s",
                    "middle_tip_object_s",
                    "ring_tip_object_s",
                ],
                "force_threshold": 0.25,
                "contact_pose_range_deg": 45.0,
            },
        )

        # -- object terms
        object_pos = ObsTerm(
            func=mdp.object_pos_b,
            noise=Gnoise(std=0.002),
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_quat = ObsTerm(
            func=mdp.object_quat_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "object_cfg": SceneEntityCfg("object"),
            },
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

        # -- gravity in robot frame (needed for orientation-dependent grasp strategy)
        # TODO: check this as well
        gravity_dir = ObsTerm(
            func=task_mdp.gravity_dir_b,
            params={"base_asset_cfg": SceneEntityCfg("robot")},
        )

        # -- command terms
        # goal_pos_diff = ObsTerm(
        #     func=mdp.goal_pos_diff,
        #     params={
        #         "asset_cfg": SceneEntityCfg("object"),
        #         "command_name": "object_pose",
        #         "robot_cfg": SceneEntityCfg("robot"),
        #     },
        # )
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "make_quat_unique": False,
                "robot_cfg": SceneEntityCfg("robot"),
            },
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
            "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0, thus is range is [0.9, 9]
            "damping_distribution_params": (0.75, 1.5),  # default: 0.1, thus the range is [0.075, 0.15]
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
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "object_asset_cfg": SceneEntityCfg("object"),
            # The range will be changed with curriculum
            "roll_range": (0.0, 0.0),
            "pitch_range": (0.0, 0.0),
            "yaw_range": (0.0, 0.0),
        },
    )

    # reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # the pose range with be changed with curriculum
            "pose_range": {
                "x": [-0.0, 0.0],
                "y": [-0.0, 0.0],
                "z": [-0.0, 0.0],
                "roll": [0.0, 0.0],
                "pitch": [0.0, 0.0],
                "yaw": [0.0, 0.0],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )

    reset_robot_joints = EventTerm(
        func=task_mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            "position_range": {".*": [0.2, 0.2]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )

    # set gravity to zero, then handle by curriculum later.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            # the gravity will be changed with curriculum
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )

    # Fires every step per-env; must be declared AFTER variable_gravity so the
    # event manager applies it after gravity has been updated.
    gravity_compensation_assist = EventTerm(
        func=task_mdp.apply_gravity_compensation_assist,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "contact_threshold": 1.0,
            "decay_ratio": 0.9,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    track_position = RewTerm(
        func=task_mdp.track_position,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "object_pose",
            "pos_scale": 2.0,
            "pos_temp": 0.5,
            "need_contact": True,
        },
    )

    track_orientation = RewTerm(
        func=task_mdp.track_orientation,
        weight=5.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "command_name": "object_pose",
            "rot_scale": 5.0,
            "rot_temp": 1.0,
            "need_contact": True,
        },
    )

    # track_orientation = RewTerm(
    #     func=task_mdp.track_orientation_exp,
    #     weight=5.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "command_name": "object_pose",
    #         "rot_scale": 2.0,
    #         "rot_temp": 1.0,
    #         "need_contact": True,
    #     }
    # )

    # track_orientation = RewTerm(
    #     func=task_mdp.track_orientation_inv_l2,
    #     weight=1.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "rot_eps": 0.1,
    #         "command_name": "object_pose",
    #         "need_contact": True
    #     }
    # )

    fingertip_obj_dist = RewTerm(
        func=task_mdp.neg_fingertip_object_distance,
        weight=0.5,
    )

    # todo: check good contact, the value doesn't look very good for now.
    good_contact = RewTerm(
        func=task_mdp.good_contact_reward,
        weight=1.0,
        params={
            "contact_sensor_names": [
                "thumb_tip_object_s",
                "index_tip_object_s",
                "middle_tip_object_s",
                "ring_tip_object_s",
            ],
            "force_threshold": 0.25,
            "contact_pose_range_deg": 45.0,
            "contact_scale": 1.0,
            "contact_temp": 1.0,
        },
    )

    success = RewTerm(
        func=task_mdp.success_bonus,
        weight=20.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    )

    # penalties
    energy = RewTerm(func=task_mdp.joint_power, weight=-1e-5, params={"asset_cfg": SceneEntityCfg("robot")})
    # todo; check this
    # joint_pos_default_l2 = RewTerm(func=task_mdp.joint_pos_default_l2, weight=-1e-3)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # abnormal robot
    abnormal_robot = RewTerm(func=mdp.abnormal_robot_state, weight=-10.0)

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
    #     params={"num_success": 100, "command_name": "object_pose"},
    # )

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)

    object_out_of_reach = DoneTerm(
        func=task_mdp.object_away_from_robot, params={"threshold": 0.2}
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
            gpu_max_rigid_contact_count=2**23,
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
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 25 Hz
        self.episode_length_s = 15  # 15 seconds
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)
        if self.curriculum is not None:
            self.curriculum.adr.params["num_success"] = 3


@configclass
class LeapObjectEnvCfg(InHandObjectEnvCfg):

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
                        OBJ_CONTACT_SENSOR_FILTER_PRIM_PATHS_EXPR
                    ],
                    update_period=0.0,
                    history_length=6,
                    debug_vis=False,
                ),
            )
