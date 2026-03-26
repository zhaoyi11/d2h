# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, dataclass
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
from src.assets.franka_leap_hand.leap import LEAP_HAND_CFG
# from src.tasks.reorient.utils.grasp_init import GraspInitData, load_grasp_init


##
# Scene definition
##


@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robots
    robot: ArticulationCfg = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                CuboidCfg(
                    size=(0.05, 0.1, 0.1),
                    physics_material=RigidBodyMaterialCfg(static_friction=0.5),
                ),
                SphereCfg(
                    radius=0.04,
                    physics_material=RigidBodyMaterialCfg(static_friction=0.5),
                ),
                CapsuleCfg(
                    radius=0.04,
                    height=0.025,
                    physics_material=RigidBodyMaterialCfg(static_friction=0.5),
                ),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        # 12 cm above the hand
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.03, 0.62), rot=(1.0, 0.0, 0.0, 0.0)),
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
        random_range=0.,
        init_pos_offset=(0.0, 0.0, 0.0),
        update_goal_on_success=False,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        marker_pos_offset=(-0.2, -0.06, 0.08),
        debug_vis=True,
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
       
        # # fingertip contact (net force)
        # fingertip_contact_force_b = ObsTerm(
        #     func=mdp.fingers_contact_force_b,
        #     params={
        #         "contact_sensor_names": [
        #             "thumb_tip_object_s",
        #             "index_tip_object_s",
        #             "middle_tip_object_s",
        #             "ring_tip_object_s",
        #         ],
        #     },
        # )

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

        # -- command terms
        goal_pose = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "make_quat_unique": False,
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

    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        interval_range_s=(1.0, 1.0),
        params={
            "gravity_distribution_params": ([0.0, 0.0, -9.81], [0.0, 0.0, -9.81]),
            "operation": "abs",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # track_pos_l2 = RewTerm(
    #     func=task_mdp.track_pos_l2,
    #     weight=-1.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "command_name": "object_pose",

    #     },
    # )
    # # -- task
    # track_orientation_inv_l2 = RewTerm(
    #     func=task_mdp.track_orientation_inv_l2,
    #     weight=1.0,
    #     params={
    #         "object_cfg": SceneEntityCfg("object"),
    #         "rot_eps": 0.1,
    #         "command_name": "object_pose",
    #     },
    # )
    
    # fingertip_object_distance = RewTerm(
    #     func=mdp.fingertip_object_distance,
    #     weight=1.0,
    # )

    fingertip_obj_dist = RewTerm(
        func=task_mdp.neg_fingertip_object_distance,
        weight=1.0,
    )
    
    # # TODO: add contact ralated info later.
    # fingertip_contact = RewTerm(
    #     func=mdp.fingertip_object_contacts,
    #     weight=1,
    #     params={
    #         "contact_sensor_names": [
    #             "thumb_tip_object_s",
    #             "index_tip_object_s",
    #             "middle_tip_object_s",
    #             "ring_tip_object_s",
    #         ],
    #     },
    # )

    # success_bonus = RewTerm(
    #     func=task_mdp.success_bonus,
    #     weight=250.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    # )

    # penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    
    object_away_penalty = RewTerm(
        func=mdp.is_terminated_term,
        weight=-2.0,
        params={"term_keys": "object_out_of_reach"},
    )


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
        gravity=(0.0, 0.0, 0.0),
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

    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    commands: CommandsCfg = CommandsCfg() # TODO: check the order, this previously after the action, before reward.
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4 # 25 Hz
        self.episode_length_s = 3 # 10 seconds
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)
        # TODO: add curriculum settings later.
        # if self.curriculum is not None:
        #     self.curriculum.adr.params["rot_tol"] = (
        #         self.commands.object_pose.orientation_success_threshold
        #     )


##
# Pre-defined configs
##
# from src.assets.leap_hand.leap import LEAP_HAND_CFG

# TODO: move this to config (initial rotation of the wrist)
# Fixed rotation to apply to initial wrist/object poses.
# Quaternion format is (w, x, y, z).

#_BASE_ROT_WXYZ = (0.0, 0.7071067811865476, 0.0, 0.7071067811865476)  # face up
#_BASE_ROT_WXYZ = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)  # face downward
_BASE_ROT_WXYZ = (1.0, 0.0, 0.0, 0.0)  # face +y axis

def _quat_mul_wxyz(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    """Quaternion multiply (wxyz): q = q1 * q2."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _rotate_pos_by_quat_wxyz(
    q: tuple[float, float, float, float], pos: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate a position vector by a unit quaternion (w, x, y, z).

    Uses the formula: p' = q * p * q_inv, implemented as:
        p' = p + 2 * cross(q_xyz, cross(q_xyz, p) + w * p)
    """
    w, qx, qy, qz = q
    px, py, pz = pos
    # t = 2 * cross(q_xyz, p)
    tx = 2.0 * (qy * pz - qz * py)
    ty = 2.0 * (qz * px - qx * pz)
    tz = 2.0 * (qx * py - qy * px)
    return (
        px + w * tx + (qy * tz - qz * ty),
        py + w * ty + (qz * tx - qx * tz),
        pz + w * tz + (qx * ty - qy * tx),
    )

# from isaaclab_assets.robots import ALLEGRO_HAND_CFG

@configclass
class LeapObjectEnvCfg2(InHandObjectEnvCfg):
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
            debug_vis=True,
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
        # fingertip_prim_paths = {
        #     "thumb_tip_object_s": "{ENV_REGEX_NS}/Robot/thumb_fingertip",
        #     "index_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip",
        #     "middle_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip_2",
        #     "ring_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip_3",
        # }
        # for sensor_name, prim_path in fingertip_prim_paths.items():
        #     setattr(
        #         self.scene,
        #         sensor_name,
        #         ContactSensorCfg(
        #             prim_path=prim_path,
        #             filter_prim_paths_expr=[
        #                 "{ENV_REGEX_NS}/Object"
        #             ],  # TODO: check this, can't filter the object properly now.
        #             update_period=0.0,
        #             history_length=6,
        #             debug_vis=True,
        #         ),
        #     )

        # # Initial robot and object state
        # if self.grasp_path is not None and self.object_urdf_path is not None:
        #     grasp_init = load_grasp_init(
        #         self.grasp_path,
        #         self.object_urdf_path,
        #         self.object_scale_override,
        #         base_pos=(0, 0, 0),
        #         base_rot=_BASE_ROT_WXYZ,
        #     )

        #     self.scene.robot = self.scene.robot.replace(
        #         init_state=ArticulationCfg.InitialStateCfg(
        #             pos=(0, 0, 0),
        #             rot=_BASE_ROT_WXYZ,
        #         ),
        #     )

        #     self._apply_grasp_events(grasp_init)

    # # TODO: check this later.
    # def _apply_grasp_events(self, grasp_init: GraspInitData):
    #     """Configure scene and reset events from precomputed grasp data."""
    #     object_asset_path = (
    #         grasp_init.object_asset_path
    #         if grasp_init.object_asset_path is not None
    #         else self.scene.object.spawn.asset_path
    #     )

    #     object_pos_rot = _rotate_pos_by_quat_wxyz(_BASE_ROT_WXYZ, grasp_init.object_pos)
    #     object_quat_rot = _quat_mul_wxyz(_BASE_ROT_WXYZ, grasp_init.object_quat)

    #     self.scene.object = self.scene.object.replace(
    #         spawn=self.scene.object.spawn.replace(
    #             asset_path=object_asset_path,
    #             scale=(
    #                 # 0.1,
    #                 # 0.1,
    #                 # 0.1,  # TODO: !!!!! change this
    #                 grasp_init.object_scale,
    #                 grasp_init.object_scale,
    #                 grasp_init.object_scale,
    #             ),
    #         ),
    #         init_state=RigidObjectCfg.InitialStateCfg(
    #             pos=object_pos_rot, rot=object_quat_rot
    #         ),
    #     )

    #     self.events.reset_object = EventTerm(
    #         func=mdp.reset_root_state_from_pose,
    #         mode="reset",
    #         params={
    #             "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #             "pose": (*object_pos_rot, *object_quat_rot),
    #             "velocity": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    #         },
    #     )

    #     # set robot joint positions to the precomputed grasp joint positions (conversion between MuJoCo order and IsaacGym order)
    #     self.events.reset_robot_joints = EventTerm(
    #         func=mdp.reset_joints_to_fixed,
    #         mode="reset",
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #             "joint_pos": grasp_init.robot_joint_pos,
    #         },
    #     )

    #     self.events.reset_robot_root = EventTerm(
    #         func=mdp.reset_root_state_from_pose,
    #         mode="reset",
    #         params={
    #             "asset_cfg": SceneEntityCfg("robot"),
    #             "pose": (0.0, 0.0, 0.0, *_BASE_ROT_WXYZ),
    #             "velocity": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    #         },
    #     )


@configclass
class LeapObjectEnvCfg_PLAY(LeapObjectEnvCfg2):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None


##
# Environment configuration with no velocity observations.
##


@configclass
class LeapObjectNoVelObsEnvCfg(LeapObjectEnvCfg2):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch observation group to no velocity group
        self.observations.policy = ObservationsCfg.NoVelocityKinematicObsGroupCfg()


@configclass
class LeapObjectNoVelObsEnvCfg_PLAY(LeapObjectNoVelObsEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
