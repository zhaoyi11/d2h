# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, dataclass

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.env.tasks.anygrasp.mdps as mdp


USD_PALM_LOWER_OFFSET = np.array([-0.1, 0.038, 0.098])
USD_PALM_LOWER_QUAT = np.array([0.0, -0.7071, 0.0, -0.7071])

# Fixed rotation to apply to initial wrist/object poses.
# Quaternion format is (w, x, y, z). Using world +Y axis, +90deg.
_Q_Y_POS_90_WXYZ = (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)


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


def _rotate_pos_y_pos_90(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """Rotate a position by +90deg about world +Y: (x,y,z) -> (z, y, -x)."""
    x, y, z = pos
    return (z, y, -x)


@dataclass
class GraspInitData:
    """Precomputed grasp initialization data."""

    object_pos: tuple[float, float, float]
    object_quat: tuple[float, float, float, float]
    object_scale: float
    object_asset_path: str
    robot_joint_pos: np.ndarray


def np_quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def np_quaternion_inverse(q: np.ndarray) -> np.ndarray:
    """Compute the inverse of a quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def np_quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def transform_object_to_robot_frame(
    mj_palm_pos: np.ndarray,
    mj_palm_quat: np.ndarray,
    object_pos: np.ndarray,
    object_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform object pose so base is identity while preserving palm-to-object relationship."""
    R_palm_mj = np_quaternion_to_matrix(mj_palm_quat)
    R_palm_mj_inv = R_palm_mj.T
    obj_rel_pos_mj = R_palm_mj_inv @ (object_pos - mj_palm_pos)

    palm_quat_inv = np_quaternion_inverse(mj_palm_quat)
    obj_rel_quat = np_quaternion_multiply(palm_quat_inv, object_quat)

    new_palm_pos = USD_PALM_LOWER_OFFSET.copy()
    new_palm_quat = USD_PALM_LOWER_QUAT.copy()
    new_palm_rot = np_quaternion_to_matrix(new_palm_quat)

    obj_rel_pos_transformed = new_palm_rot @ obj_rel_pos_mj
    new_object_pos = new_palm_pos + obj_rel_pos_transformed
    new_object_quat = np_quaternion_multiply(new_palm_quat, obj_rel_quat)

    new_object_quat = new_object_quat / np.linalg.norm(new_object_quat)
    if new_object_quat[0] < 0:
        new_object_quat = -new_object_quat

    # example
    # pos  array([0.14013682, 0.02385659, 0.09118012])
    # quat array([0.02073345, 0.96638001, 0.24646428, 0.07025066])
    return new_object_pos, new_object_quat


##
# Scene definition
##


@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robots
    robot: ArticulationCfg = MISSING

    # # objects
    # # TODO: compare different rigid_props
    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             kinematic_enabled=False,
    #             disable_gravity=False,
    #             enable_gyroscopic_forces=True,
    #             solver_position_iteration_count=8,
    #             solver_velocity_iteration_count=0,
    #             sleep_threshold=0.005,
    #             stabilization_threshold=0.0025,
    #             max_depenetration_velocity=1000.0,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(density=400.0),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.19, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    # object
    # object: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[
    #             # CuboidCfg(size=(0.08, 0.08, 0.08), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             # original
    #             CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #             ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
    #         ],
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=0,
    #             disable_gravity=False,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.2),  # 200g (~iphone's weight)
    #         # mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
    #     ),
    #     # init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.10, 0.6),
    #     # rot=(1.0, 0.0, 0.0, 0.0)
    #     rot=(0.7071, 0.0, 0.7071, 0.0)
    #     ),
    # )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/yizhao/yi/DexGraspBench/assets/object/DGN_2k/processed_data/core_camera_fb3b5fae94f7b02a3b269928487f8a4c/urdf/coacd.urdf",
            scale=(0.08, 0.08, 0.08),  # todo; fix this to configurable from grasp data.
            activate_contact_sensors=True,
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None, damping=None
                ),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,  # Disable articulation for rigid object
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_rotate_pos_y_pos_90((0.0, -0.10, 0.6)),
            rot=_quat_mul_wxyz(_Q_Y_POS_90_WXYZ, (0.7071, 0.0, 0.7071, 0.0)),
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=1000.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.02, 0.02, 0.02), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    object_pose = mdp.InHandReOrientationCommandCfg(
        asset_name="object",
        # init_pos_offset=(0.0, 0.0, -0.04),
        init_pos_offset=(0.0, 0.0, 0.0),  # TODO: remove z offset, confirm this.
        update_goal_on_success=True,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        marker_pos_offset=(-0.2, -0.06, 0.08),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        alpha=0.95,
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
    record_object_init_state = EventTerm(
        func=mdp.record_object_init_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            # "position_range": {".*": [0.2, 0.2]},
            "position_range": {".*": [0.0, 0.0]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # track_pos_l2 = RewTerm(
    #     func=mdp.track_pos_l2,
    #     weight=-10.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    # )
    track_orientation_inv_l2 = RewTerm(
        func=mdp.track_orientation_inv_l2,
        weight=1.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "rot_eps": 0.1,
            "command_name": "object_pose",
        },
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    )

    # -- penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # fingertip_contact = RewTerm(
    #     func=mdp.fingertip_object_contacts,
    #     weight=0.1,
    #     params={
    #         "contact_sensor_names": [
    #             "thumb_tip_object_s",
    #             "index_tip_object_s",
    #             "middle_tip_object_s",
    #             "ring_tip_object_s",
    #         ],
    #         "threshold": 1e-3,
    #     },
    # )
    object_stability = RewTerm(
        func=mdp.object_stay_close,
        weight=0.01,
    )

    # -- optional penalties (these are disabled by default)
    # object_away_penalty = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=-0.0,
    #     params={"term_keys": "object_out_of_reach"},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success,
        params={"num_success": 50, "command_name": "object_pose"},
    )

    object_out_of_reach = DoneTerm(
        func=mdp.object_away_from_robot, params={"threshold": 0.3}
    )

    # object_out_of_reach = DoneTerm(
    #     func=mdp.object_away_from_goal, params={"threshold": 0.24, "command_name": "object_pose"}
    # )


##
# Environment configuration
##


@configclass
class InHandObjectEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the in hand reorientation environment."""

    grasp_data_path: str | None = None
    object_urdf_path: str | None = None
    object_scale_override: float | None = None
    use_grasp_init: bool = False

    # Scene settings
    scene: InHandObjectSceneCfg = InHandObjectSceneCfg(
        num_envs=8192, env_spacing=0.6, replicate_physics=False
    )
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        gravity=(0.0, 0.0, -9.8),
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
    grasp_init: GraspInitData | None = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)

        if self.grasp_data_path is not None:
            self.use_grasp_init = True

        if self.use_grasp_init:
            if self.grasp_init is None:
                raise ValueError(
                    "use_grasp_init=True requires grasp_init to be precomputed and set on the cfg."
                )
            self._apply_grasp_events(self.grasp_init)

    def _apply_grasp_events(self, grasp_init: GraspInitData):
        """Configure scene and reset events from precomputed grasp data."""
        object_asset_path = (
            grasp_init.object_asset_path
            if grasp_init.object_asset_path is not None
            else self.scene.object.spawn.asset_path
        )

        object_pos_rot = _rotate_pos_y_pos_90(grasp_init.object_pos)
        object_quat_rot = _quat_mul_wxyz(_Q_Y_POS_90_WXYZ, grasp_init.object_quat)

        self.scene.object = self.scene.object.replace(
            spawn=self.scene.object.spawn.replace(
                asset_path=object_asset_path,
                scale=(
                    grasp_init.object_scale,
                    grasp_init.object_scale,
                    grasp_init.object_scale,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=object_pos_rot, rot=object_quat_rot
            ),
        )

        self.events.reset_object = EventTerm(
            func=mdp.reset_root_state_from_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "pose": (*object_pos_rot, *object_quat_rot),
                "velocity": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            },
        )

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_to_fixed,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "joint_pos": grasp_init.robot_joint_pos,
            },
        )

        self.events.reset_robot_root = EventTerm(
            func=mdp.reset_root_state_from_pose,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose": (0.0, 0.0, 0.0, *_Q_Y_POS_90_WXYZ),
                "velocity": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            },
        )


##
# Pre-defined configs
##
from src.assets.leap_hand.leap import LEAP_HAND_CFG


@configclass
class LeapObjectEnvCfg(InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to leap hand
        self.scene.robot = LEAP_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # attach contact sensors to fingertip links for contact-based rewards/observations
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
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )

        if self.use_grasp_init:
            # self.scene.robot = self.scene.robot.replace(
            #     spawn=self.scene.robot.spawn.replace(
            #         articulation_props=self.scene.robot.spawn.articulation_props.replace(
            #             fix_root_link=False,
            #         ),
            #     ),
            # )
            # TODO: check the init state here.
            self.scene.robot = self.scene.robot.replace(
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.0),
                    rot=_Q_Y_POS_90_WXYZ,
                    joint_pos=self.scene.robot.init_state.joint_pos,
                ),
            )
        else:
            # Rotate the default wrist/root initial orientation.
            self.scene.robot = self.scene.robot.replace(
                init_state=self.scene.robot.init_state.replace(
                    rot=_quat_mul_wxyz(
                        _Q_Y_POS_90_WXYZ, self.scene.robot.init_state.rot
                    )
                )
            )
        # enable clone in fabric
        # self.scene.clone_in_fabric = True


@configclass
class LeapObjectEnvCfg_PLAY(LeapObjectEnvCfg):
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
class LeapObjectNoVelObsEnvCfg(LeapObjectEnvCfg):
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
