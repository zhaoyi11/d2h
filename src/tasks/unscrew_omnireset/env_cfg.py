# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG
import src.tasks.common.mdps as task_mdps
import src.tasks.reorient.mdps as reorient_mdp
from src.tasks.unscrew_omnireset.mdps import commands as command_mdp
from src.tasks.unscrew_omnireset.mdps import events as event_mdp
from src.tasks.unscrew_omnireset.mdps import task_mdps as mdp
from src.tasks.unscrew_omnireset.mdps.contact_filters import (
    contact_filter_prim_paths,
    external_indices,
    object_indices,
)
from src.tasks.unscrew_omnireset.mdps.curriculums import CurriculumCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]

_UWLAB_CLOUD_ASSETS_DIR = (
    "https://huggingface.co/datasets/UW-Lab/uwlab-assets/resolve/main"
)
_ASSET_SCALE = 1.5
_INSTALLED_OBJECT_POS = (
    0.4659913182258606,
    0.08350233733654022,
    0.3423238694667816,
)
_INSTALLED_OBJECT_QUAT = (
    0.97566819190979,
    0.00011654444824671373,
    -0.0003023587341886014,
    0.21925236284732819,
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = FRANKA_LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                f"{_UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/"
                "SquareLeg/square_leg.usd"
            ),
            scale=(_ASSET_SCALE, _ASSET_SCALE, _ASSET_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_INSTALLED_OBJECT_POS,
            rot=_INSTALLED_OBJECT_QUAT,
        ),
    )
    receptive_object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                f"{_UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/"
                "SquareTableTop/square_table_top.usd"
            ),
            scale=(_ASSET_SCALE, _ASSET_SCALE, _ASSET_SCALE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                rest_offset=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.271),
            rot=(1.0, 0.0, 0.0, 0.0),
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
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )


_FINGERTIP_CONTACT_SENSOR_NAMES = [
    "thumb_fingertip_object_s",
    "fingertip_object_s",
    "fingertip_2_object_s",
    "fingertip_3_object_s",
]


@configclass
class ObservationsCfg:
    @configclass
    class LowLevelObsCfg(ObsGroup):
        """Current-step 155-D observation for the frozen reorientation policy."""

        joint_pos = ObsTerm(
            func=task_mdps.joint_pos_limit_normalized,
            noise=Gnoise(std=0.005),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        joint_vel = ObsTerm(
            func=task_mdps.joint_vel_rel,
            scale=0.2,
            noise=Gnoise(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        fingertip_pose = ObsTerm(
            func=task_mdps.body_state_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=".*fingertip.*"),
                "base_body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
        contact_mask = ObsTerm(
            func=reorient_mdp.tip_contact_mask_obs,
            params={
                "contact_sensor_names": _FINGERTIP_CONTACT_SENSOR_NAMES,
                "force_threshold": 0.25,
                "filter_indices": object_indices(),
            },
        )
        contact_force_mag = ObsTerm(
            func=reorient_mdp.tip_contact_force_mag_obs,
            params={
                "contact_sensor_names": _FINGERTIP_CONTACT_SENSOR_NAMES,
                "force_threshold": 0.25,
                "filter_indices": object_indices(),
            },
        )
        contact_pose = ObsTerm(
            func=reorient_mdp.tip_contact_pose_flat,
            params={
                "contact_sensor_names": _FINGERTIP_CONTACT_SENSOR_NAMES,
                "force_threshold": 0.25,
                "contact_pose_range_deg": 90.0,
                "filter_indices": object_indices(),
            },
        )
        external_contact_mask = ObsTerm(
            func=reorient_mdp.tip_contact_mask_obs,
            params={
                "contact_sensor_names": _FINGERTIP_CONTACT_SENSOR_NAMES,
                "force_threshold": 0.25,
                "filter_indices": external_indices(),
            },
        )
        external_contact_force_mag = ObsTerm(
            func=reorient_mdp.tip_contact_force_mag_obs,
            params={
                "contact_sensor_names": _FINGERTIP_CONTACT_SENSOR_NAMES,
                "force_threshold": 0.25,
                "filter_indices": external_indices(),
            },
        )
        external_contact_pose = ObsTerm(
            func=reorient_mdp.tip_contact_pose_flat,
            params={
                "contact_sensor_names": _FINGERTIP_CONTACT_SENSOR_NAMES,
                "force_threshold": 0.25,
                "contact_pose_range_deg": 45.0,
                "filter_indices": external_indices(),
            },
        )
        object_pos = ObsTerm(
            func=task_mdps.object_pos_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_quat = ObsTerm(
            func=task_mdps.object_quat_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_lin_vel = ObsTerm(
            func=task_mdps.object_lin_vel_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_ang_vel = ObsTerm(
            func=task_mdps.object_ang_vel_body_b,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        gravity_dir = ObsTerm(
            func=task_mdps.gravity_dir_body_b,
            params={"body_asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        goal_pos_diff = ObsTerm(
            func=task_mdps.goal_pos_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        goal_quat_diff = ObsTerm(
            func=task_mdps.goal_quat_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "make_quat_unique": False,
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        last_action = ObsTerm(
            func=task_mdps.last_action,
            params={"action_name": "hand_action"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    @configclass
    class ResidualObsCfg(ObsGroup):
        """Current-step 55-D arm residual-policy observation."""

        arm_joint_pos = ObsTerm(
            func=task_mdps.joint_pos_limit_normalized,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])
            },
        )
        arm_joint_vel = ObsTerm(
            func=task_mdps.joint_vel_rel,
            scale=0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"])
            },
        )
        hand_base_state = ObsTerm(
            func=task_mdps.body_state_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        receptive_pose = ObsTerm(
            func=mdp.receptive_pose_b,
            params={"receptive_cfg": SceneEntityCfg("receptive_object")},
        )
        object_pose_receptive = ObsTerm(
            func=mdp.object_pose_receptive,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "receptive_cfg": SceneEntityCfg("receptive_object"),
            },
        )
        trajectory_command = ObsTerm(
            func=task_mdps.generated_commands,
            params={"command_name": "object_pose"},
        )
        last_arm_action = ObsTerm(
            func=task_mdps.last_action,
            params={"action_name": "arm_action"},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    low_level: LowLevelObsCfg = LowLevelObsCfg()
    residual: ResidualObsCfg = ResidualObsCfg()


@configclass
class CommandsCfg:
    object_pose = command_mdp.UnscrewOmniResetTrajectoryCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(20.0, 20.0),
        debug_vis=False,
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
            "mass_distribution_params": [0.05, 0.2],
            "operation": "abs",
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
    reset_from_dataset: EventTerm | None = EventTerm(
        func=event_mdp.ResetSceneFromInstantDexterity,
        mode="reset",
        params={},
    )
    reset_scene_to_default: EventTerm | None = None


@configclass
class RewardsCfg:
    action_l2 = RewTerm(func=task_mdps.action_l2_clamped, weight=-0.001)
    action_rate_l2 = RewTerm(
        func=task_mdps.action_rate_l2_clamped,
        weight=-0.001,
    )
    fingers_to_object = RewTerm(
        func=task_mdps.object_ee_distance,
        params={"std": 0.4},
        weight=1.0,
    )
    good_finger_contact = RewTerm(
        func=task_mdps.contacts,
        params={"threshold": 1.0},
        weight=0.5,
    )
    # trajectory_position = RewTerm(
    #     func=task_mdps.position_command_error_tanh,
    #     params={
    #         "std": 0.10,
    #         "command_name": "object_pose",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "align_asset_cfg": SceneEntityCfg("object"),
    #     },
    #     weight=1.0,
    # )
    # trajectory_orientation = RewTerm(
    #     func=task_mdps.orientation_command_error_tanh,
    #     params={
    #         "std": 0.50,
    #         "command_name": "object_pose",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "align_asset_cfg": SceneEntityCfg("object"),
    #     },
    #     weight=1.0,
    # )
    
    clearance_progress = RewTerm(
        func=mdp.DenseUnscrewClearance,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "start_clearance": -0.040,
            "target_clearance": 0.030,
        },
        weight=1.0,
    )
    success = RewTerm(
        func=mdp.UnscrewSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "clearance_margin": 0.030,
            "force_threshold": 1.0,
        },
        weight=250.0,
    )
    abnormal_robot = RewTerm(
        func=task_mdps.abnormal_robot_state,
        weight=-100.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
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
    abnormal_robot = DoneTerm(
        func=task_mdps.abnormal_robot_state,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    success = DoneTerm(
        func=mdp.UnscrewSuccess,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "receptive_cfg": SceneEntityCfg("receptive_object"),
            "clearance_margin": 0.030,
            "force_threshold": 1.0,
        },
    )


@configclass
class DexsuiteFrankaLeapUnscrewOmniResetEnvCfg(ManagerBasedRLEnvCfg):
    reset_dataset_dir: str = str(_REPO_ROOT / "dataets" / "unscrew_bc")
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
        finger_tip_body_list = [
            "thumb_fingertip",
            "fingertip",
            "fingertip_2",
            "fingertip_3",
        ]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/"
                        f"leap_hand_right/{link_name}"
                    ),
                    filter_prim_paths_expr=contact_filter_prim_paths(),
                    debug_vis=False,
                ),
            )
        self.scene.fingertip_transforms = FrameTransformerCfg(
            prim_path=(
                "{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/base"
            ),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/"
                        "leap_hand_right/thumb_fingertip"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.045, -0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/"
                        "leap_hand_right/fingertip"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/"
                        "leap_hand_right/fingertip_2"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Franka_LeapHand/"
                        "leap_hand_right/fingertip_3"
                    ),
                    offset=OffsetCfg(pos=(0.0, -0.03, 0.015)),
                ),
            ],
            debug_vis=False,
        )
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            body_names=[".*fingertip.*"],
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
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31


@configclass
class DexsuiteFrankaLeapUnscrewOmniResetEnvCfg_PLAY(
    DexsuiteFrankaLeapUnscrewOmniResetEnvCfg
):
    """Unscrew OmniReset evaluation environment using the hardest reset."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_from_dataset = None
        self.events.reset_scene_to_default = EventTerm(
            func=task_mdps.reset_scene_to_default,
            mode="reset",
            params={},
        )
        self.curriculum = None


__all__ = [
    "DexsuiteFrankaLeapUnscrewOmniResetEnvCfg",
    "DexsuiteFrankaLeapUnscrewOmniResetEnvCfg_PLAY",
]
