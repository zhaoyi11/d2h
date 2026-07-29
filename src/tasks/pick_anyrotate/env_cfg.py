# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import CapsuleCfg, ConeCfg, CuboidCfg, RigidBodyMaterialCfg, SphereCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg

import src.tasks.pick_anyrotate.mdps as mdp
import src.tasks.reorient.mdps as task_mdps
from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG


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
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""

    # robot
    robot = FRANKA_LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        # disable self collisions
        spawn=FRANKA_LEAP_HAND_CFG.spawn.replace(
            articulation_props=FRANKA_LEAP_HAND_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=False,
            ),
        ),
    )  

    # object
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=_get_visdex_usd_paths(),
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
            pos=[0.55, 0.1, 0.34],
            rot=[1.0, 0.0, 0.0, 0.0],
        ), 
    )


    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # trick: we let visualizer's color to show the table with success coloring
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)
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


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=True,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.7),
            pos_y=(-0.25, 0.25),
            pos_z=(0.35, 0.75),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
    )


@configclass
class HrlCommandsCfg:
    """Pick, lift, and six-target trajectory command."""

    object_pose = mdp.PickAnyRotateTrajectoryObjectAndHandBasePoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
        position_only=False,
        success_vis_asset_name="table",
        enable_drop_recovery=True,
        recovery_settle_speed=0.05,
        recovery_settle_steps=5,
        drop_object_hand_distance=0.12,
        recovery_arm_after_stage=2,
        capture_goal_after_settle=True,
        grasp_stall_steps=60,
        hand_open_until_stage=0,
        hand_base_hold_until_stage=1,
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
        target_object_pose_b = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        """Observations for proprioception group."""

        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            # good behaving number for position in m, velocity in m/s, rad/s,
            # and quaternion are unlikely to exceed -2 to 2 range
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

    @configclass
    class LowLevelObsCfg(ObsGroup):
        """155-D hand-frame observation used by the frozen anyreorient policy."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_limit_normalized,
            noise=Gnoise(std=0.005),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.2,
            noise=Gnoise(std=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["a_.*"])},
        )
        fingertip_pose = ObsTerm(
            func=mdp.body_state_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names=".*fingertip.*"),
                "base_body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
            },
        )
        contact_mask = ObsTerm(
            func=task_mdps.tip_contact_mask_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
            },
        )
        contact_force_mag = ObsTerm(
            func=task_mdps.tip_contact_force_mag_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
            },
        )
        contact_pose = ObsTerm(
            func=task_mdps.tip_contact_pose_flat,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "contact_pose_range_deg": 90.0,
            },
        )
        external_contact_mask = ObsTerm(
            func=task_mdps.tip_contact_mask_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "filter_indices": [1],
            },
        )
        external_contact_force_mag = ObsTerm(
            func=task_mdps.tip_contact_force_mag_obs,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "filter_indices": [1],
            },
        )
        external_contact_pose = ObsTerm(
            func=task_mdps.tip_contact_pose_flat,
            params={
                "contact_sensor_names": [
                    "thumb_fingertip_object_s",
                    "fingertip_object_s",
                    "fingertip_2_object_s",
                    "fingertip_3_object_s",
                ],
                "force_threshold": 0.25,
                "contact_pose_range_deg": 45.0,
                "filter_indices": [1],
            },
        )
        object_pos = ObsTerm(
            func=mdp.object_pos_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_quat = ObsTerm(
            func=mdp.object_quat_body_b,
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_lin_vel = ObsTerm(
            func=mdp.object_lin_vel_body_b,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_ang_vel = ObsTerm(
            func=mdp.object_ang_vel_body_b,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        gravity_dir = ObsTerm(
            func=mdp.gravity_dir_body_b,
            params={"body_asset_cfg": SceneEntityCfg("robot", body_names="base")},
        )
        goal_pos_diff = ObsTerm(
            func=mdp.goal_pos_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff_body_b,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "command_name": "object_pose",
                "make_quat_unique": False,
                "body_asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "command_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        last_action = ObsTerm(func=mdp.last_action, params={"action_name": "hand_action"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 1

    # # observation groups
    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()
    low_level: LowLevelObsCfg | None = None


@configclass
class EventCfg:
    """Configuration for randomization."""

    # # -- pre-startup
    # randomize_object_scale = EventTerm(
    #     func=mdp.randomize_rigid_body_scale,
    #     mode="prestartup",
    #     params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},
    # )
     
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
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
        func=mdp.randomize_rigid_body_material,
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
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.5, 2.0],
            "damping_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )

    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.2, 0.2],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.4],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    # Note (Octi): This is a deliberate trick in Remake to accelerate learning.
    # By scheduling gravity as a curriculum — starting with no gravity (easy)
    # and gradually introducing full gravity (hard) — the agent learns more smoothly.
    # This removes the need for a special "Lift" reward (often required to push the
    # agent to counter gravity), which has bonus effect of simplifying reward composition overall.
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )


@configclass
class HrlEventCfg(EventCfg):
    """Table-top reset and reduced gravity for the frozen-hand rollout."""

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03], "yaw": [0.0, 0.0]},
            "velocity_range": {"x": [0.0, 0.0], "y": [0.0, 0.0], "z": [0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_robot_joints: EventTerm | None = None
    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.81], [0.0, 0.0, -1.81]),
            "operation": "abs",
        },
    )


@configclass
class ActionsCfg:

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.1
    )


@configclass
class HrlActionsCfg:
    """cuRobo arm target plus the frozen policy's 16-DOF hand action."""

    arm_action = mdp.CommandHandBaseCuroboMpcActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="base",
        command_name="object_pose",
        robot_config_file=(
            f"{Path(__file__).resolve().parents[2]}/assets/franka_leap_hand/curobo/franka_leap.yml"
        ),
        obstacle_cuboids={},
    )
    hand_action = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)

    fingers_to_object = RewTerm(
        func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0
    )

    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.2,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(
        func=mdp.contacts,
        weight=0.5,
        params={"threshold": 1.0},
    )

    success = RewTerm(
        func=mdp.success_reward,
        weight=15,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_std": 0.1,
            "rot_std": 0.5,
            "command_name": "object_pose",
            "align_asset_cfg": SceneEntityCfg("object"),
        },
    )

    early_termination = RewTerm(
        func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"}
    )


@configclass
class HrlRewardsCfg:
    """Rewards that consume trajectory-command metrics instead of its 14-D public command."""

    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.002)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)
    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.25}, weight=1.0)
    good_finger_contact = RewTerm(func=mdp.contacts, weight=1.0, params={"threshold": 1.0})
    object_lifted = RewTerm(
        func=mdp.object_lifted_above_table,
        weight=2.0,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "table_cfg": SceneEntityCfg("table"),
            "height": 0.10,
            "table_half_height": 0.02,
        },
    )
    orientation_tracking = RewTerm(
        func=mdp.trajectory_orientation_tracking,
        weight=4.0,
        params={"command_name": "object_pose", "std": 0.5},
    )
    target_achieved = RewTerm(
        func=mdp.reorientation_goal_achieved,
        weight=15.0,
        params={"command_name": "object_pose"},
    )
    sequence_complete = RewTerm(
        func=mdp.sequence_completion_reward,
        weight=50.0,
        params={"command_name": "object_pose"},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-0.5, 1.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class HrlTerminationsCfg(TerminationsCfg):
    sequence_complete = DoneTerm(
        func=mdp.reorientation_sequence_complete,
        params={"command_name": "object_pose"},
    )


@configclass
class DexsuiteReorientEnvCfg(ManagerBasedRLEnvCfg):
    """Dexsuite reorientation task definition, also the base definition for derivative Lift task and evaluation task"""

    # Scene settings
    viewer: ViewerCfg = ViewerCfg(
        eye=(2.25, 0.0, 0.75), lookat=(0.0, 0.0, 0.45), origin_type="env"
    )
    scene: SceneCfg = SceneCfg(num_envs=4096,
                               env_spacing=3,
                               replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: mdp.CurriculumCfg | None = mdp.CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4  # 50 Hz

        # *single-goal setup
        self.commands.object_pose.resampling_time_range = (10.0, 10.0)
        self.commands.object_pose.position_only = False
        self.commands.object_pose.success_visualizer_cfg.markers["failure"] = (
            self.scene.table.spawn.replace(
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.15, 0.15), roughness=0.25
                ),
                visible=True,
            )
        )
        self.commands.object_pose.success_visualizer_cfg.markers["success"] = (
            self.scene.table.spawn.replace(
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.25, 0.15), roughness=0.25
                ),
                visible=True,
            )
        )

        self.episode_length_s = 4.0
        self.is_finite_horizon = True

        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15

        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = (
                self.rewards.success.params["pos_std"] / 2
            )

            self.curriculum.adr.params["rot_tol"] = (
                self.rewards.success.params["rot_std"] / 2
            )


class DexsuiteLiftEnvCfg(DexsuiteReorientEnvCfg):
    """Dexsuite lift task definition"""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.orientation_tracking = None  # no orientation reward
        self.commands.object_pose.position_only = True
        if self.curriculum is not None:
            self.rewards.success.params["rot_std"] = (
                None  # make success reward not consider orientation
            )
            self.curriculum.adr.params["rot_tol"] = (
                None  # make adr not tracking orientation
            )


class DexsuiteReorientEnvCfg_PLAY(DexsuiteReorientEnvCfg):
    """Dexsuite reorientation task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]


class DexsuiteLiftEnvCfg_PLAY(DexsuiteLiftEnvCfg):
    """Dexsuite lift task evaluation environment definition"""

    def __post_init__(self):
        super().__post_init__()
        self.commands.object_pose.resampling_time_range = (2.0, 3.0)
        self.commands.object_pose.debug_vis = True
        self.commands.object_pose.position_only = True
        self.curriculum.adr.params["init_difficulty"] = self.curriculum.adr.params[
            "max_difficulty"
        ]


##########
#  Tasks
##########


@configclass
class FrankaLeapMixinCfg:

    def __post_init__(self: DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "base"  # TODO: check this !!
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
                    prim_path="{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object/baseLink"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={
                "contact_sensor_names": [
                    f"{link}_object_s" for link in finger_tip_body_list
                ]
            },
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )

        self.observations.proprio.hand_tips_state_b.params[
            "body_asset_cfg"
        ].body_names = [".*fingertip.*"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[".*fingertip.*"]
        )


@configclass
class DexsuiteFrankaLeapReorientEnvCfg(FrankaLeapMixinCfg, DexsuiteReorientEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapReorientEnvCfg_PLAY(
    FrankaLeapMixinCfg, DexsuiteReorientEnvCfg_PLAY
):
    pass


@configclass
class DexsuiteFrankaLeapLiftEnvCfg(FrankaLeapMixinCfg, DexsuiteLiftEnvCfg):
    pass


@configclass
class DexsuiteFrankaLeapLiftEnvCfg_PLAY(FrankaLeapMixinCfg, DexsuiteLiftEnvCfg_PLAY):
    pass


@configclass
class DexsuiteFrankaLeapAnyRotateHrlEnvCfg(FrankaLeapMixinCfg, DexsuiteReorientEnvCfg):
    """Pick an object from the table and reach six sampled in-hand orientations."""

    commands: HrlCommandsCfg = HrlCommandsCfg()
    actions: HrlActionsCfg = HrlActionsCfg()
    rewards: HrlRewardsCfg = HrlRewardsCfg()
    terminations: HrlTerminationsCfg = HrlTerminationsCfg()
    events: HrlEventCfg = HrlEventCfg()
    curriculum: mdp.CurriculumCfg | None = None

    def __post_init__(self):
        self.observations.low_level = ObservationsCfg.LowLevelObsCfg()
        super().__post_init__()

        self.observations.policy.target_object_pose_b.func = mdp.command_object_pose_b
        self.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)
        self.commands.object_pose.position_only = False

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
        for sensor_name in (
            "thumb_fingertip_object_s",
            "fingertip_object_s",
            "fingertip_2_object_s",
            "fingertip_3_object_s",
        ):
            getattr(self.scene, sensor_name).filter_prim_paths_expr = [
                "{ENV_REGEX_NS}/Object/baseLink",
                "{ENV_REGEX_NS}/Table",
            ]
        self.observations.proprio.contact.params["filter_indices"] = [0]

        self.decimation = 4  # 30 Hz, matching the frozen hand policy.
        self.episode_length_s = 30.0
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

        # The MPC action restores arm gains on reset; the LEAP hand remains position-controlled.
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0
