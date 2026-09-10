"""Reusable observation groups; task policy observations remain in each task."""

from dataclasses import MISSING

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import src.tasks.common.mdps as mdp

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
    """155-D current-step state used by the frozen LEAP hand policy."""

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
        func=mdp.tip_contact_mask_obs,
        params={
            "contact_sensor_names": [
                "thumb_fingertip_object_s",
                "fingertip_object_s",
                "fingertip_2_object_s",
                "fingertip_3_object_s",
            ],
            "force_threshold": 0.25,
            "filter_indices": [0],
        },
    )
    contact_force_mag = ObsTerm(
        func=mdp.tip_contact_force_mag_obs,
        params={
            "contact_sensor_names": [
                "thumb_fingertip_object_s",
                "fingertip_object_s",
                "fingertip_2_object_s",
                "fingertip_3_object_s",
            ],
            "force_threshold": 0.25,
            "filter_indices": [0],
        },
    )
    contact_pose = ObsTerm(
        func=mdp.tip_contact_pose_flat,
        params={
            "contact_sensor_names": [
                "thumb_fingertip_object_s",
                "fingertip_object_s",
                "fingertip_2_object_s",
                "fingertip_3_object_s",
            ],
            "force_threshold": 0.25,
            "contact_pose_range_deg": 90.0,
            "filter_indices": [0],
        },
    )
    external_contact_mask = ObsTerm(
        func=mdp.tip_contact_mask_obs,
        params={
            "contact_sensor_names": [
                "thumb_fingertip_object_s",
                "fingertip_object_s",
                "fingertip_2_object_s",
                "fingertip_3_object_s",
            ],
            "force_threshold": 0.25,
            "filter_indices": [1, 2],
        },
    )
    external_contact_force_mag = ObsTerm(
        func=mdp.tip_contact_force_mag_obs,
        params={
            "contact_sensor_names": [
                "thumb_fingertip_object_s",
                "fingertip_object_s",
                "fingertip_2_object_s",
                "fingertip_3_object_s",
            ],
            "force_threshold": 0.25,
            "filter_indices": [1, 2],
        },
    )
    external_contact_pose = ObsTerm(
        func=mdp.tip_contact_pose_flat,
        params={
            "contact_sensor_names": [
                "thumb_fingertip_object_s",
                "fingertip_object_s",
                "fingertip_2_object_s",
                "fingertip_3_object_s",
            ],
            "force_threshold": 0.25,
            "contact_pose_range_deg": 45.0,
            "filter_indices": [1, 2],
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
    last_action = ObsTerm(
        func=mdp.last_action,
        params={"action_name": "hand_action"},
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True
        self.history_length = 1
