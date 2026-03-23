from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import src.tasks.common.mdps as mdp
import src.tasks.reorient.mdps as task_mdp
from isaaclab.envs.mdp.actions import RelativeJointPositionActionCfg
from isaaclab.envs.mdp.actions import JointPositionToLimitsActionCfg
from src.tasks.common.mdps.action_manager.action_cfg import EMARelativeJointPositionToLimitsActionCfg
from src.tasks.reorient.env_cfg import LeapObjectEnvCfg, ObservationsCfg


_ROBOT_ROOT_POSE = (0.0, 0.0, 0.5, 0.5, 0.5, -0.5, 0.5)
_TIP_SENSOR_PATHS = {
    "thumb_tip_object_s": "{ENV_REGEX_NS}/Robot/thumb_fingertip",
    "index_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip",
    "middle_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip_2",
    "ring_tip_object_s": "{ENV_REGEX_NS}/Robot/fingertip_3",
}
_OBJECT_FILTER_EXPR = [
    "{ENV_REGEX_NS}/Object",
    "{ENV_REGEX_NS}/Object/.*",
    "{ENV_REGEX_NS}/Object.*",
]


def _default_grasp_cache_path() -> str:
    """Return the default repo-local path for saved grasp caches."""
    return str(Path(__file__).resolve().parents[3] / "saved_grasp" / "stable_grasps.npy")


def _resolve_object_asset_path(cfg: "StableGraspMixinCfg") -> str | None:
    """Resolve the object asset path from explicit config or saved grasp metadata."""
    if cfg.object_asset_path is not None:
        return cfg.object_asset_path
    if getattr(cfg, "object_urdf_path", None) is not None:
        return cfg.object_urdf_path
    scene_object = getattr(getattr(cfg, "scene", None), "object", None)
    if scene_object is not None and getattr(scene_object, "spawn", None) is not None:
        asset_path = getattr(scene_object.spawn, "asset_path", None)
        if asset_path:
            return asset_path

    cache_path = cfg.grasp_cache_path or cfg.grasp_path
    if cache_path is None or not Path(cache_path).is_file():
        return None

    try:
        payload = np.load(cache_path, allow_pickle=True)
        data = payload.item() if hasattr(payload, "item") else payload
    except Exception:
        return None

    if isinstance(data, dict):
        asset_path = data.get("object_asset_path")
        if isinstance(asset_path, str) and asset_path:
            return asset_path
    return None


def _make_single_object_cfg(asset_path: str, scale: float) -> RigidObjectCfg:
    """Create the single-object rigid-body config used for generation and replay."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=asset_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
            ),
            fix_base=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=None,
                    damping=None,
                ),
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            scale=(scale, scale, scale),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.05, 0.65),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _attach_grasp_contact_sensors(cfg: "StableGraspMixinCfg"):
    """Attach fingertip and palm contact sensors filtered to the object."""
    for sensor_name, prim_path in _TIP_SENSOR_PATHS.items():
        setattr(
            cfg.scene,
            sensor_name,
            ContactSensorCfg(
                prim_path=prim_path,
                filter_prim_paths_expr=list(_OBJECT_FILTER_EXPR),
                update_period=0.0,
                history_length=6,
            ),
        )

    cfg.scene.palm_object_s = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        filter_prim_paths_expr=list(_OBJECT_FILTER_EXPR),
        update_period=0.0,
        history_length=6,
    )


@configclass
class StableGraspCommandsCfg:
    """Stable-grasp generator does not need task commands."""


@configclass
class StableGraspRewardsCfg:
    """Stable-grasp generation does not use shaped rewards."""


@configclass
class StableGraspCurriculumCfg:
    """No curriculum terms are used for stable-grasp generation."""


@configclass
class StableGraspObservationsCfg(ObservationsCfg):
    @configclass
    class KinematicObsGroupCfg(ObservationsCfg.KinematicObsGroupCfg):
        def __post_init__(self):
            super().__post_init__()
            self.goal_pose = None
            self.goal_quat_diff = None

    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()


@configclass
class StableGraspGenActionsCfg:
    """Relative action targets let zero actions hold the sampled grasp pose."""

    # joint_pos = JointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     rescale_to_limits=True,
    # )
    joint_pos = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        debug_vis=True,
        use_zero_offset=False,
        scale=0.25,
        offset=0.0,
    )

    # joint_pos = EMARelativeJointPositionToLimitsActionCfg(
    #     asset_name="robot",
    #     joint_names=[".*"],
    #     alpha=0.95,
    #     scale=0.25,
    #     rescale_to_limits=False,
    # )


@configclass
class StableGraspGenEventsCfg:
    collect_stable_grasps = EventTerm(
        func=task_mdp.collect_stable_grasp_states,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "object_asset_cfg": SceneEntityCfg("object"),
            "cache_path": _default_grasp_cache_path(),
            "max_cached_grasp_size": 1024,
            "object_asset_path": None,
        },
    )

    reset_object = EventTerm(
        func=task_mdp.reset_object_pose_in_robot_root_frame,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "object_asset_cfg": SceneEntityCfg("object"),
            "position_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
            "euler_range": {
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_root = EventTerm(
        func=task_mdp.reset_root_state_from_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose": _ROBOT_ROOT_POSE,
            "velocity": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=task_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "joint_reset_delta": 0.25,
        },
    )


@configclass
class StableGraspReplayEventsCfg:
    reset_from_saved_stable_grasp = EventTerm(
        func=task_mdp.sample_saved_stable_grasps,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "object_asset_cfg": SceneEntityCfg("object"),
            "cache_path": _default_grasp_cache_path(),
        },
    )


@configclass
class StableGraspGenTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # stable_grasp_invalid = DoneTerm(
    #     func=task_mdp.stable_grasp_invalid,
    #     params={
    #         "tip_contact_sensor_names": list(_TIP_SENSOR_PATHS.keys()),
    #         "required_tip_contact_sensor_names": [
    #             "thumb_tip_object_s",
    #             "middle_tip_object_s",
    #         ],
    #         "non_tip_contact_sensor_names": ["palm_object_s"],
    #         "tip_contact_force_threshold": 0.25,
    #         "min_tip_contacts": 2,
    #         "fingertip_frame_sensor_name": "fingertip_transforms",
    #         "fingertip_distance_sum_limit": 0.25,
    #         "object_cfg": SceneEntityCfg("object"),
    #     },
    # )

    object_out_of_reach = DoneTerm(
        func=task_mdp.object_away_from_robot,
        params={"threshold": 0.35},
    )


class StableGraspMixinCfg:
    """Shared fields and helpers for stable-grasp generator and replay configs."""

    object_asset_path: str | None = None
    grasp_cache_path: str | None = None
    max_cached_grasp_size: int = 4096
    contact_force_threshold: float = 0.25
    min_tip_contacts: int = 2
    fingertip_distance_sum_limit: float = 0.25
    joint_reset_delta: float = 0.25
    object_pos_range_handframe: dict[str, tuple[float, float]] = {
        "x": (-0.01, 0.01),
        "y": (-0.01, 0.01),
        "z": (0.01, 0.01),
    }
    object_euler_range_handframe: dict[str, tuple[float, float]] = {
        "roll": (-torch.pi, torch.pi),
        "pitch": (-torch.pi, torch.pi),
        "yaw": (-torch.pi, torch.pi),
    }

    def _configure_single_object_scene(self):
        """Replace the multi-asset object with a single fixed object asset when provided."""
        asset_path = _resolve_object_asset_path(self)
        if asset_path is None:
            return

        scale = 1.0 if self.object_scale_override is None else float(self.object_scale_override)
        self.scene.object = _make_single_object_cfg(asset_path, scale)

    def _configure_grasp_contact_sensors(self):
        """Install fingertip and palm contact sensors for the object."""
        _attach_grasp_contact_sensors(self)


@configclass
class LeapObjectStableGraspGenEnvCfg(StableGraspMixinCfg, LeapObjectEnvCfg):
    """Generate stable grasp libraries by holding randomized candidates under zero action."""

    observations: StableGraspObservationsCfg = StableGraspObservationsCfg()
    actions: StableGraspGenActionsCfg = StableGraspGenActionsCfg()
    commands: StableGraspCommandsCfg = StableGraspCommandsCfg()
    rewards: StableGraspRewardsCfg = StableGraspRewardsCfg()
    terminations: StableGraspGenTerminationsCfg = StableGraspGenTerminationsCfg()
    events: StableGraspGenEventsCfg = StableGraspGenEventsCfg()
    curriculum: StableGraspCurriculumCfg = StableGraspCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        self.grasp_cache_path = self.grasp_cache_path or self.grasp_path or _default_grasp_cache_path()
        self.object_asset_path = _resolve_object_asset_path(self)
        self._configure_single_object_scene()
        self._configure_grasp_contact_sensors()

        self.episode_length_s = 4.0

        self.events.collect_stable_grasps.params["cache_path"] = self.grasp_cache_path
        self.events.collect_stable_grasps.params["max_cached_grasp_size"] = self.max_cached_grasp_size
        self.events.collect_stable_grasps.params["object_asset_path"] = self.object_asset_path
        # self.events.reset_robot_joints.params["joint_reset_delta"] = self.joint_reset_delta

        self.events.reset_object.params["position_range"] = self.object_pos_range_handframe
        self.events.reset_object.params["euler_range"] = self.object_euler_range_handframe

        # self.terminations.stable_grasp_invalid.params["tip_contact_force_threshold"] = (
        #     self.contact_force_threshold
        # )
        # self.terminations.stable_grasp_invalid.params["min_tip_contacts"] = self.min_tip_contacts
        # self.terminations.stable_grasp_invalid.params["fingertip_distance_sum_limit"] = (
        #     self.fingertip_distance_sum_limit
        # )


@configclass
class LeapObjectGraspInitEnvCfg(StableGraspMixinCfg, LeapObjectEnvCfg):
    """Replay previously generated stable grasps as reset initializations."""

    events: StableGraspReplayEventsCfg = StableGraspReplayEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.grasp_cache_path = self.grasp_cache_path or self.grasp_path or _default_grasp_cache_path()
        self.object_asset_path = _resolve_object_asset_path(self)
        self._configure_single_object_scene()
        self._configure_grasp_contact_sensors()

        self.events.reset_from_saved_stable_grasp.params["cache_path"] = (
            self.grasp_cache_path or _default_grasp_cache_path()
        )
