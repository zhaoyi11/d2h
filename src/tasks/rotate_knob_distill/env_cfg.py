"""Fixed LEAP/ARIA-knob environment for teacher-student distillation."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from src.assets.franka_leap_hand.leap import LEAP_HAND_CFG
import src.tasks.rotate_knob_distill.mdps as mdp
from src.tasks.rotate_object.env_cfg import ObservationsCfg as RotateObjectObservationsCfg


ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
FINGER_PATHS = {
    "thumb_fingertip_object_s": "{ENV_REGEX_NS}/Robot/thumb_fingertip",
    "fingertip_object_s": "{ENV_REGEX_NS}/Robot/fingertip",
    "fingertip_2_object_s": "{ENV_REGEX_NS}/Robot/fingertip_2",
    "fingertip_3_object_s": "{ENV_REGEX_NS}/Robot/fingertip_3",
}
CONTACT_TARGETS = [
    "{ENV_REGEX_NS}/Object/handle",
    "{ENV_REGEX_NS}/ReceptiveObject",
    "{ENV_REGEX_NS}/Table",
]


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=LEAP_HAND_CFG.init_state.replace(
            pos=(-0.07, 0.0, 0.13),
            rot=(0.7071, 0.0, 0.7071, 0.0),
        ),
    )
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / "aria/knob1/knob1_handle.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                disable_gravity=True,
                enable_gyroscopic_forces=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    receptive_object = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.01, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=False,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.35)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.25)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
        collision_group=-1,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    fingertip_transforms = FrameTransformerCfg(
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
    )
    thumb_fingertip_object_s = ContactSensorCfg(
        prim_path=FINGER_PATHS["thumb_fingertip_object_s"], filter_prim_paths_expr=CONTACT_TARGETS
    )
    fingertip_object_s = ContactSensorCfg(
        prim_path=FINGER_PATHS["fingertip_object_s"], filter_prim_paths_expr=CONTACT_TARGETS
    )
    fingertip_2_object_s = ContactSensorCfg(
        prim_path=FINGER_PATHS["fingertip_2_object_s"], filter_prim_paths_expr=CONTACT_TARGETS
    )
    fingertip_3_object_s = ContactSensorCfg(
        prim_path=FINGER_PATHS["fingertip_3_object_s"], filter_prim_paths_expr=CONTACT_TARGETS
    )


@configclass
class CommandsCfg:
    object_pose = mdp.KnobTargetCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(1.0e6, 1.0e6),
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class StudentCfg(ObsGroup):
        frame = ObsTerm(func=mdp.student_frame, history_length=3, flatten_history_dim=True)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: StudentCfg = StudentCfg()
    low_level: RotateObjectObservationsCfg.LowLevelObsCfg = RotateObjectObservationsCfg.LowLevelObsCfg()


@configclass
class ActionsCfg:
    hand_action = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=["a_.*"],
        alpha=0.5,
        rescale_to_limits=True,
    )


@configclass
class RewardsCfg:
    yaw_tracking = RewTerm(func=mdp.yaw_tracking, weight=4.0, params={"std": 0.5})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.knob_success,
        params={"angle_threshold": 0.02, "velocity_threshold": 0.2},
    )


@configclass
class EventCfg:
    anchor_object_z_axis_joint = EventTerm(
        func=mdp.anchor_object_z_axis_joint,
        mode="prestartup",
        params={"asset_name": "object", "joint_name": "z_axis_joint"},
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    reset_hand_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.008, 0.008), "y": (-0.008, 0.008), "z": (-0.008, 0.008),
                "roll": (-0.08, 0.08), "pitch": (-0.08, 0.08), "yaw": (-0.08, 0.08),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_hand_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RotateKnobDistillEnvCfg(ManagerBasedRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(eye=(0.5, 0.5, 0.4), lookat=(0.0, 0.0, 0.08), origin_type="env")
    scene: SceneCfg = SceneCfg(num_envs=2048, env_spacing=0.6, replicate_physics=False)
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 240.0,
        render_interval=4,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.02,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.is_finite_horizon = True


@configclass
class RotateKnobDistillEnvCfg_PLAY(RotateKnobDistillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
