"""Recorded pouring with cuRobo arm control and a frozen LEAP hand policy."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.utils import configclass

from src.policy.high_level.anchor_correction import AnchorCorrectionCfg
from src.tasks.clean_table.env_cfg import (
    DexsuiteReorientEnvCfg,
    FrankaLeapMixinCfg,
    HrlActionsCfg,
    ObservationsCfg,
)
from src.tasks.common.mdps.events import reset_arm_mpc, reset_joints_to_init_state
from src.tasks.pouring.commands import PouringTrajectoryCommandCfg
from src.tasks.pouring.trajectory import ASSET_DIR, load_pouring_trajectory


@configclass
class CommandsCfg:
    object_pose = PouringTrajectoryCommandCfg(
        asset_name="robot", object_name="object", success_vis_asset_name="table",
        resampling_time_range=(1.0e6, 1.0e6), position_only=False,
        hand_base_to_anchor_pose=(
            0.06116479, 0.02922082, 0.07563708,
            -0.99926311, 0.01318283, 0.02230061, 0.02832354,
        ),
        correction=AnchorCorrectionCfg(
            enable=True, slew_pos=1.0, slew_rot=10.0, max_pos=0.05, max_rot=0.2,
            anchor_achieved_pos=0.01, anchor_achieved_rot=0.05,
            stall_window=5, stall_delta_pos=0.003, stall_delta_rot=0.01,
        ),
    )


@configclass
class PouringEnvCfg(FrankaLeapMixinCfg, DexsuiteReorientEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    actions: HrlActionsCfg = HrlActionsCfg()
    bottle_scale: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        # cuRobo emits one interpolated target per env step; match its clock to physics.
        self.actions.arm_action.optimization_dt = (
            self.decimation * self.sim.dt * self.actions.arm_action.interpolation_steps
        )
        self.episode_length_s = 120.0
        self.sim.render_interval = self.decimation
        self.sim.gravity = (0.0, 0.0, -1.81)
        self.scene.num_envs = 1
        self.scene.lazy_sensor_update = False
        # Keep the table below the elbow throughout the pouring sweep (top at 5.5 cm).
        self.scene.table.init_state.pos = (0.55, 0.0, 0.035)
        self.scene.robot.actuators["joints"].stiffness = 0.0
        self.scene.robot.actuators["joints"].damping = 0.0

        # Rough MANO palm alignment, with fingers settled by the frozen policy offline.
        initial_joint_pos = {
            "panda_joint1": 0.33303285,
            "panda_joint2": 0.02724354,
            "panda_joint3": 0.10259372,
            "panda_joint4": -2.01124477,
            "panda_joint5": -1.19782615,
            "panda_joint6": 1.69792521,
            "panda_joint7": 0.26261455,
            "a_1": 0.87079108,
            "a_12": 0.66728115,
            "a_5": 1.28090024,
            "a_9": 1.30813885,
            "a_0": -0.66696095,
            "a_13": 1.10134733,
            "a_4": -0.63867444,
            "a_8": -0.52542347,
            "a_2": 1.11289513,
            "a_14": 1.25342405,
            "a_6": 0.38950193,
            "a_10": 0.21116957,
            "a_3": 1.67980564,
            "a_15": 1.39808440,
            "a_7": 1.52062285,
            "a_11": 1.63532317,
        }
        # Keep zero finger defaults for the distance gate's open-hand fallback.
        self.scene.robot.init_state.joint_pos.update({
            name: value for name, value in initial_joint_pos.items() if name.startswith("panda_")
        })
        self.events.reset_robot_joints = EventTermCfg(
            func=reset_joints_to_init_state, mode="reset", params={"joint_pos": initial_joint_pos},
        )
        self.events.reset_arm_mpc = EventTermCfg(func=reset_arm_mpc, mode="reset")

        command = self.commands.object_pose
        poses, _, _ = load_pouring_trajectory(
            command.trajectory_path, command.trajectory_stride, command.grasp_frame
        )
        mesh = MeshConverter(MeshConverterCfg(
            asset_path=str(ASSET_DIR.parent / "pick_and_place/bottle/bottle.obj"),
            usd_dir="/tmp/d2h_pouring_bottle_usd", make_instanceable=False,
            scale=(self.bottle_scale,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=True, enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=sim_utils.ConvexHullPropertiesCfg(),
        ))
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=mesh.usd_path, activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=command.initial_object_position, rot=tuple(float(v) for v in poses[0, 3:]),
            ),
        )
        self.scene.receptive_object = None
        self.scene.bottle_hand_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            # PhysX needs one rigid body per filter; a wildcard also matches non-body prims.
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right/" + name
                for name in (
                    "base", "mcp_joint", "pip", "dip", "fingertip",
                    "mcp_joint_2", "pip_2", "dip_2", "fingertip_2",
                    "mcp_joint_3", "pip_3", "dip_3", "fingertip_3",
                    "thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip",
                )
            ],
            history_length=self.decimation,
        )
        for name in ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"):
            sensor = getattr(self.scene, f"{name}_object_s")
            sensor.filter_prim_paths_expr = ["{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/Table"]

        low_level = ObservationsCfg.LowLevelObsCfg()
        # PhysX and the commanded hand pose can use opposite quaternion signs.
        # Give the policy +identity when object and goal orientations coincide.
        low_level.goal_quat_diff.params["make_quat_unique"] = True
        for name in ("external_contact_mask", "external_contact_force_mag", "external_contact_pose"):
            getattr(low_level, name).params["filter_indices"] = [1]
        self.observations.low_level = low_level
        self.observations.policy = low_level.copy()
        self.observations.proprio = None
        self.observations.perception = None
        for name in ("lift", "transport", "inside_box", "success"):
            setattr(self.rewards, name, None)
        # Deterministic validation: fixed robot/object resets.
        for name in (
            "robot_physics_material", "object_physics_material", "joint_stiffness_and_damping",
            "joint_friction", "object_scale_mass", "reset_table", "reset_receptive_object",
            "variable_gravity",
        ):
            setattr(self.events, name, None)
        self.events.reset_object.params["pose_range"] = {}
