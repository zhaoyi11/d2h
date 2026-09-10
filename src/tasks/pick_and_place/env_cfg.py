"""Franka + LEAP picks a recorded pig object off the table and deposits it in a box."""

import trimesh
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.utils import configclass

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.tasks.clean_table.env_cfg import (
    CommandsCfg as CleanTableCommandsCfg,
    DexsuiteFrankaLeapCleanTableHrlEnvCfg,
    ObservationsCfg,
)
from src.tasks.common.mdps.events import reset_arm_mpc, reset_joints_to_init_state
from src.tasks.pick_and_place.commands import PickAndPlaceTrajectoryCommandCfg
from src.tasks.pick_and_place.trajectory import PIG_MESH, load_carry_trajectory


@configclass
class CommandsCfg:
    object_pose = PickAndPlaceTrajectoryCommandCfg(
        asset_name="robot", object_name="object", success_vis_asset_name="table",
        resampling_time_range=(1.0e6, 1.0e6), position_only=False,
        enable_drop_recovery=True, capture_goal_after_settle=True,
        recovery_settle_speed=0.05, recovery_settle_steps=5,
        recovery_arm_after_stage=1, grasp_stall_steps=60, hand_open_until_stage=0,
        correction=CleanTableCommandsCfg().object_pose.correction,
    )


@configclass
class PickAndPlaceEnvCfg(DexsuiteFrankaLeapCleanTableHrlEnvCfg):
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        command = self.commands.object_pose
        command.stage_object_tolerances = (StageObjTol(0.05, 1.0),) * 6
        self.episode_length_s = 120.0
        self.sim.gravity = (0.0, 0.0, -1.81)
        self.scene.num_envs = 1
        self.scene.lazy_sensor_update = False
        self.actions.arm_action.optimization_dt = (
            self.decimation * self.sim.dt * self.actions.arm_action.interpolation_steps
        )
        self.events.reset_robot_joints = EventTermCfg(
            func=reset_joints_to_init_state, mode="reset",
            params={"joint_pos": self.scene.robot.init_state.joint_pos},
        )
        self.events.reset_arm_mpc = EventTermCfg(func=reset_arm_mpc, mode="reset")

        carry, _, _ = load_carry_trajectory(
            command.trajectory_path, command.carry_start_frame, command.carry_end_frame,
            command.trajectory_stride,
        )
        vertices = trimesh.load(PIG_MESH, force="mesh").vertices
        rotation = Rotation.from_quat(carry[0, 3:], scalar_first=True)
        table_top = self.scene.table.init_state.pos[2] + command.table_half_height
        # A floor-level grasp is outside this tabletop task; retry from the supported start.
        self.terminations.object_out_of_bound.params["in_bound_range"]["z"] = (table_top - 0.02, 2.0)
        start_z = table_top - rotation.apply(vertices)[:, 2].min() + 0.002
        mesh = MeshConverter(MeshConverterCfg(
            asset_path=str(PIG_MESH), usd_dir="/tmp/d2h_pick_and_place_pig_usd",
            make_instanceable=False, scale=(0.6, 0.6, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False, enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mesh_collision_props=sim_utils.ConvexHullPropertiesCfg(),
        ))
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=mesh.usd_path, activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.55, 0.10, float(start_z)), rot=tuple(float(v) for v in carry[0, 3:]),
            ),
        )
        for name in ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"):
            getattr(self.scene, f"{name}_object_s").filter_prim_paths_expr = [
                "{ENV_REGEX_NS}/Object", "{ENV_REGEX_NS}/ReceptiveObject", "{ENV_REGEX_NS}/Table",
            ]
        low_level = ObservationsCfg.LowLevelObsCfg()
        low_level.goal_quat_diff.params["make_quat_unique"] = True
        self.observations.low_level = low_level
        self.observations.policy = low_level.copy()
        self.observations.proprio = None
        self.observations.perception = None
        for name in (
            "robot_physics_material", "object_physics_material", "joint_stiffness_and_damping",
            "joint_friction", "object_scale_mass", "variable_gravity",
        ):
            setattr(self.events, name, None)
        for name in ("reset_object", "reset_table", "reset_receptive_object"):
            getattr(self.events, name).params["pose_range"] = {}
