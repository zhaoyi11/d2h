"""Shared task configuration pieces, assembled explicitly by each environment."""

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from src.assets.franka_leap_hand.franka_leap import FRANKA_LEAP_HAND_CFG

import src.tasks.common.mdps as mdp


FRANKA_HAND_PRIM_PATH = "{ENV_REGEX_NS}/Robot/Franka_LeapHand/leap_hand_right"
FINGERTIP_NAMES = ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3")


def get_visdex_usd_paths() -> list[str]:
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
class JointActionsCfg:

    action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.1
    )


@configclass
class HrlActionsCfg:
    """cuRobo arm plus the externally supplied frozen-policy hand action."""

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
class ObjectTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-0.5, 1.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class ArmHandActionsCfg:
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.1,
    )
    hand_action = HrlActionsCfg().hand_action


def table_cfg(pos=(0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visible=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def ground_plane_cfg(pos=(0.0, 0.0, 0.0)):
    return AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )


def light_cfg():
    return AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


def fingertip_transforms_cfg(hand_prim_path=FRANKA_HAND_PRIM_PATH):
    return FrameTransformerCfg(
        prim_path=hand_prim_path + "/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=hand_prim_path + "/" + name,
                offset=OffsetCfg(pos=offset),
            )
            for name, offset in zip(
                FINGERTIP_NAMES,
                ((0.0, -0.045, -0.015),) + ((0.0, -0.03, 0.015),) * 3,
            )
        ],
        debug_vis=False,
    )


def configure_fingertip_contacts(scene, filter_prim_paths):
    for name in FINGERTIP_NAMES:
        setattr(scene, f"{name}_object_s", ContactSensorCfg(
            prim_path=FRANKA_HAND_PRIM_PATH + "/" + name,
            filter_prim_paths_expr=list(filter_prim_paths),
            debug_vis=False,
        ))


def fingertip_contact_observation(filter_indices=None):
    params = {"contact_sensor_names": [f"{name}_object_s" for name in FINGERTIP_NAMES]}
    if filter_indices is not None:
        params["filter_indices"] = filter_indices
    return ObsTerm(func=mdp.fingers_contact_force_b, params=params, clip=(-20.0, 20.0))


def configure_contact_physics(sim, *, found_lost_pairs=2**23, collision_stack_size=2**31):
    sim.physx.solver_type = 1
    sim.physx.max_position_iteration_count = 192
    sim.physx.max_velocity_iteration_count = 1
    sim.physx.bounce_threshold_velocity = 0.02
    sim.physx.friction_offset_threshold = 0.01
    sim.physx.friction_correlation_distance = 0.0005
    sim.physx.gpu_found_lost_aggregate_pairs_capacity = found_lost_pairs
    sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
    sim.physx.gpu_max_rigid_contact_count = 2**23
    sim.physx.gpu_max_rigid_patch_count = 2**23
    sim.physx.gpu_collision_stack_size = collision_stack_size


def configure_success_visualization(command, asset):
    for name, color in (("failure", (0.25, 0.15, 0.15)), ("success", (0.15, 0.25, 0.15))):
        command.success_visualizer_cfg.markers[name] = asset.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.25),
            visible=True,
        )


def franka_robot_cfg():
    return FRANKA_LEAP_HAND_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FRANKA_LEAP_HAND_CFG.spawn.replace(
            articulation_props=FRANKA_LEAP_HAND_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=False,
            ),
        ),
    )


def tote_cfg():
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(
                Path(__file__).resolve().parents[2] / "assets/symdex/tote_collision.usd"
            ),
            scale=(0.6, 0.6, 0.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, -0.35, 0.271),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),
        ),
    )


@configclass
class TabletopSceneCfg(InteractiveSceneCfg):
    table: RigidObjectCfg = table_cfg()
    robot: ArticulationCfg = franka_robot_cfg()
    object: RigidObjectCfg = MISSING
    receptive_object: RigidObjectCfg | None = None
    plane = ground_plane_cfg()
    light = light_cfg()


@configclass
class TabletopEventsCfg:
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

    reset_receptive_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.10, 0.10], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("receptive_object"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.03, 0.03], "y": [-0.03, 0.03], "yaw": [0.0, 0.0]},
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

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, -1.81], [0.0, 0.0, -1.81]),
            "operation": "abs",
        },
    )
