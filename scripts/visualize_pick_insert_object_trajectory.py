"""Live IsaacSim visualization for the pick-insert object trajectory."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_TASK_DIR = REPO_ROOT / "src" / "tasks" / "pick_insert demo"


parser = argparse.ArgumentParser(description="Visualize the pick-insert object trajectory in IsaacSim.")
parser.add_argument(
    "--segment_steps",
    type=int,
    nargs=5,
    default=None,
    metavar=("MOVE", "ALIGN", "APPROACH", "INSERT", "HOLD"),
    help="Interpolation samples for each trajectory segment.",
)
parser.add_argument("--frame_dt", type=float, default=0.03, help="Wall-clock seconds between displayed frames.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop the trajectory until the viewer closes.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms  # noqa: E402


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _current_object_pose_b(scene: InteractiveScene) -> torch.Tensor:
    robot = scene["robot"]
    obj = scene["object"]
    pos_b, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        obj.data.root_pos_w,
        obj.data.root_quat_w,
    )
    return torch.cat((pos_b[0], quat_b[0]), dim=0)


def _trajectory_b_to_w(scene: InteractiveScene, trajectory_b: torch.Tensor) -> torch.Tensor:
    robot = scene["robot"]
    root_pos_w = robot.data.root_pos_w[0:1].repeat(trajectory_b.shape[0], 1)
    root_quat_w = robot.data.root_quat_w[0:1].repeat(trajectory_b.shape[0], 1)
    pos_w, quat_w = combine_frame_transforms(
        root_pos_w,
        root_quat_w,
        trajectory_b[:, :3],
        trajectory_b[:, 3:7],
    )
    return torch.cat((pos_w, quat_w), dim=1)


def _make_anchor_marker() -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PickInsertObjectTrajectory/AnchorFrame",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.05, 0.05, 0.05),
            )
        },
    )
    markers = VisualizationMarkers(cfg)
    markers.set_visibility(True)
    return markers


def _leap_joint_names(scene: InteractiveScene) -> list[str]:
    names = [name for name in scene["robot"].joint_names if name.startswith("a_")]
    if not names:
        raise RuntimeError("Could not find LEAP joints matching 'a_*' on scene['robot'].")
    return names


def _joint_indices(scene: InteractiveScene, joint_names: Sequence[str]) -> list[int]:
    name_to_index = {name: index for index, name in enumerate(scene["robot"].joint_names)}
    return [name_to_index[name] for name in joint_names]


def _write_demo_frame(
    scene: InteractiveScene,
    object_pose_w: torch.Tensor,
    leap_joint_indices: Sequence[int],
    leap_joint_pos: torch.Tensor,
) -> None:
    obj = scene["object"]
    obj.write_root_pose_to_sim(object_pose_w.unsqueeze(0))
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=object_pose_w.device, dtype=object_pose_w.dtype))

    robot = scene["robot"]
    joint_pos = robot.data.joint_pos[0].clone()
    joint_vel = robot.data.joint_vel[0].clone()
    joint_pos[list(leap_joint_indices)] = leap_joint_pos.to(device=joint_pos.device, dtype=joint_pos.dtype)
    joint_vel[list(leap_joint_indices)] = 0.0
    robot.write_joint_state_to_sim(joint_pos.unsqueeze(0), joint_vel.unsqueeze(0))
    scene.write_data_to_sim()


def main() -> None:
    print("[INFO]: Loading pick-insert object trajectory helper.", flush=True)
    trajectory_module = _load_module(
        "pick_insert_demo_object_trajectory",
        DEMO_TASK_DIR / "object_trajectory.py",
    )
    print("[INFO]: Loading pick-insert demo scene config.", flush=True)
    env_cfg_module = _load_module("pick_insert_demo_env_cfg", DEMO_TASK_DIR / "env_cfg.py")

    print("[INFO]: Creating SimulationContext.", flush=True)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    print("[INFO]: Creating demo InteractiveScene.", flush=True)
    scene = InteractiveScene(env_cfg_module.SceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False))
    print("[INFO]: Resetting simulation and scene.", flush=True)
    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())
    print("[INFO]: Updating SimulationApp once after reset.", flush=True)
    simulation_app.update()

    print("[INFO]: Building object, hand, and anchor trajectories.", flush=True)
    current_pose_b = _current_object_pose_b(scene)
    segment_steps = tuple(args_cli.segment_steps or trajectory_module.DEFAULT_SEGMENT_STEPS)
    leap_joint_names = _leap_joint_names(scene)
    leap_joint_indices = _joint_indices(scene, leap_joint_names)
    motion_b = trajectory_module.build_pick_insert_demo_motion(
        current_pose_b,
        leap_joint_names,
        segment_steps=segment_steps,
    )
    trajectory_w = _trajectory_b_to_w(scene, motion_b["object_pose_b"])
    anchor_trajectory_w = _trajectory_b_to_w(scene, motion_b["anchor_pose_b"])

    anchor_marker = _make_anchor_marker()

    sim.set_camera_view(eye=(1.0, -1.2, 0.75), target=(0.35, 0.0, 0.33))

    print(
        "[INFO]: Visualizing "
        f"{trajectory_w.shape[0]} object trajectory frames. "
        "Close the IsaacSim window or stop the process to exit."
        ,
        flush=True,
    )
    frame = 0
    while not simulation_app.is_exiting():
        _write_demo_frame(
            scene,
            trajectory_w[frame],
            leap_joint_indices,
            motion_b["leap_joint_pos"][frame],
        )
        anchor_marker.visualize(
            anchor_trajectory_w[frame : frame + 1, :3],
            anchor_trajectory_w[frame : frame + 1, 3:7],
        )
        sim.render()
        scene.update(sim.get_physics_dt())

        if frame < trajectory_w.shape[0] - 1:
            frame += 1
        elif args_cli.loop:
            frame = 0

        if args_cli.frame_dt > 0.0:
            time.sleep(args_cli.frame_dt)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("[ERROR]: visualize_pick_insert_object_trajectory.py failed before the visualization loop.", flush=True)
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
