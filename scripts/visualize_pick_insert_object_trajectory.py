"""Live IsaacSim visualization for the pick-insert object trajectory."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
# The object-pose builder lives with the task; the anchor + hand-base are computed by the same runtime
# function the command term uses (``hand_base_pose_from_object_command_b``), so the visualization shows
# exactly the root-aligned anchor the cuRobo-MPC arm tracks (open-loop -- no PI(D) correction here).
PICK_INSERT_TRAJECTORY_PATH = REPO_ROOT / "src" / "tasks" / "pick_insert" / "mdps" / "trajectory.py"
ENV_CFG_PATH = REPO_ROOT / "src" / "tasks" / "pick_insert" / "env_cfg.py"


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
from src.policy.high_level.anchor_kinematics import (  # noqa: E402
    DEFAULT_HAND_BASE_TO_ANCHOR_POSE,
    DEFAULT_OBJECT_TO_ANCHOR_POSE,
    hand_base_pose_from_object_command_b,
)


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name!r} from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_robotless_scene_cfg(env_cfg_module: ModuleType):
    scene_cfg = env_cfg_module.SceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    scene_cfg.robot = None
    return scene_cfg


def _current_object_pose_w(scene: InteractiveScene) -> torch.Tensor:
    obj = scene["object"]
    return torch.cat((obj.data.root_pos_w[0], obj.data.root_quat_w[0]), dim=0)


def _make_frame_marker(prim_path: str) -> VisualizationMarkers:
    cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
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


def _write_demo_frame(scene: InteractiveScene, object_pose_w: torch.Tensor) -> None:
    obj = scene["object"]
    obj.write_root_pose_to_sim(object_pose_w.unsqueeze(0))
    obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=object_pose_w.device, dtype=object_pose_w.dtype))
    scene.write_data_to_sim()


def main() -> None:
    print("[INFO]: Loading pick-insert object trajectory helper.", flush=True)
    trajectory_module = _load_module(
        "pick_insert_trajectory",
        PICK_INSERT_TRAJECTORY_PATH,
    )
    print("[INFO]: Loading pick-insert scene config.", flush=True)
    env_cfg_module = _load_module("pick_insert_env_cfg", ENV_CFG_PATH)

    print("[INFO]: Creating SimulationContext.", flush=True)
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    print("[INFO]: Creating demo InteractiveScene.", flush=True)
    scene = InteractiveScene(_make_robotless_scene_cfg(env_cfg_module))
    print("[INFO]: Resetting simulation and scene.", flush=True)
    sim.reset()
    scene.reset()
    scene.update(sim.get_physics_dt())
    print("[INFO]: Updating SimulationApp once after reset.", flush=True)
    simulation_app.update()

    print("[INFO]: Building object and anchor trajectories.", flush=True)
    current_pose_w = _current_object_pose_w(scene)
    segment_steps = tuple(args_cli.segment_steps or trajectory_module.DEFAULT_SEGMENT_STEPS)
    trajectory_w = trajectory_module.build_pick_insert_object_pose_sequence(
        current_pose_w,
        segment_steps=segment_steps,
    )
    # Anchor + hand-base from the SAME runtime function the command term uses: anchor is root-aligned
    # (object goal + offset), hand-base = anchor composed with the inverse hand-base->anchor transform.
    hand_base_to_anchor = torch.tensor(DEFAULT_HAND_BASE_TO_ANCHOR_POSE)
    base_trajectory_w, anchor_trajectory_w = hand_base_pose_from_object_command_b(
        trajectory_w, hand_base_to_anchor, DEFAULT_OBJECT_TO_ANCHOR_POSE, None
    )

    anchor_marker = _make_frame_marker("/Visuals/PickInsertObjectTrajectory/AnchorFrame")
    base_marker = _make_frame_marker("/Visuals/PickInsertObjectTrajectory/BaseFrame")

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
        _write_demo_frame(scene, trajectory_w[frame])
        anchor_marker.visualize(
            anchor_trajectory_w[frame : frame + 1, :3],
            anchor_trajectory_w[frame : frame + 1, 3:7],
        )
        base_marker.visualize(
            base_trajectory_w[frame : frame + 1, :3],
            base_trajectory_w[frame : frame + 1, 3:7],
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
