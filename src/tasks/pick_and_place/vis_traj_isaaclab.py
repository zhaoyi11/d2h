"""Replay object poses and available hand keypoints in IsaacLab, without scene.xml.

Run in env_isaaclab: python src/tasks/pick_and_place/vis_traj_isaaclab.py --loop
The original NPZ stays unchanged. Converted meshes are cached in /tmp/d2h_tape_measure_usd.
Object scale changes only mesh size, not trajectory positions; the pig mesh uses its native scale.
CPU physics is used for pose updates; rendering still uses the GPU.
Keypoint recordings without timing metadata play at 30 FPS; use --fps to override.
MANO joint recordings use MuJoCo and the bundled hand model to compute skeleton points in memory.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tasks.common.trajectory import load_object_trajectory, positive_float


def frame_index(timestamps, elapsed, loop=False):
    """Select the recorded sample at elapsed time; retain the final sample interval when looping."""
    if loop and len(timestamps) > 1:
        elapsed %= timestamps[-1] + (timestamps[-1] - timestamps[-2])
    return int(np.clip(np.searchsorted(timestamps, elapsed, side="right") - 1, 0, len(timestamps) - 1))


def main(argv=None):
    from isaaclab.app import AppLauncher

    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=directory / "pick_and_place_pig_mano_cropped_isaac.npz")
    parser.add_argument("--mesh", type=Path, default=directory / "pick_and_place_pig/pick_and_place/visual.obj")
    parser.add_argument("--object-scale", type=positive_float, default=1.0)
    parser.add_argument("--fps", type=positive_float, help="Override playback rate (keypoint fallback: 30 FPS)")
    parser.add_argument("--loop", action="store_true", help="Repeat instead of holding the final pose")
    AppLauncher.add_app_launcher_args(parser)
    # GPU PhysX pose writes fail for this collider-disabled kinematic scene in the installed simulator.
    parser.set_defaults(device="cpu")
    args = parser.parse_args(argv)
    timestamps, poses = load_object_trajectory(args.trajectory, args.fps)
    points = None
    with np.load(args.trajectory, allow_pickle=False) as recorded:
        if "qpos_wrist_right" in recorded or (
            "robot_joint_names" in recorded and "right_j_thumb1x" in recorded["robot_joint_names"]
        ):
            from vis_traj import FINGER_COLORS, load_trajectory, skeleton_edges

            _, points = load_trajectory(args.trajectory)
            edges = skeleton_edges(points.shape[1])
    simulation_app = AppLauncher(args).app
    sim = None
    try:
        import torch
        from pxr import Usd, UsdGeom, Vt

        import isaaclab.sim as sim_utils
        from isaaclab.assets import RigidObject, RigidObjectCfg
        from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))
        mesh = MeshConverter(MeshConverterCfg(
            asset_path=str(args.mesh.resolve()),
            usd_dir="/tmp/d2h_tape_measure_usd",
            make_instanceable=False,
            scale=(args.object_scale,) * 3,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ))
        obj = RigidObject(RigidObjectCfg(
            prim_path="/World/Object",
            spawn=sim_utils.UsdFileCfg(usd_path=mesh.usd_path),
            init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(poses[0, :3]), rot=tuple(poses[0, 3:])),
        ))
        if points is not None:
            skeleton = UsdGeom.BasisCurves.Define(sim.stage, "/World/HandSkeleton")
            skeleton.CreateTypeAttr("linear")
            skeleton.CreateWrapAttr("nonperiodic")
            skeleton.CreateCurveVertexCountsAttr([2] * len(edges))
            skeleton.CreateWidthsAttr([0.003])
            skeleton.SetWidthsInterpolation("constant")
            colors = np.tile(FINGER_COLORS / 255., (len(edges) // 5, 1)).astype(np.float32)
            skeleton.CreateDisplayColorAttr(Vt.Vec3fArray.FromNumpy(colors))
            skeleton.GetDisplayColorPrimvar().SetInterpolation("uniform")
            bone_points = points[:, edges].reshape(len(points), -1, 3).astype(np.float32)
            skeleton.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(bone_points[0]))
        light = sim_utils.DomeLightCfg(intensity=2000.0)
        light.func("/World/Light", light)

        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
            sim.stage.GetPrimAtPath("/World/Object")
        ).ComputeAlignedRange()
        corners = np.array([bounds.GetMin(), bounds.GetMax()])
        radius = np.linalg.norm(np.max(np.abs(corners - poses[0, :3]), axis=0))
        lower, upper = poses[:, :3].min(axis=0) - radius, poses[:, :3].max(axis=0) + radius
        if points is not None:
            lower = np.minimum(lower, points.min(axis=(0, 1)) - .003)
            upper = np.maximum(upper, points.max(axis=(0, 1)) + .003)
        # Keep the display floor below the full trajectory without changing recorded poses.
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/Ground", ground, translation=(0., 0., min(0., float(lower[2]) - .01)))
        center = (lower + upper) / 2
        extent = max(float(np.linalg.norm(upper - lower)), 0.3)
        sim.set_camera_view(eye=center + extent * np.array([1., -1., .7]), target=center)

        sim.reset()
        obj.reset()
        pose_tensor = torch.as_tensor(poses, dtype=torch.float32, device=sim.device)
        hand_info = f" and {points.shape[1]} hand points" if points is not None else ""
        print(f"Loaded {len(poses)} object poses{hand_info}, spanning {timestamps[-1]:.3f} s. Close the window to exit.", flush=True)
        start = time.monotonic()
        while simulation_app.is_running():
            tick = time.monotonic()
            frame = frame_index(timestamps, tick - start, args.loop)
            obj.write_root_pose_to_sim(pose_tensor[frame:frame + 1])
            if points is not None:
                skeleton.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(bone_points[frame]))
            # Rendering flushes pose updates without advancing physics or altering the recorded motion.
            sim.render()
            time.sleep(max(0., 1 / 60 - (time.monotonic() - tick)))
    finally:
        if sim is not None:
            sim.clear_all_callbacks()
            sim.clear_instance()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
