"""Play a recorded object and hand skeleton in Viser.

Run with env_isaaclab active: python src/tasks/pick_and_place/vis_traj.py
Original MANO recordings require MuJoCo (python -m pip install mujoco==3.11.0).
Their skeleton is computed in memory at startup using --scene; no NPZ is written.
"""

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np


FINGER_COLORS = np.array(
    [(255, 100, 100), (255, 190, 60), (100, 220, 120), (80, 170, 255), (200, 120, 255)],
    dtype=np.uint8,
)
DEFAULT_SCENE = Path(__file__).resolve().with_name("mano_right.xml")


def load_trajectory(path, scene=DEFAULT_SCENE):
    """Return object XYZ/WXYZ poses and 16 or 21 world-frame hand points."""
    shapes = {
        "qpos_obj_right": (7,),
        "qpos_wrist_right": (7,),
        "qpos_pip_right": (5, 3),
        "qpos_dip_right": (5, 3),
        "qpos_finger_right": (5, 7),
    }
    arrays = {}
    with np.load(path, allow_pickle=False) as source:
        data = source
        if "robot_joint_pos" in source and "qpos_wrist_right" not in source:
            from export_hand_keypoints import load_hand_keypoints

            data = load_hand_keypoints(path, scene)
        if "qpos_mcp_right" in data:
            shapes["qpos_mcp_right"] = (5, 3)
        for key, shape in shapes.items():
            if key not in data:
                raise ValueError(f"Missing trajectory array: {key}")
            value = data[key]
            if value.ndim != len(shape) + 1 or value.shape[1:] != shape or len(value) == 0:
                raise ValueError(f"{key} must have shape (N, {', '.join(map(str, shape))}) with N > 0")
            if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                raise ValueError(f"{key} must contain finite numeric values")
            arrays[key] = value.astype(np.float64)

    poses = arrays["qpos_obj_right"]
    if any(len(value) != len(poses) for value in arrays.values()):
        raise ValueError("Trajectory arrays must have matching frame counts")
    for key in ("qpos_obj_right", "qpos_wrist_right", "qpos_finger_right"):
        norms = np.linalg.norm(arrays[key][..., 3:], axis=-1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms < 1e-12):
            raise ValueError(f"{key} contains invalid quaternions")
        arrays[key][..., 3:] /= norms

    groups = [arrays["qpos_wrist_right"][:, None, :3]]
    if "qpos_mcp_right" in arrays:
        groups.append(arrays["qpos_mcp_right"])
    groups.extend((arrays["qpos_pip_right"], arrays["qpos_dip_right"], arrays["qpos_finger_right"][..., :3]))
    points = np.concatenate(groups, axis=1)
    return poses, points


def skeleton_edges(point_count):
    """Connect a wrist followed by groups of five finger landmarks."""
    return np.array([(0, i) for i in range(1, 6)] + [(i, i + 5) for i in range(1, point_count - 5)])


def playback_fps(path, override=None):
    with np.load(path, allow_pickle=False) as data:
        recorded = float(data["frequency"]) if "frequency" in data else 30.0
    return positive_float(recorded if override is None else override)


def positive_float(value):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return value


def main(args):
    import trimesh
    import viser

    poses, points = load_trajectory(args.trajectory, args.scene)
    edges = skeleton_edges(points.shape[1])
    levels = (points.shape[1] - 1) // 5
    initial_fps = playback_fps(args.trajectory, args.fps)
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"No triangle mesh found in {args.mesh}")

    server = viser.ViserServer(host="127.0.0.1", port=args.port, label="Hand and object trajectory")
    server.scene.set_up_direction("+z")
    object_handle = server.scene.add_mesh_trimesh(
        "/object", mesh, scale=args.object_scale, position=poses[0, :3], wxyz=poses[0, 3:]
    )
    skeleton = server.scene.add_line_segments(
        "/hand/bones",
        points=points[0, edges],
        colors=np.tile(FINGER_COLORS, (levels, 1))[:, None, :].repeat(2, axis=1),
        line_width=4.0,
    )
    joints = server.scene.add_point_cloud(
        "/hand/joints",
        points=points[0],
        colors=np.concatenate(([[240, 240, 240]], np.tile(FINGER_COLORS, (levels, 1)))).astype(np.uint8),
        point_size=0.006,
        point_shape="circle",
        precision="float32",
    )

    playing = server.gui.add_checkbox("Play", initial_value=False)
    looping = server.gui.add_checkbox("Loop", initial_value=True)
    frame = server.gui.add_slider(
        "Frame", min=0, max=max(1, len(poses) - 1), step=1, initial_value=0, disabled=len(poses) == 1
    )
    fps = server.gui.add_number("FPS", initial_value=initial_fps, min=0.01, step=1.0)
    scale = server.gui.add_number("Object scale", initial_value=args.object_scale, min=0.0001, step=0.005)
    server.gui.add_markdown(
        f"{len(poses)} frames · {points.shape[1]} hand points\n\n"
        + ("Wrist → MCP → PIP → DIP → tip" if levels == 4 else "Wrist → PIP → DIP → tip (MCPs unavailable)")
        + "\n\nObject scale is an adjustable estimate."
    )

    mesh_radius = np.linalg.norm(mesh.vertices, axis=1).max() * args.object_scale
    bounds = np.concatenate((points.reshape(-1, 3), poses[:, :3] - mesh_radius, poses[:, :3] + mesh_radius))
    center = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
    extent = max(float(np.linalg.norm(np.ptp(bounds, axis=0))), 0.1)

    @server.on_client_connect
    async def on_connect(client):
        client.camera.position = center + extent * np.array([1.0, -1.0, 0.7])
        client.camera.look_at = center
        client.camera.up_direction = (0.0, 0.0, 1.0)

    @frame.on_update
    async def on_scrub(event):
        if event.client is not None:
            playing.value = False

    last_tick = time.monotonic()

    @playing.on_update
    @fps.on_update
    async def reset_clock(_):
        nonlocal last_tick
        last_tick = time.monotonic()

    print(f"Loaded {len(poses)} frames. Open http://localhost:{server.get_port()}", flush=True)

    async def playback():
        nonlocal last_tick
        displayed_frame = 0
        while True:
            now = time.monotonic()
            if playing.value:
                elapsed_frames = int((now - last_tick) * fps.value)
                if elapsed_frames:
                    next_frame = frame.value + elapsed_frames
                    frame.value = next_frame % len(poses) if looping.value else min(next_frame, len(poses) - 1)
                    if not looping.value and next_frame >= len(poses) - 1:
                        playing.value = False
                    last_tick += elapsed_frames / fps.value
            if frame.value != displayed_frame:
                displayed_frame = frame.value
                with server.atomic():
                    object_handle.position = poses[displayed_frame, :3]
                    object_handle.wxyz = poses[displayed_frame, 3:]
                    skeleton.points = points[displayed_frame, edges]
                    joints.points = points[displayed_frame]
            object_handle.scale = scale.value
            await asyncio.sleep(0.005)

    # Keep playback and GUI callbacks on Viser's event loop.
    future = asyncio.run_coroutine_threadsafe(playback(), server.get_event_loop())
    try:
        future.result()
    finally:
        future.cancel()
        server.stop()


if __name__ == "__main__":
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=directory / "23963_mano_isaac_trajectory.npz")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="MANO hand XML for joint-state recordings")
    parser.add_argument("--mesh", type=Path, default=directory / "tape_measure/tape_measure.obj")
    parser.add_argument("--fps", type=positive_float, help="Override recorded frequency (30 FPS if absent)")
    parser.add_argument("--object-scale", type=positive_float, default=0.1)
    parser.add_argument("--port", type=int, default=8080)
    try:
        main(parser.parse_args())
    except KeyboardInterrupt:
        pass
