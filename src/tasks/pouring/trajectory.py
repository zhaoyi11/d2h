"""Recorded bottle center poses, in demonstration world coordinates."""

from pathlib import Path

import numpy as np

from src.tasks.pick_and_place.vis_traj_isaaclab import load_object_trajectory


ASSET_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = ASSET_DIR.parent / "pick_and_place/pouring_mano_isaac_trajectory.npz"


def load_pouring_trajectory(path=DEFAULT_TRAJECTORY, stride=7, grasp_frame=0):
    """Return XYZ/WXYZ poses, source indices and reach/grasp/manipulation lengths.

    The first pose is repeated for contact confirmation before replaying the clip.
    Timing selects ordered samples only; the controller waits for tracking.
    """
    _, poses = load_object_trajectory(path)
    poses = poses.astype(np.float32)
    count = len(poses)
    if count < 2:
        raise ValueError("Trajectory must have at least two samples.")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    if not isinstance(grasp_frame, int) or not 0 <= grasp_frame < count - 1:
        raise ValueError("grasp_frame must be nonnegative and precede the final frame.")
    frames = np.unique(np.r_[np.arange(0, count, stride), grasp_frame, count - 1])
    grasp_index = int(np.searchsorted(frames, grasp_frame))
    frames = np.insert(frames, grasp_index + 1, grasp_frame)
    segments = (grasp_index, 1, len(frames) - grasp_index - 2)
    return poses[frames], frames, segments
