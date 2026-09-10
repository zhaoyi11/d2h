"""Adapt the recorded pig carry to a live tabletop start and box target."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from src.tasks.pick_and_place.vis_traj_isaaclab import load_object_trajectory


ASSET_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = ASSET_DIR / "pick_and_place_pig_mano_cropped_isaac.npz"
PIG_MESH = ASSET_DIR / "pick_and_place_pig/pick_and_place/visual.obj"
DEPOSIT_STEPS = 10


def load_carry_trajectory(path=DEFAULT_TRAJECTORY, start_frame=320, end_frame=511, stride=28):
    """Read original XYZ/WXYZ poses, retaining both ends of the lift/carry interval."""
    _, poses = load_object_trajectory(path)
    if any(not isinstance(value, int) for value in (start_frame, end_frame, stride)):
        raise ValueError("Carry frame indices and stride must be integers.")
    if not 0 <= start_frame < end_frame < len(poses) or stride < 1:
        raise ValueError("Expected 0 <= start_frame < end_frame < frame count and stride > 0.")
    frames = np.unique(np.r_[np.arange(start_frame, end_frame + 1, stride), end_frame])
    carry = poses[frames]
    displacement = carry[-1, :3] - carry[0, :3]
    if np.linalg.norm(displacement[:2]) < 1e-6 or displacement[2] <= 1e-6:
        raise ValueError("The carry interval must rise and have horizontal displacement.")
    segments = (0, 1, len(carry) - 1, DEPOSIT_STEPS, 1, 1)
    # Synthetic deposit/release/retreat targets have no source frame.
    source_frames = np.r_[frames[0], frames, np.full(DEPOSIT_STEPS + 2, -1)]
    return carry, source_frames, segments


def build_pick_and_place_trajectory(
    carry, current_pose_w, box_pose_w,
    above_box_offset=(0.0, 0.0, 0.25), box_target_offset=(0.0, 0.0, 0.10),
):
    """Return reach/grasp/carry/deposit/release/retreat poses in world coordinates.

    Fit horizontal motion with a yaw and uniform XY scale, and fit height with a
    separate Z scale. Relative object rotations begin at the live settled pose.
    Called only when constructing/resetting a trajectory, not each control step.
    """
    box_rotation = Rotation.from_quat(box_pose_w[3:], scalar_first=True)
    above = box_pose_w[:3] + box_rotation.apply(above_box_offset)
    deposit = box_pose_w[:3] + box_rotation.apply(box_target_offset)
    relative = carry[:, :3] - carry[0, :3]
    source = relative[-1]
    target = above - current_pose_w[:3]
    if target[2] <= 0:
        raise ValueError("The above-box target must be higher than the settled object.")
    yaw = np.arctan2(target[1], target[0]) - np.arctan2(source[1], source[0])
    c, s = np.cos(yaw), np.sin(yaw)
    positions = relative.copy()
    positions[:, :2] = relative[:, :2] @ np.array([[c, s], [-s, c]])
    positions[:, :2] *= np.linalg.norm(target[:2]) / np.linalg.norm(source[:2])
    positions[:, 2] *= target[2] / source[2]
    positions += current_pose_w[:3]

    recorded_rotations = Rotation.from_quat(carry[:, 3:], scalar_first=True)
    live_rotation = Rotation.from_quat(current_pose_w[3:], scalar_first=True)
    rotations = live_rotation * recorded_rotations[0].inv() * recorded_rotations
    transported = np.c_[positions, rotations.as_quat(scalar_first=True)]
    alpha = np.linspace(0.0, 1.0, DEPOSIT_STEPS + 1)[1:, None]
    deposited = np.c_[above + alpha * (deposit - above),
                      np.tile(transported[-1, 3:], (DEPOSIT_STEPS, 1))]
    return np.concatenate((current_pose_w[None], transported, deposited,
                           deposited[[-1]], deposited[[-1]]))
