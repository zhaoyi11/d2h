"""Replay the pouring recording and MANO hand skeleton in IsaacLab.

Run in env_isaaclab: python src/tasks/pouring/vis_traj_isaaclab.py --loop
"""

import sys
from pathlib import Path


directory = Path(__file__).resolve().parent
sys.path.insert(0, str(directory.parent / "pick_and_place"))
from vis_traj_isaaclab import main


if __name__ == "__main__":
    try:
        main([
            "--trajectory", str(directory.parent / "pick_and_place/pouring_mano_isaac_trajectory.npz"),
            "--mesh", str(directory.parent / "pick_and_place/bottle/bottle.obj"),
            "--object-scale", "0.2",
            *sys.argv[1:],
        ])
    except KeyboardInterrupt:
        pass
