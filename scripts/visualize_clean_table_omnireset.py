"""Visualize one frozen reset state from the clean-table OmniReset dataset."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Visualize one frozen clean-table OmniReset state.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of reset states to visualize.")
parser.add_argument(
    "--reset_dataset_dir",
    type=str,
    default=None,
    help="Optional override for the Instant Dexterity reset-state dataset.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import omni.timeline  # noqa: E402
import src.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


TASK_ID = "Clean_Table_OmniReset-v0"


def main() -> None:
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.reset_dataset_dir is not None:
        env_cfg.reset_dataset_dir = args_cli.reset_dataset_dir

    env = gym.make(TASK_ID, cfg=env_cfg)
    try:
        env.reset()
        omni.timeline.get_timeline_interface().pause()
        print("[INFO]: Reset state frozen. Close Isaac Sim to exit.")
        while simulation_app.is_running():
            simulation_app.update()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
