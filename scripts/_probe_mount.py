"""Probe the constant Franka-flange -> LEAP-hand `base` mount transform (calibration utility).

One-off developer tool, not part of any train/eval pipeline. In sim the arm+hand mount is fused
into the USD asset (FRANKA_LEAP_HAND_CFG), but the cuRobo MPC arm controller needs its own
kinematics URDF whose `panda_link7 -> base` fixed joint must match that mount exactly. This script
boots the HRL env, reads the live world poses of the rigidly-linked panda flange (panda_link7) and
the LEAP `base` body from the articulation, and prints their constant relative offset (xyz +
quat_wxyz) so it can be baked into the combined cuRobo URDF.

Where the printed offset goes:
    probe (this script)
      -> hand-typed into src/assets/franka_leap_hand/curobo/build_franka_leap_urdf.py --xyz --quat
      -> baked into the `panda_link7 -> base` fixed joint of franka_leap.urdf
      -> referenced by franka_leap.yml
      -> loaded by CommandHandBaseCuroboMpcAction (src/tasks/common/mdps/action_manager/curobo_mpc.py)

Re-run only when the physical mount changes (new hand offset or regenerated USD), then re-bake the
URDF so the cuRobo planner and the sim stay consistent.

    python scripts/_probe_mount.py --demo_cfg --headless --num_envs 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[1]
PICK_INSERT_TASK_DIR = REPO_ROOT / "src" / "tasks" / "pick_insert"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Probe Franka-flange -> LEAP-base mount transform.")
parser.add_argument("--task", type=str, default="Pick_Insert_HRL-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--demo_cfg", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.util  # noqa: E402
from types import ModuleType  # noqa: E402

import gymnasium as gym  # noqa: E402, F401
import isaaclab_tasks  # noqa: F401, E402
import src.tasks  # noqa: F401, E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402


def _load_module(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _make_env_cfg():
    if args_cli.demo_cfg:
        demo = _load_module("pick_insert_env_cfg_probe", PICK_INSERT_TASK_DIR / "env_cfg.py")
        env_cfg = demo.DexsuiteFrankaLeapInsertHrlEnvCfg()
        env_cfg.sim.device = args_cli.device
        env_cfg.scene.num_envs = args_cli.num_envs
        return env_cfg
    return parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)


def main() -> None:
    env = gym.make(args_cli.task, cfg=_make_env_cfg())
    e = env.unwrapped
    env.reset()
    robot = e.scene["robot"]

    print("[PROBE] all body names:", robot.body_names, flush=True)
    # The sim arm chain ends at panda_link7 (no panda_link8/panda_hand); LEAP `base` mounts on it.
    flange_name = next((b for b in ("panda_link7", "panda_link6") if b in robot.body_names), None)
    if flange_name is None:
        raise RuntimeError("No panda flange body (panda_link7) found.")
    flange_idx = robot.body_names.index(flange_name)
    base_idx = robot.body_names.index("base")

    fl_pos_w = robot.data.body_pos_w[:, flange_idx]
    fl_quat_w = robot.data.body_quat_w[:, flange_idx]
    base_pos_w = robot.data.body_pos_w[:, base_idx]
    base_quat_w = robot.data.body_quat_w[:, base_idx]

    rel_pos, rel_quat = subtract_frame_transforms(fl_pos_w, fl_quat_w, base_pos_w, base_quat_w)
    p = [round(float(v), 6) for v in rel_pos[0].tolist()]
    q = [round(float(v), 6) for v in rel_quat[0].tolist()]  # (w, x, y, z)
    print(f"[PROBE] flange link = {flange_name}", flush=True)
    print(f"[PROBE] base relative to {flange_name}:  xyz = {p}   quat_wxyz = {q}", flush=True)
    # Feed these to build_franka_leap_urdf.py --xyz / --quat; it converts the quat to the URDF
    # fixed-joint rpy (roll-pitch-yaw) internally.
    print(f"[PROBE] URDF fixed-joint origin: xyz='{p[0]} {p[1]} {p[2]}'  quat_wxyz={q}", flush=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
