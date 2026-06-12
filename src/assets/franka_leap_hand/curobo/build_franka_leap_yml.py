"""Generate cuRobo robot config `franka_leap.yml` for the combined Franka+LEAP URDF.

Arm collision spheres are taken from cuRobo's shipped franka.yml (panda_link0..7). LEAP hand
spheres are derived from the LEAP URDF collision boxes (each box covered by spheres along its
longest axis). Finger joints are locked (rigid hand envelope); the 7 panda joints are the cspace.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

HERE = Path(__file__).parent
LEAP_URDF = Path("/home/yizhao/yi/dex-urdf/robots/hands/leap_hand/leap_hand_right.urdf")
FRANKA_YML = Path("/home/yizhao/yi/curobo/curobo/content/configs/robot/franka.yml")
URDF_PATH = HERE / "franka_leap.urdf"
OUT = HERE / "franka_leap.yml"

LEAP_COLLISION_LINKS = [
    "palm_lower",
    "mcp_joint", "pip", "dip", "fingertip",
    "mcp_joint_2", "pip_2", "dip_2", "fingertip_2",
    "mcp_joint_3", "pip_3", "dip_3", "fingertip_3",
    "thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip",
]
LEAP_JOINTS = [str(i) for i in range(16)]  # URDF revolute joint names of the LEAP hand


def boxes_to_spheres(size, origin, max_per_box: int = 2):
    """Cover an axis-aligned box (in link frame) with 1..max_per_box spheres along its long axis."""
    sx, sy, sz = size
    ox, oy, oz = origin
    dims = [sx, sy, sz]
    long_axis = max(range(3), key=lambda i: dims[i])
    smid = sorted(dims)[1]
    radius = round(max(0.008, 0.5 * smid), 4)  # half the middle dimension, floored
    length = dims[long_axis]
    n = max(1, min(max_per_box, int(round(length / (2 * radius)))))
    spheres = []
    if n == 1:
        spheres.append({"center": [round(ox, 4), round(oy, 4), round(oz, 4)], "radius": radius})
    else:
        span = length - 2 * radius
        for k in range(n):
            t = -span / 2 + span * k / (n - 1)
            c = [ox, oy, oz]
            c[long_axis] += t
            spheres.append({"center": [round(v, 4) for v in c], "radius": radius})
    return spheres


def leap_spheres():
    root = ET.parse(LEAP_URDF).getroot()
    out = {}
    for link in root.findall("link"):
        name = link.get("name")
        if name not in LEAP_COLLISION_LINKS:
            continue
        sph = []
        for col in link.findall("collision"):
            g = col.find("geometry")
            box = g.find("box") if g is not None else None
            if box is None:
                continue  # skip the few mesh-based tip collisions; boxes dominate
            size = [float(v) for v in box.get("size").split()]
            o = col.find("origin")
            origin = [float(v) for v in (o.get("xyz").split() if o is not None else ["0", "0", "0"])]
            sph.extend(boxes_to_spheres(size, origin))
        out[name] = sph
    return out


def main():
    franka = yaml.safe_load(open(FRANKA_YML))["robot_cfg"]["kinematics"]
    arm_links = [f"panda_link{i}" for i in range(8)]
    arm_spheres = {k: v for k, v in franka["collision_spheres"].items() if k in arm_links}

    hand_spheres = leap_spheres()
    collision_spheres = {**arm_spheres, **hand_spheres}
    collision_link_names = arm_links[:8] + LEAP_COLLISION_LINKS

    # Self-collision ignore. The LEAP hand joints are LOCKED, so the whole hand is a single rigid
    # body and can never self-collide: ignore every intra-hand pair and wrist(link5/6/7)-vs-hand.
    # Keep the franka arm-chain ignores; distant arm links vs hand stay checked.
    self_collision_ignore = {
        "panda_link0": ["panda_link1", "panda_link2"],
        "panda_link1": ["panda_link2", "panda_link3", "panda_link4"],
        "panda_link2": ["panda_link3", "panda_link4"],
        "panda_link3": ["panda_link4", "panda_link6"],
        "panda_link4": ["panda_link5", "panda_link6", "panda_link7"],
        "panda_link5": ["panda_link6", "panda_link7"] + LEAP_COLLISION_LINKS,
        "panda_link6": ["panda_link7"] + LEAP_COLLISION_LINKS,
        "panda_link7": LEAP_COLLISION_LINKS,
    }
    # every hand link ignores every other hand link + the wrist links it's mounted near
    for h in LEAP_COLLISION_LINKS:
        self_collision_ignore[h] = [x for x in LEAP_COLLISION_LINKS if x != h] + [
            "panda_link5",
            "panda_link6",
            "panda_link7",
        ]

    cfg = {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": str(URDF_PATH),
                "asset_root_path": str(HERE),
                "base_link": "panda_link0",
                "tool_frames": ["base"],
                "collision_link_names": collision_link_names,
                "collision_spheres": collision_spheres,
                "collision_sphere_buffer": 0.005,
                "self_collision_ignore": self_collision_ignore,
                "self_collision_buffer": {k: 0.0 for k in collision_link_names},
                "lock_joints": {j: 0.0 for j in LEAP_JOINTS},
                "cspace": {
                    "joint_names": [f"panda_joint{i}" for i in range(1, 8)],
                    "null_space_weight": [1] * 7,
                    "cspace_distance_weight": [1] * 7,
                    "max_acceleration": 15.0,
                    "max_jerk": 500.0,
                    "default_joint_position": [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741],
                },
            },
            "load_dynamics": False,
        }
    }
    with open(OUT, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None, sort_keys=False)
    n_sph = sum(len(v) for v in collision_spheres.values())
    print(f"Wrote {OUT}  ({len(collision_link_names)} collision links, {n_sph} spheres)")


if __name__ == "__main__":
    main()
