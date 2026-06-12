"""Generate a kinematics-only combined Franka+LEAP URDF for cuRobo.

Merges cuRobo's Franka arm (panda_link0..7, panda_joint1..7) with the LEAP right-hand URDF,
joined by a fixed joint panda_link7 -> base using the measured flange->hand-base mount transform.
Visual/collision (mesh) tags are stripped so cuRobo parses pure kinematics (collision is supplied
separately as spheres in franka_leap.yml). Inertials are kept.

Usage:
    python build_franka_leap_urdf.py --xyz X Y Z --quat W X Y Z [--out franka_leap.urdf]

The mount transform comes from scripts/_probe_mount.py (base relative to panda_link7).
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

FRANKA_URDF = Path(
    "/home/yizhao/yi/curobo/curobo/content/assets/robot/franka_description/franka_panda.urdf"
)
LEAP_URDF = Path("/home/yizhao/yi/dex-urdf/robots/hands/leap_hand/leap_hand_right.urdf")

FRANKA_KEEP_LINKS = ["base_link", *[f"panda_link{i}" for i in range(8)]]  # link0..7 + base_link
FRANKA_KEEP_JOINTS = ["panda_fixed", *[f"panda_joint{i}" for i in range(1, 8)]]


def quat_wxyz_to_rpy(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert a (w,x,y,z) quaternion to URDF roll-pitch-yaw (XYZ fixed-axis)."""
    # roll (x), pitch (y), yaw (z)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def strip_meshes(link: ET.Element) -> ET.Element:
    """Remove <visual> and <collision> (mesh-bearing) children; keep <inertial>."""
    for tag in ("visual", "collision"):
        for child in link.findall(tag):
            link.remove(child)
    return link


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xyz", type=float, nargs=3, required=True, help="mount translation x y z")
    ap.add_argument("--quat", type=float, nargs=4, required=True, help="mount quaternion w x y z")
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "franka_leap.urdf"))
    args = ap.parse_args()

    franka = ET.parse(FRANKA_URDF).getroot()
    leap = ET.parse(LEAP_URDF).getroot()

    robot = ET.Element("robot", {"name": "franka_leap"})

    # Franka arm: kept links (mesh-stripped) + joints
    for link in franka.findall("link"):
        if link.get("name") in FRANKA_KEEP_LINKS:
            robot.append(strip_meshes(link))
    for joint in franka.findall("joint"):
        if joint.get("name") in FRANKA_KEEP_JOINTS:
            robot.append(joint)

    # Mount joint: panda_link7 -> LEAP base
    roll, pitch, yaw = quat_wxyz_to_rpy(*args.quat)
    mount = ET.SubElement(robot, "joint", {"name": "panda_link7_to_leap_base", "type": "fixed"})
    ET.SubElement(mount, "parent", {"link": "panda_link7"})
    ET.SubElement(mount, "child", {"link": "base"})
    ET.SubElement(
        mount,
        "origin",
        {"xyz": f"{args.xyz[0]} {args.xyz[1]} {args.xyz[2]}", "rpy": f"{roll} {pitch} {yaw}"},
    )

    # LEAP hand: all links (mesh-stripped) + all joints
    for link in leap.findall("link"):
        robot.append(strip_meshes(link))
    for joint in leap.findall("joint"):
        robot.append(joint)

    tree = ET.ElementTree(robot)
    ET.indent(tree, space="  ")
    tree.write(args.out, encoding="utf-8", xml_declaration=True)
    n_links = len(robot.findall("link"))
    n_joints = len(robot.findall("joint"))
    print(f"Wrote {args.out}  ({n_links} links, {n_joints} joints)")


if __name__ == "__main__":
    main()
