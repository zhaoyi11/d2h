"""Export a MANO joint-state recording to world-frame skeleton keypoints.

Requires MuJoCo only for conversion. Example:
    python export_hand_keypoints.py pouring_mano_isaac_trajectory.npz --scene right.xml

The XML can be the original scene or the upstream MANO hand model:
https://github.com/malik-group/do-as-i-do/blob/824591b808c342b20079c3b4198a8c2bdf88c74e/retargeting/retargeting/assets/robots/mano/right.xml
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def load_hand_keypoints(trajectory, scene):
    """Compute world-frame skeleton arrays in memory from a MANO recording."""
    import mujoco

    trajectory, scene = Path(trajectory), Path(scene)
    with np.load(trajectory, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    names = arrays["robot_joint_names"].tolist()
    qpos = arrays["robot_joint_pos"]
    if qpos.ndim != 2 or qpos.shape[1] != len(names) or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError("robot_joint_pos must be finite with shape (N, number of joint names)")
    if len(set(names)) != len(names):
        raise ValueError("robot_joint_names must be unique")
    for key, size in (("hand_root_pos", 3), ("hand_root_quat_wxyz", 4), ("object_pos", 3), ("object_quat_wxyz", 4)):
        if arrays[key].shape != (len(qpos), size) or not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} must be finite with shape ({len(qpos)}, {size})")

    # Keep the floating MANO hand and its kinematics; visual meshes are unnecessary.
    xml_bytes = scene.read_bytes()
    source_xml = ET.fromstring(xml_bytes)
    palm = source_xml.find("./worldbody/body[@name='right_palm']")
    if palm is None:
        raise ValueError("Expected the floating MANO hand body 'right_palm' under worldbody")
    root = ET.Element("mujoco")
    for tag in ("compiler", "default"):
        node = source_xml.find(tag)
        if node is not None:
            root.append(node)
    ET.SubElement(root, "worldbody").append(palm)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and (child.get("type") == "mesh" or "mesh" in child.attrib):
                parent.remove(child)
            child.attrib.pop("material", None)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if set(names) != {model.joint(i).name for i in range(model.njnt)}:
        raise ValueError("Recording joint names must match all joints in the MANO hand model")
    if not np.isin(model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all():
        raise ValueError("Expected scalar hinge/slide joints in the MANO hand model")
    addresses = [model.joint(name).qposadr[0] for name in names]
    mcp_joints = [model.joint(f"right_j_{finger}1{'x' if finger == 'thumb' else 'y'}").id for finger in FINGERS]
    sites = {
        level: [model.site(f"right_{finger}_{level}").id for finger in FINGERS]
        for level in ("pip", "dip", "tip")
    }
    keypoints = {level: np.empty((len(qpos), 5, 3), dtype=np.float32) for level in ("mcp", *sites)}
    data = mujoco.MjData(model)
    for i, joint_pos in enumerate(qpos):
        data.qpos[addresses] = joint_pos
        mujoco.mj_forward(model, data)
        palm_pose = data.body("right_palm")
        recorded_quat = arrays["hand_root_quat_wxyz"][i].astype(np.float64)
        norm = np.linalg.norm(recorded_quat)
        if norm < 1e-12 or not np.isfinite(norm):
            raise ValueError(f"Invalid hand root quaternion at frame {i}")
        if not np.allclose(palm_pose.xpos, arrays["hand_root_pos"][i], atol=1e-5, rtol=0) or abs(
            np.dot(palm_pose.xquat, recorded_quat / norm)
        ) < 1 - 1e-6:
            raise ValueError(f"Hand model does not reproduce the recorded wrist pose at frame {i}")
        keypoints["mcp"][i] = data.xanchor[mcp_joints]
        for level, site_ids in sites.items():
            keypoints[level][i] = data.site_xpos[site_ids]

    # Reuse the viewer's keypoint format; fingertip orientation is unused.
    tip_poses = np.zeros((len(qpos), 5, 7), dtype=np.float32)
    tip_poses[..., :3] = keypoints["tip"]
    tip_poses[..., 3] = 1
    arrays.update(
        qpos_obj_right=np.concatenate((arrays["object_pos"], arrays["object_quat_wxyz"]), axis=1),
        qpos_wrist_right=np.concatenate((arrays["hand_root_pos"], arrays["hand_root_quat_wxyz"]), axis=1),
        qpos_mcp_right=keypoints["mcp"],
        qpos_pip_right=keypoints["pip"],
        qpos_dip_right=keypoints["dip"],
        qpos_finger_right=tip_poses,
        keypoint_export_metadata_json=np.array(json.dumps({
            "model_path": str(scene.resolve()),
            "model_sha256": hashlib.sha256(xml_bytes).hexdigest(),
            "mujoco_version": mujoco.__version__,
            "coordinate_system": "right-handed, Z-up, metres",
            "finger_order": FINGERS,
            "method": "mj_forward; MCP joint anchors and PIP/DIP/tip sites",
        })),
    )
    return arrays


def export_keypoints(trajectory, scene, output):
    arrays = load_hand_keypoints(trajectory, scene)
    output = Path(output)
    # Exclusive creation preserves both the input and any existing export.
    with output.open("xb") as file:
        np.savez_compressed(file, **arrays)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.trajectory.with_name(f"{args.trajectory.stem}_keypoints.npz")
    print(f"Exported skeleton keypoints: {export_keypoints(args.trajectory, args.scene, output)}")
