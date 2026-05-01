from __future__ import annotations

import math as _math
from typing import Literal

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor


_FINGERTIP_SENSOR_NAMES = [
    "thumb_tip_object_s",
    "index_tip_object_s",
    "middle_tip_object_s",
    "ring_tip_object_s",
]


def _compute_contact_metrics(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str,
    force_threshold: float,
    contact_pose_range_deg: float,
) -> dict[str, torch.Tensor]:
    """Shared layer: per-fingertip forces rotated into fingertip frame, theta/phi, contact masks.

    Returns dict with keys:
      in_contact   (num_envs, n_fingers) bool
      force_mag    (num_envs, n_fingers) float
      contact_pose (num_envs, n_fingers, 2) float  — [theta, phi] clamped to pose_limit
    """
    from isaaclab.utils.math import quat_apply_inverse

    pose_limit = _math.radians(contact_pose_range_deg)
    # (num_envs, n_fingers, 4) world-frame fingertip orientations as (w, x, y, z)
    target_quat_w = env.scene.sensors[fingertip_transforms_name].data.target_quat_w

    in_contact_list, force_mag_list, pose_list = [], [], []

    for i, name in enumerate(contact_sensor_names):
        sensor = env.scene.sensors[name]
        fm = sensor.data.force_matrix_w

        if fm is None or fm.numel() == 0:
            zeros = torch.zeros(env.num_envs, device=env.device)
            in_contact_list.append(zeros.bool())
            force_mag_list.append(zeros)
            pose_list.append(torch.zeros(env.num_envs, 2, device=env.device))
            continue

        F_world = torch.nan_to_num(fm, nan=0.0).sum(dim=(1, 2))        # (N, 3)
        F_mag = torch.linalg.norm(F_world, dim=-1)                      # (N,)
        in_contact = F_mag > force_threshold

        # Rotate reaction force into fingertip frame.
        # Frame convention: red (X) = object-to-finger (reaction direction),
        # so theta=0, phi=0 at centered contact without negation.
        F_tip = quat_apply_inverse(target_quat_w[:, i, :], F_world)  # (N, 3)
        F_tip_mag = F_mag  # rotation preserves norm

        # Spherical coordinates in fingertip frame: theta=0, phi=0 when
        # reaction is along +X (red axis = fingertip outward normal).
        theta = torch.where(
            F_tip_mag < force_threshold,
            torch.zeros_like(F_tip_mag),
            torch.atan2(F_tip[:, 1], F_tip[:, 0]),
        )
        phi = torch.where(
            F_tip_mag < force_threshold,
            torch.zeros_like(F_tip_mag),
            torch.acos(torch.clamp(F_tip[:, 2] / (F_tip_mag + 1e-8), -1.0, 1.0)) - torch.pi / 2,
        )

        pose = torch.clamp(torch.stack([theta, phi], dim=-1), -pose_limit, pose_limit)

        in_contact_list.append(in_contact)
        force_mag_list.append(F_mag)
        pose_list.append(pose)

    return {
        "in_contact":   torch.stack(in_contact_list, dim=1),   # (N, n)
        "force_mag":    torch.stack(force_mag_list, dim=1),     # (N, n)
        "contact_pose": torch.stack(pose_list, dim=1),          # (N, n, 2)
    }


def good_contact_count(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str = "fingertip_transforms",
    force_threshold: float = 0.25,
    contact_pose_range_deg: float = 50.0,
) -> torch.Tensor:
    """Count fingertips whose contact force is above threshold AND within the angular pose limit."""
    m = _compute_contact_metrics(
        env, contact_sensor_names, fingertip_transforms_name, force_threshold, contact_pose_range_deg
    )
    good_pose = (torch.abs(m["contact_pose"]) < _math.radians(contact_pose_range_deg)).all(dim=-1)
    return (m["in_contact"] & good_pose).float().sum(dim=-1)    # (N,)


def contacts(env: ManagerBasedRLEnv, threshold: float, mode: Literal["opposite", "any"] = "opposite") -> torch.Tensor:
    """Check if the fingertip contacts with the object is above a threshold."""
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_tip_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_tip_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_tip_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_tip_object_s"]
    # check if contact force is above threshold
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    if mode == "opposite":
        contact_cond = (thumb_contact_mag > threshold) & (
            (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
        )
    if mode == "any":
        finger_contacts = torch.stack([
            thumb_contact_mag > threshold,
            index_contact_mag > threshold,
            middle_contact_mag > threshold,
            ring_contact_mag > threshold,
        ], dim=-1)  # (num_envs, 4)
        contact_cond = finger_contacts.sum(dim=-1) >= 2
    return contact_cond
