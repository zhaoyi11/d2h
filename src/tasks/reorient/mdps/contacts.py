from __future__ import annotations

import math as _math
from typing import Literal

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor

from src.tasks.common.mdps.contacts import _compute_contact_metrics


_FINGERTIP_SENSOR_NAMES = [
    "thumb_tip_object_s",
    "index_tip_object_s",
    "middle_tip_object_s",
    "ring_tip_object_s",
]


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


def _object_contact_force(
    sensor: ContactSensor, num_envs: int, device: torch.device
) -> torch.Tensor:
    """Per-env object contact force (N, 3) for a fingertip sensor.

    Selects only the object's contact-filter index (0), so it stays correct when external
    contact targets (receptacle, table, ...) add more filters to the sensor.
    """
    fm = sensor.data.force_matrix_w
    obj_idx = 0
    if fm is None or fm.numel() == 0 or fm.shape[2] <= obj_idx:
        return torch.zeros(num_envs, 3, device=device)
    # (N, n_bodies, 3) for the object filter -> sum over bodies -> (N, 3)
    return torch.nan_to_num(fm[:, :, obj_idx, :], nan=0.0).sum(dim=1)


def contacts(env: ManagerBasedRLEnv, threshold: float, mode: Literal["opposite", "any"] = "opposite") -> torch.Tensor:
    """Check if the fingertip contacts with the object is above a threshold."""
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_tip_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_tip_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_tip_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_tip_object_s"]
    # check if object contact force is above threshold (object filter index 0 only)
    thumb_contact = _object_contact_force(thumb_contact_sensor, env.num_envs, env.device)
    index_contact = _object_contact_force(index_contact_sensor, env.num_envs, env.device)
    middle_contact = _object_contact_force(middle_contact_sensor, env.num_envs, env.device)
    ring_contact = _object_contact_force(ring_contact_sensor, env.num_envs, env.device)
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
