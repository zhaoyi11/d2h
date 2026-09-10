from __future__ import annotations

import math as _math
from typing import TYPE_CHECKING

import torch
from isaaclab.envs import ManagerBasedRLEnv

if TYPE_CHECKING:
    from isaaclab.sensors import ContactSensor


def _compute_contact_metrics(
    env: "ManagerBasedRLEnv",
    contact_sensor_names: list[str],
    fingertip_transforms_name: str,
    force_threshold: float,
    contact_pose_range_deg: float,
    filter_indices: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Shared layer: per-fingertip forces rotated into fingertip frame, theta/phi, contact masks.

    Aggregates a *set* of contact-filter indices into one per-fingertip reading. The default
    (``None`` -> object filter index 0) reproduces the original single-filter behavior exactly.
    Passing ``external_indices()`` merges all external contact targets (receptacle + table + ...)
    into a single fixed-size external channel:
      - net force = vector sum of the selected filters' forces,
      - in_contact = OR of the per-filter masks (so two simultaneous opposing external contacts
        cannot cancel to "no contact"),
      - theta/phi = direction of the net force in the fingertip frame.

    Returns dict with keys:
      in_contact   (num_envs, n_fingers) bool
      force_mag    (num_envs, n_fingers) float
      contact_pose (num_envs, n_fingers, 2) float  — [theta, phi] clamped to pose_limit
    """
    from isaaclab.utils.math import quat_apply_inverse

    if filter_indices is None:
        filter_indices = [0]  # object is always the first contact-filter target

    pose_limit = _math.radians(contact_pose_range_deg)
    # (num_envs, n_fingers, 4) world-frame fingertip orientations as (w, x, y, z)
    target_quat_w = env.scene.sensors[fingertip_transforms_name].data.target_quat_w

    in_contact_list, force_mag_list, pose_list = [], [], []

    for i, name in enumerate(contact_sensor_names):
        sensor = env.scene.sensors[name]
        fm = sensor.data.force_matrix_w

        # Keep only the requested filters that actually exist on this sensor.
        valid = (
            [k for k in filter_indices if 0 <= k < fm.shape[2]]
            if (fm is not None and fm.numel() > 0)
            else []
        )
        if not valid:
            zeros = torch.zeros(env.num_envs, device=env.device)
            in_contact_list.append(zeros.bool())
            force_mag_list.append(zeros)
            pose_list.append(torch.zeros(env.num_envs, 2, device=env.device))
            continue

        fm = torch.nan_to_num(fm, nan=0.0)
        # Per-filter force summed over the sensor's bodies: (N, len(valid), 3)
        per_filter = fm[:, :, valid, :].sum(dim=1)
        # Net (merged) force across the selected filters: (N, 3)
        F_world = per_filter.sum(dim=1)
        F_mag = torch.linalg.norm(F_world, dim=-1)                      # (N,)
        # OR of the per-filter contact masks (robust to vector cancellation).
        per_filter_mag = torch.linalg.norm(per_filter, dim=-1)          # (N, len(valid))
        in_contact = (per_filter_mag > force_threshold).any(dim=-1)     # (N,)

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


def _object_contact_force_w(sensor: ContactSensor, num_envs: int, device) -> torch.Tensor:
    """Per-env object contact force (N, 3) for a fingertip sensor.

    Selects only the object's contact-filter index (0) and sums over the sensor's bodies, so it
    stays correct when external contact targets (receptacle, table, ...) add more filter columns
    to ``force_matrix_w``. Backward compatible with single-filter sensors (index 0 == only filter).
    """
    fm = sensor.data.force_matrix_w
    if fm is None or fm.numel() == 0 or fm.shape[2] < 1:
        return torch.zeros(num_envs, 3, device=device)
    return torch.nan_to_num(fm[:, :, 0, :], nan=0.0).sum(dim=1)

def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Check for object contact above threshold on the thumb and another fingertip."""

    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_fingertip_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["fingertip_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["fingertip_2_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["fingertip_3_object_s"]
    # check if object contact force is above threshold (object filter index 0 only)
    thumb_contact = _object_contact_force_w(thumb_contact_sensor, env.num_envs, env.device)
    index_contact = _object_contact_force_w(index_contact_sensor, env.num_envs, env.device)
    middle_contact = _object_contact_force_w(middle_contact_sensor, env.num_envs, env.device)
    ring_contact = _object_contact_force_w(ring_contact_sensor, env.num_envs, env.device)
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
    )

    return good_contact_cond1
