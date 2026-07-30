"""Pure-Torch primitives for physics-based unscrew trajectory verification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class VerificationResult(str, Enum):
    PASS = "PASS"
    INVALID_PHYSICS_MODEL = "INVALID PHYSICS MODEL"
    TRAJECTORY_INFEASIBLE = "TRAJECTORY INFEASIBLE"
    CONTROLLER_INCONCLUSIVE = "CONTROLLER INCONCLUSIVE"


@dataclass(frozen=True)
class WrenchCommand:
    force_w: torch.Tensor
    torque_w: torch.Tensor
    force_saturated: torch.Tensor
    torque_saturated: torch.Tensor


@dataclass(frozen=True)
class TrialSummary:
    finite: bool
    held_clear: bool
    accumulated_yaw: float
    final_position_error: float
    final_orientation_error: float
    max_lateral_drift: float
    force_saturation_fraction: float
    torque_saturation_fraction: float
    peak_force: float
    peak_torque: float
    final_clearance: float


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _rotation_error_axis_angle(current_quat: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
    current_quat = current_quat / torch.linalg.vector_norm(current_quat, dim=-1, keepdim=True)
    target_quat = target_quat / torch.linalg.vector_norm(target_quat, dim=-1, keepdim=True)
    current_conjugate = torch.cat((current_quat[..., :1], -current_quat[..., 1:]), dim=-1)
    error_quat = _quat_multiply(target_quat, current_conjugate)
    error_quat = torch.where(error_quat[..., :1] < 0.0, -error_quat, error_quat)

    vector = error_quat[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, error_quat[..., :1])
    scale = torch.where(vector_norm > 1.0e-8, angle / vector_norm.clamp_min(1.0e-8), 2.0)
    return vector * scale


def _clamp_vector_norm(vector: torch.Tensor, limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    saturated = norm.squeeze(-1) > limit
    scale = torch.clamp(vector.new_tensor(limit) / norm.clamp_min(torch.finfo(vector.dtype).eps), max=1.0)
    return vector * scale, saturated


def bounded_pd_wrench(
    current_pose_w: torch.Tensor,
    target_pose_w: torch.Tensor,
    linear_velocity_w: torch.Tensor,
    angular_velocity_w: torch.Tensor,
    mass: torch.Tensor | float,
    gravity_w: torch.Tensor,
    *,
    position_stiffness: float,
    position_damping: float,
    rotation_stiffness: float,
    rotation_damping: float,
    max_force: float,
    max_torque: float,
) -> WrenchCommand:
    """Compute a world-frame PD wrench with gravity compensation and norm bounds."""
    mass_tensor = torch.as_tensor(mass, dtype=current_pose_w.dtype, device=current_pose_w.device)
    while mass_tensor.ndim < current_pose_w[..., :3].ndim:
        mass_tensor = mass_tensor.unsqueeze(-1)
    gravity_w = gravity_w.to(dtype=current_pose_w.dtype, device=current_pose_w.device)

    force_w = (
        position_stiffness * (target_pose_w[..., :3] - current_pose_w[..., :3])
        - position_damping * linear_velocity_w
        - mass_tensor * gravity_w
    )
    rotation_error = _rotation_error_axis_angle(current_pose_w[..., 3:7], target_pose_w[..., 3:7])
    torque_w = rotation_stiffness * rotation_error - rotation_damping * angular_velocity_w

    force_w, force_saturated = _clamp_vector_norm(force_w, max_force)
    torque_w, torque_saturated = _clamp_vector_norm(torque_w, max_torque)
    return WrenchCommand(force_w, torque_w, force_saturated, torque_saturated)


def straight_pull_targets(helical_targets: torch.Tensor) -> torch.Tensor:
    """Keep helical positions while fixing every target to its initial orientation."""
    targets = helical_targets.clone()
    targets[..., 3:7] = helical_targets[0, 3:7]
    return targets


def accumulate_world_yaw(
    accumulated_yaw: torch.Tensor, angular_velocity_w: torch.Tensor, dt: float
) -> torch.Tensor:
    """Integrate the world-frame angular velocity about Z."""
    return accumulated_yaw + angular_velocity_w[..., 2] * dt


def bolt_bottom_clearance(
    object_pos_r: torch.Tensor,
    object_quat_r: torch.Tensor,
    bolt_corners: torch.Tensor,
    socket_top_z: torch.Tensor | float,
) -> torch.Tensor:
    """Return the lowest rotated bolt-corner height above the socket top."""
    quat = object_quat_r / torch.linalg.vector_norm(object_quat_r, dim=-1, keepdim=True)
    batch_shape = quat.shape[:-1]
    corners = bolt_corners.to(dtype=object_pos_r.dtype, device=object_pos_r.device)
    corners = corners.reshape((1,) * len(batch_shape) + corners.shape).expand(batch_shape + corners.shape)
    quat_vector = quat[..., None, 1:].expand_as(corners)
    twice_cross = 2.0 * torch.cross(quat_vector, corners, dim=-1)
    rotated_corners = corners + quat[..., None, :1] * twice_cross + torch.cross(
        quat_vector, twice_cross, dim=-1
    )
    corner_z = rotated_corners[..., 2] + object_pos_r[..., None, 2]
    socket_top_z = torch.as_tensor(socket_top_z, dtype=object_pos_r.dtype, device=object_pos_r.device)
    return corner_z.amin(dim=-1) - socket_top_z


def classify_verification(
    helix: TrialSummary,
    straight_pull: TrialSummary,
    *,
    expected_yaw: float,
    yaw_tolerance: float = 0.35,
    position_tolerance: float = 0.005,
    orientation_tolerance: float = 0.35,
    lateral_tolerance: float = 0.005,
    saturation_inconclusive_fraction: float = 0.90,
) -> VerificationResult:
    """Classify paired helical and straight-pull physics trials."""
    if not helix.finite or not straight_pull.finite:
        return VerificationResult.CONTROLLER_INCONCLUSIVE
    if straight_pull.held_clear:
        return VerificationResult.INVALID_PHYSICS_MODEL

    helix_passed = (
        helix.held_clear
        and abs(helix.accumulated_yaw - expected_yaw) <= yaw_tolerance
        and helix.final_position_error <= position_tolerance
        and helix.final_orientation_error <= orientation_tolerance
        and helix.max_lateral_drift <= lateral_tolerance
    )
    if helix_passed:
        return VerificationResult.PASS
    if (
        helix.force_saturation_fraction >= saturation_inconclusive_fraction
        or helix.torque_saturation_fraction >= saturation_inconclusive_fraction
    ):
        return VerificationResult.CONTROLLER_INCONCLUSIVE
    return VerificationResult.TRAJECTORY_INFEASIBLE


__all__ = [
    "TrialSummary",
    "VerificationResult",
    "WrenchCommand",
    "accumulate_world_yaw",
    "bolt_bottom_clearance",
    "bounded_pd_wrench",
    "classify_verification",
    "straight_pull_targets",
]
