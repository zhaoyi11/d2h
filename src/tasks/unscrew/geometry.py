"""Pure-Torch calibrated geometry for the square-leg unscrew task."""

from __future__ import annotations

from collections.abc import Sequence

import torch


# Collision bounds authored in the UW square-leg USD, scaled by the task's 1.5 asset scale.
BOLT_AABB_MIN = (-0.018282, -0.017355, -0.084987)
BOLT_AABB_MAX = (0.018281, 0.019207, -0.043598)
SOCKET_TOP_Z = 0.0233475


def _aabb_corners(
    lower: Sequence[float], upper: Sequence[float], *, device: str | torch.device
) -> torch.Tensor:
    lo = torch.tensor(lower, dtype=torch.float32, device=device)
    hi = torch.tensor(upper, dtype=torch.float32, device=device)
    return torch.stack(
        [
            torch.stack((x, y, z))
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )


def bolt_aabb_corners(device: str | torch.device) -> torch.Tensor:
    return _aabb_corners(BOLT_AABB_MIN, BOLT_AABB_MAX, device=device)


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


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    )
    quat_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(quat_vector, vector, dim=-1)
    return (
        vector
        + quaternion[..., :1] * twice_cross
        + torch.cross(quat_vector, twice_cross, dim=-1)
    )


def relative_pose(
    parent_pose: torch.Tensor, child_pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a WXYZ child pose expressed in its parent frame."""
    parent_quat = parent_pose[..., 3:7]
    parent_quat_inverse = torch.cat(
        (parent_quat[..., :1], -parent_quat[..., 1:]), dim=-1
    )
    position = _quat_apply(
        parent_quat_inverse,
        child_pose[..., :3] - parent_pose[..., :3],
    )
    orientation = _quat_multiply(parent_quat_inverse, child_pose[..., 3:7])
    return position, orientation


def bolt_bottom_clearance(
    object_pos_r: torch.Tensor,
    object_quat_r: torch.Tensor,
    bolt_corners: torch.Tensor,
    socket_top_z: float = SOCKET_TOP_Z,
) -> torch.Tensor:
    """Return the lowest rotated bolt-corner height above the socket top."""
    quaternion = object_quat_r / torch.linalg.vector_norm(
        object_quat_r, dim=-1, keepdim=True
    )
    batch_shape = quaternion.shape[:-1]
    corners = bolt_corners.to(dtype=object_pos_r.dtype, device=object_pos_r.device)
    corners = corners.reshape((1,) * len(batch_shape) + corners.shape).expand(
        batch_shape + corners.shape
    )
    quat_vector = quaternion[..., None, 1:].expand_as(corners)
    twice_cross = 2.0 * torch.cross(quat_vector, corners, dim=-1)
    rotated_corners = (
        corners
        + quaternion[..., None, :1] * twice_cross
        + torch.cross(quat_vector, twice_cross, dim=-1)
    )
    corner_z = rotated_corners[..., 2] + object_pos_r[..., None, 2]
    return corner_z.amin(dim=-1) - socket_top_z


__all__ = [
    "BOLT_AABB_MAX",
    "BOLT_AABB_MIN",
    "SOCKET_TOP_Z",
    "bolt_aabb_corners",
    "bolt_bottom_clearance",
    "relative_pose",
]
