"""Simulator-free geometry helpers for pick-insert reset states."""

from __future__ import annotations

import torch


SUCCESS_POS_TOL = 0.015
SUCCESS_AXIS_TOL = 0.15
SUCCESS_DEPTH = 0.015


def unfinished_state_indices(
    object_root_pose: torch.Tensor,
    hole_root_pose: torch.Tensor,
    pos_tol: float = SUCCESS_POS_TOL,
    axis_tol: float = SUCCESS_AXIS_TOL,
    depth: float = SUCCESS_DEPTH,
) -> torch.Tensor:
    """Return indices whose peg is not yet successfully inserted."""
    target_position = hole_root_pose[:, :3].clone()
    target_position[:, 2] += depth
    position_distance = torch.norm(object_root_pose[:, :3] - target_position, dim=1)

    object_axis = _local_z_axis(object_root_pose[:, 3:7])
    hole_axis = _local_z_axis(hole_root_pose[:, 3:7])
    axis_error = 1.0 - torch.sum(object_axis * hole_axis, dim=1).abs().clamp(0.0, 1.0)
    unfinished = (position_distance >= pos_tol) | (axis_error >= axis_tol)
    indices = torch.nonzero(unfinished, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        raise ValueError("Reset archive contains no unfinished states.")
    return indices


def _local_z_axis(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(dim=1)
    return torch.stack(
        (
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=1,
    )


__all__ = [
    "SUCCESS_AXIS_TOL",
    "SUCCESS_DEPTH",
    "SUCCESS_POS_TOL",
    "unfinished_state_indices",
]
