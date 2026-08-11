"""Simulator-free geometry helpers for clean-table reset states."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def outside_box_state_indices(
    object_root_pose: torch.Tensor,
    box_root_pose: torch.Tensor,
    box_min: Sequence[float],
    box_max: Sequence[float],
) -> torch.Tensor:
    """Return state indices whose object origin lies outside the oriented box bounds."""
    offset = object_root_pose[:, :3] - box_root_pose[:, :3]
    quaternion = box_root_pose[:, 3:7]
    quaternion_xyz = quaternion[:, 1:]
    twice_cross = 2.0 * quaternion_xyz.cross(offset, dim=-1)
    position_box = (
        offset
        - quaternion[:, :1] * twice_cross
        + quaternion_xyz.cross(twice_cross, dim=-1)
    )
    lower = position_box.new_tensor(box_min)
    upper = position_box.new_tensor(box_max)
    inside = ((position_box >= lower) & (position_box <= upper)).all(dim=1)
    indices = torch.nonzero(~inside, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        raise ValueError("Reset dataset contains no outside-box states.")
    return indices


__all__ = ["outside_box_state_indices"]
