"""Simulator-free geometry helpers for pick-insert success and reset states."""

from __future__ import annotations

from dataclasses import dataclass

import torch


INSERTION_SHAPING_TARGET_DEPTH = 0.015


@dataclass(frozen=True)
class RectangularInsertionGeometry:
    """Collision geometry expressed in the peg root and hole insertion frames."""

    peg_bottom_corners_o: torch.Tensor
    peg_opposite_corners_o: torch.Tensor
    hole_bottom_position_h: torch.Tensor
    hole_bottom_quaternion_h: torch.Tensor
    aperture_min_i: torch.Tensor
    aperture_max_i: torch.Tensor
    cavity_floor_z_i: torch.Tensor
    cavity_mouth_z_i: torch.Tensor

    def to(self, device: str | torch.device) -> RectangularInsertionGeometry:
        return RectangularInsertionGeometry(
            **{
                field: value.to(device=device)
                for field, value in vars(self).items()
            }
        )


def peg_inside_rectangular_hole(
    object_root_pose: torch.Tensor,
    hole_root_pose: torch.Tensor,
    geometry: RectangularInsertionGeometry,
) -> torch.Tensor:
    """Return whether the metadata-designated peg end is contained by the cavity."""
    bottom_i = _object_points_in_insertion_frame(
        object_root_pose,
        hole_root_pose,
        geometry.peg_bottom_corners_o,
        geometry,
    )
    opposite_i = _object_points_in_insertion_frame(
        object_root_pose,
        hole_root_pose,
        geometry.peg_opposite_corners_o,
        geometry,
    )

    def pose_tensor(value: torch.Tensor) -> torch.Tensor:
        return value.to(
            device=object_root_pose.device,
            dtype=object_root_pose.dtype,
        )

    floor_z = pose_tensor(geometry.cavity_floor_z_i)
    mouth_z = pose_tensor(geometry.cavity_mouth_z_i)
    scale = torch.stack(
        (
            (pose_tensor(geometry.aperture_max_i) - pose_tensor(geometry.aperture_min_i)).max(),
            mouth_z - floor_z,
            torch.linalg.vector_norm(
                pose_tensor(geometry.peg_opposite_corners_o)
                - pose_tensor(geometry.peg_bottom_corners_o),
                dim=-1,
            ).max(),
        )
    ).max()
    epsilon = torch.finfo(object_root_pose.dtype).eps * scale * 32

    bottom_contained = (
        _points_inside_aperture(bottom_i[..., :2], geometry, epsilon)
        & (bottom_i[..., 2] >= floor_z - epsilon)
        & (bottom_i[..., 2] < mouth_z - epsilon)
    ).all(dim=1)

    edge_height = opposite_i[..., 2] - bottom_i[..., 2]
    crosses_mouth = (
        (edge_height > epsilon)
        & (opposite_i[..., 2] >= mouth_z + epsilon)
    ).all(dim=1)
    fraction = (
        (mouth_z - bottom_i[..., 2])
        / edge_height.clamp_min(epsilon)
    )
    mouth_points_xy = bottom_i[..., :2] + fraction.unsqueeze(-1) * (
        opposite_i[..., :2] - bottom_i[..., :2]
    )
    mouth_contained = _points_inside_aperture(
        mouth_points_xy,
        geometry,
        epsilon,
    ).all(dim=1)
    return bottom_contained & crosses_mouth & mouth_contained


def unfinished_state_indices(
    object_root_pose: torch.Tensor,
    hole_root_pose: torch.Tensor,
    geometry: RectangularInsertionGeometry,
) -> torch.Tensor:
    """Return indices whose peg is not already physically inserted."""
    unfinished = ~peg_inside_rectangular_hole(
        object_root_pose,
        hole_root_pose,
        geometry,
    )
    indices = torch.nonzero(unfinished, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        raise ValueError("Reset archive contains no unfinished states.")
    return indices


def _object_points_in_insertion_frame(
    object_root_pose: torch.Tensor,
    hole_root_pose: torch.Tensor,
    points_o: torch.Tensor,
    geometry: RectangularInsertionGeometry,
) -> torch.Tensor:
    points_o = points_o.to(device=object_root_pose.device, dtype=object_root_pose.dtype)
    points_w = _quat_apply(
        object_root_pose[:, None, 3:7],
        points_o[None],
    ) + object_root_pose[:, None, :3]
    points_h = _quat_apply_inverse(
        hole_root_pose[:, None, 3:7],
        points_w - hole_root_pose[:, None, :3],
    )
    bottom_position_h = geometry.hole_bottom_position_h.to(
        device=points_h.device,
        dtype=points_h.dtype,
    )
    bottom_quaternion_h = geometry.hole_bottom_quaternion_h.to(
        device=points_h.device,
        dtype=points_h.dtype,
    )
    return _quat_apply_inverse(
        bottom_quaternion_h[None, None],
        points_h - bottom_position_h[None, None],
    )


def _points_inside_aperture(
    points_xy: torch.Tensor,
    geometry: RectangularInsertionGeometry,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    aperture_min = geometry.aperture_min_i.to(
        device=points_xy.device,
        dtype=points_xy.dtype,
    )
    aperture_max = geometry.aperture_max_i.to(
        device=points_xy.device,
        dtype=points_xy.dtype,
    )
    return ((points_xy >= aperture_min - epsilon) & (points_xy <= aperture_max + epsilon)).all(
        dim=-1
    )


def _quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(
        xyz,
        twice_cross,
        dim=-1,
    )


def _quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    inverse = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    return _quat_apply(inverse, vector)


__all__ = [
    "INSERTION_SHAPING_TARGET_DEPTH",
    "RectangularInsertionGeometry",
    "peg_inside_rectangular_hole",
    "unfinished_state_indices",
]
