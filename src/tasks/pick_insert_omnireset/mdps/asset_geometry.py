"""Derive rectangular insertion geometry from UWLab asset metadata and collision meshes."""

from __future__ import annotations

import functools
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch
import yaml
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .geometry import RectangularInsertionGeometry


_GEOMETRY_CACHE: dict[tuple[Any, ...], RectangularInsertionGeometry] = {}


def insertion_geometry_from_assets(
    object_asset,
    hole_asset,
    device: str | torch.device,
) -> RectangularInsertionGeometry:
    """Return cached geometry for the assets as they were scaled and spawned."""
    object_scale = _spawn_scale(object_asset)
    hole_scale = _spawn_scale(hole_asset)
    key = (
        object_asset.cfg.spawn.usd_path,
        object_scale,
        hole_asset.cfg.spawn.usd_path,
        hole_scale,
    )
    geometry = _GEOMETRY_CACHE.get(key)
    if geometry is None:
        import isaaclab.sim as sim_utils

        object_root = sim_utils.find_first_matching_prim(object_asset.cfg.prim_path)
        hole_root = sim_utils.find_first_matching_prim(hole_asset.cfg.prim_path)
        if object_root is None or hole_root is None:
            raise ValueError("Insertion assets must be spawned before geometry is derived.")

        object_mesh = _single_collision_mesh(object_root, sim_utils)
        hole_mesh = _single_collision_mesh(hole_root, sim_utils)
        geometry = _build_geometry(
            object_mesh,
            object_root,
            object_scale,
            _load_bottom_offset(object_asset.cfg.spawn.usd_path),
            hole_mesh,
            hole_root,
            hole_scale,
            _load_bottom_offset(hole_asset.cfg.spawn.usd_path),
        )
        _GEOMETRY_CACHE[key] = geometry
    return geometry.to(device)


def _spawn_scale(asset) -> tuple[float, float, float]:
    scale = asset.cfg.spawn.scale
    return (1.0, 1.0, 1.0) if scale is None else tuple(float(value) for value in scale)


def _single_collision_mesh(root: Usd.Prim, sim_utils) -> Usd.Prim:
    meshes = sim_utils.get_all_matching_child_prims(
        root.GetPath(),
        predicate=lambda prim: (
            prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)
        ),
    )
    if len(meshes) != 1:
        raise ValueError(
            f"Expected one collision mesh below {root.GetPath()}, found {len(meshes)}."
        )
    return meshes[0]


@functools.cache
def _load_bottom_offset(usd_path: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    metadata_path = _sibling_metadata_path(usd_path)
    from isaaclab.utils.assets import read_file

    metadata = yaml.safe_load(read_file(metadata_path))

    bottom_offset = metadata["bottom_offset"]
    return (
        tuple(float(value) for value in bottom_offset["pos"]),
        tuple(float(value) for value in bottom_offset["quat"]),
    )


def _sibling_metadata_path(usd_path: str) -> str:
    parsed = urlsplit(usd_path)
    path = parsed.path.rsplit("/", 1)[0] + "/metadata.yaml"
    return urlunsplit(parsed._replace(path=path))


def _build_geometry(
    object_mesh: Usd.Prim,
    object_root: Usd.Prim,
    object_scale: tuple[float, float, float],
    object_bottom_offset: tuple[tuple[float, ...], tuple[float, ...]],
    hole_mesh: Usd.Prim,
    hole_root: Usd.Prim,
    hole_scale: tuple[float, float, float],
    hole_bottom_offset: tuple[tuple[float, ...], tuple[float, ...]],
) -> RectangularInsertionGeometry:
    object_points, _ = _mesh_data_in_scaled_root(
        object_mesh,
        object_root,
        object_scale,
    )
    hole_points, hole_faces = _mesh_data_in_scaled_root(
        hole_mesh,
        hole_root,
        hole_scale,
    )

    object_bottom_position, object_bottom_quaternion = _scaled_offset(
        object_bottom_offset,
        object_scale,
    )
    hole_bottom_position, hole_bottom_quaternion = _scaled_offset(
        hole_bottom_offset,
        hole_scale,
    )
    bottom_corners_o, opposite_corners_o = _derive_peg_corners(
        object_points,
        object_bottom_position,
        object_bottom_quaternion,
    )
    aperture_min, aperture_max, floor_z, mouth_z = _derive_cavity(
        hole_points,
        hole_faces,
        hole_bottom_position,
        hole_bottom_quaternion,
    )

    def tensor(value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32)

    return RectangularInsertionGeometry(
        peg_bottom_corners_o=tensor(bottom_corners_o),
        peg_opposite_corners_o=tensor(opposite_corners_o),
        hole_bottom_position_h=tensor(hole_bottom_position),
        hole_bottom_quaternion_h=tensor(hole_bottom_quaternion),
        aperture_min_i=tensor(aperture_min),
        aperture_max_i=tensor(aperture_max),
        cavity_floor_z_i=tensor(floor_z),
        cavity_mouth_z_i=tensor(mouth_z),
    )


def _mesh_data_in_scaled_root(
    mesh_prim: Usd.Prim,
    root_prim: Usd.Prim,
    scale: tuple[float, float, float],
) -> tuple[np.ndarray, list[np.ndarray]]:
    mesh = UsdGeom.Mesh(mesh_prim)
    transform = UsdGeom.XformCache().ComputeRelativeTransform(mesh_prim, root_prim)[0]
    points = np.asarray(
        [
            transform.Transform(Gf.Vec3d(*point))
            for point in mesh.GetPointsAttr().Get()
        ],
        dtype=np.float64,
    )
    points *= np.asarray(scale)

    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
    faces = []
    start = 0
    for count in counts:
        stop = start + int(count)
        faces.append(indices[start:stop])
        start = stop
    return points, faces


def _scaled_offset(
    offset: tuple[tuple[float, ...], tuple[float, ...]],
    scale: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    position, quaternion = offset
    return np.asarray(position) * np.asarray(scale), np.asarray(quaternion)


def _derive_peg_corners(
    points_o: np.ndarray,
    bottom_position_o: np.ndarray,
    bottom_quaternion_o: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points_i = _quat_apply_inverse(
        bottom_quaternion_o,
        points_o - bottom_position_o,
    )
    extent = np.ptp(points_i, axis=0)
    tolerance = _inference_tolerance(extent)
    minimum = points_i.min(axis=0)
    maximum = points_i.max(axis=0)
    if (
        np.any(extent <= tolerance)
        or abs(minimum[2]) > tolerance
        or maximum[2] <= tolerance
    ):
        raise ValueError("Peg bottom_offset does not identify the bottom of a rectangular collision box.")

    on_box = np.logical_or(
        np.isclose(points_i, minimum, atol=tolerance, rtol=0.0),
        np.isclose(points_i, maximum, atol=tolerance, rtol=0.0),
    ).all(axis=1)
    if not on_box.all():
        raise ValueError("Peg collision mesh must be a rectangular box.")

    xy = np.asarray(
        [
            [minimum[0], minimum[1]],
            [minimum[0], maximum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
        ]
    )
    bottom_i = np.column_stack((xy, np.full(4, minimum[2])))
    opposite_i = np.column_stack((xy, np.full(4, maximum[2])))
    return (
        _quat_apply(bottom_quaternion_o, bottom_i) + bottom_position_o,
        _quat_apply(bottom_quaternion_o, opposite_i) + bottom_position_o,
    )


def _derive_cavity(
    points_h: np.ndarray,
    faces: list[np.ndarray],
    bottom_position_h: np.ndarray,
    bottom_quaternion_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    points_i = _quat_apply_inverse(
        bottom_quaternion_h,
        points_h - bottom_position_h,
    )
    tolerance = _inference_tolerance(np.ptp(points_i, axis=0))
    edges = {
        tuple(sorted((int(first), int(second))))
        for face in faces
        for first, second in zip(face, np.roll(face, -1))
    }

    cavity_edges = []
    for first, second in edges:
        endpoint_a = points_i[first]
        endpoint_b = points_i[second]
        lower, upper = sorted((endpoint_a, endpoint_b), key=lambda point: point[2])
        if (
            np.allclose(lower[:2], upper[:2], atol=tolerance, rtol=0.0)
            and upper[2] - lower[2] > tolerance
            and abs(lower[2]) <= tolerance
        ):
            cavity_edges.append((lower, upper))

    lower_points = _unique_rows(
        np.asarray([lower for lower, _ in cavity_edges]),
        tolerance,
    )
    upper_points = _unique_rows(
        np.asarray([upper for _, upper in cavity_edges]),
        tolerance,
    )
    if lower_points.shape[0] != 4 or upper_points.shape[0] != 4:
        raise ValueError("Hole collision mesh must expose four cavity edges at bottom_offset.")

    aperture_min = lower_points[:, :2].min(axis=0)
    aperture_max = lower_points[:, :2].max(axis=0)
    rectangular_corners = np.logical_or(
        np.isclose(lower_points[:, :2], aperture_min, atol=tolerance, rtol=0.0),
        np.isclose(lower_points[:, :2], aperture_max, atol=tolerance, rtol=0.0),
    ).all(axis=1)
    if (
        not rectangular_corners.all()
        or np.any(aperture_max - aperture_min <= tolerance)
        or np.ptp(lower_points[:, 2]) > tolerance
        or np.ptp(upper_points[:, 2]) > tolerance
    ):
        raise ValueError("Hole cavity edges must form one rectangular vertical aperture.")

    return (
        aperture_min,
        aperture_max,
        float(lower_points[:, 2].mean()),
        float(upper_points[:, 2].mean()),
    )


def _unique_rows(points: np.ndarray, tolerance: float) -> np.ndarray:
    unique: list[np.ndarray] = []
    for point in points:
        if not any(np.allclose(point, other, atol=tolerance, rtol=0.0) for other in unique):
            unique.append(point)
    return np.asarray(unique)


def _inference_tolerance(extent: np.ndarray) -> float:
    return float(np.finfo(np.float32).eps * np.max(extent) * 128)


def _quat_apply(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:]
    twice_cross = 2.0 * np.cross(xyz, vectors)
    return vectors + quaternion[0] * twice_cross + np.cross(xyz, twice_cross)


def _quat_apply_inverse(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    inverse = np.concatenate((quaternion[:1], -quaternion[1:]))
    return _quat_apply(inverse, vectors)


__all__ = ["insertion_geometry_from_assets"]
