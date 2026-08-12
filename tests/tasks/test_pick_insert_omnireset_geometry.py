from __future__ import annotations

import math

import pytest
import torch
from pxr import Usd, UsdGeom

from src.tasks.pick_insert_omnireset.mdps.geometry import (
    RectangularInsertionGeometry,
    peg_inside_rectangular_hole,
)


def _geometry() -> RectangularInsertionGeometry:
    return RectangularInsertionGeometry(
        peg_bottom_corners_o=torch.tensor(
            [
                [-0.0225, -0.0225, -0.0450],
                [-0.0225, 0.0225, -0.0450],
                [0.0225, -0.0225, -0.0450],
                [0.0225, 0.0225, -0.0450],
            ]
        ),
        peg_opposite_corners_o=torch.tensor(
            [
                [-0.0225, -0.0225, 0.0450],
                [-0.0225, 0.0225, 0.0450],
                [0.0225, -0.0225, 0.0450],
                [0.0225, 0.0225, 0.0450],
            ]
        ),
        hole_bottom_position_h=torch.tensor([0.0, 0.0, -0.015163]),
        hole_bottom_quaternion_h=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        aperture_min_i=torch.tensor([-0.0320, -0.0320]),
        aperture_max_i=torch.tensor([0.0320, 0.0320]),
        cavity_floor_z_i=torch.tensor(0.0),
        cavity_mouth_z_i=torch.tensor(0.036703),
    )


def _pose(
    xyz: tuple[float, float, float],
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    return torch.tensor([[*xyz, *quat]], dtype=torch.float32)


def test_partially_inserted_designated_end_succeeds() -> None:
    geometry = _geometry()
    object_pose = _pose((0.0, 0.0, 0.060))
    hole_pose = _pose((0.0, 0.0, 0.0))

    assert bool(peg_inside_rectangular_hole(object_pose, hole_pose, geometry)[0])


@pytest.mark.parametrize(
    "object_pose",
    [
        _pose((0.0, 0.0, 0.070)),
        _pose((0.012, 0.0, 0.060)),
        _pose((0.0, 0.0, 0.020)),
        _pose((0.0, 0.0, -0.060), (0.0, 1.0, 0.0, 0.0)),
    ],
    ids=["above-mouth", "outside-aperture", "below-floor", "wrong-end"],
)
def test_non_contained_poses_fail(object_pose: torch.Tensor) -> None:
    assert not bool(
        peg_inside_rectangular_hole(
            object_pose,
            _pose((0.0, 0.0, 0.0)),
            _geometry(),
        )[0]
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_containment_accepts_cpu_geometry_with_cuda_poses() -> None:
    result = peg_inside_rectangular_hole(
        _pose((0.0, 0.0, 0.060)).cuda(),
        _pose((0.0, 0.0, 0.0)).cuda(),
        _geometry(),
    )

    assert bool(result[0])


def test_tilted_peg_clipping_mouth_fails() -> None:
    half_angle = math.radians(20.0) / 2.0
    object_pose = _pose(
        (0.0, 0.0, 0.060),
        (math.cos(half_angle), 0.0, math.sin(half_angle), 0.0),
    )

    assert not bool(
        peg_inside_rectangular_hole(
            object_pose,
            _pose((0.0, 0.0, 0.0)),
            _geometry(),
        )[0]
    )


def test_containment_is_invariant_to_hole_world_pose() -> None:
    half_angle = math.pi / 4.0
    hole_pose = _pose(
        (0.4, -0.2, 0.3),
        (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)),
    )
    object_pose = _pose(
        (0.4, -0.2, 0.360),
        (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)),
    )

    assert bool(peg_inside_rectangular_hole(object_pose, hole_pose, _geometry())[0])


def _mesh(
    stage: Usd.Stage,
    path: str,
    points: list[tuple[float, float, float]],
    faces: list[list[int]],
) -> Usd.Prim:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
    return mesh.GetPrim()


def test_geometry_is_derived_from_scaled_collision_meshes_and_bottom_offsets() -> None:
    from src.tasks.pick_insert_omnireset.mdps.asset_geometry import _build_geometry

    stage = Usd.Stage.CreateInMemory()
    object_root = UsdGeom.Xform.Define(stage, "/Object").GetPrim()
    object_points = [
        (x, y, z)
        for z in (-1.0, 1.0)
        for y in (-1.0, 1.0)
        for x in (-1.0, 1.0)
    ]
    object_mesh = _mesh(
        stage,
        "/Object/Collision",
        object_points,
        [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ],
    )

    hole_root = UsdGeom.Xform.Define(stage, "/Hole").GetPrim()
    hole_points = []
    hole_faces = []
    for x, y in ((-2.0, -3.0), (-2.0, 3.0), (2.0, -3.0), (2.0, 3.0)):
        first = len(hole_points)
        hole_points.extend(((x, y, -2.0), (x, y, 3.0), (x + 0.5, y, 3.0)))
        hole_faces.append([first, first + 1, first + 2])
    hole_mesh = _mesh(stage, "/Hole/Collision", hole_points, hole_faces)

    geometry = _build_geometry(
        object_mesh,
        object_root,
        (2.0, 3.0, 4.0),
        ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0, 0.0)),
        hole_mesh,
        hole_root,
        (2.0, 2.0, 1.0),
        ((0.0, 0.0, -2.0), (1.0, 0.0, 0.0, 0.0)),
    )

    torch.testing.assert_close(
        geometry.peg_bottom_corners_o.min(dim=0).values,
        torch.tensor([-2.0, -3.0, -4.0]),
    )
    torch.testing.assert_close(
        geometry.peg_opposite_corners_o.max(dim=0).values,
        torch.tensor([2.0, 3.0, 4.0]),
    )
    torch.testing.assert_close(geometry.aperture_min_i, torch.tensor([-4.0, -6.0]))
    torch.testing.assert_close(geometry.aperture_max_i, torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(geometry.cavity_floor_z_i, torch.tensor(0.0))
    torch.testing.assert_close(geometry.cavity_mouth_z_i, torch.tensor(5.0))
