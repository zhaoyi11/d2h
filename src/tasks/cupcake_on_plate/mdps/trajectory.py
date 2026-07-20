# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cupcake-on-plate object-pose trajectory builder.

The scripted reach->lift->move->reorient->place->hold cupcake trajectory that is specific to the
cupcake-on-plate task, composed from the task-agnostic primitives in
:mod:`src.policy.high_level.utils`. The task is to grasp a cupcake that spawns upside-down, lift it a
little, carry it (still upside-down) to above the plate, flip it upright *in-hand* there (the reorient
segment slerps the object-goal quaternion from its grasped upside-down orientation to ``upright_quat``),
and lower it onto the plate.

The in-hand flip is a direct generalization of the pick-insert "align" segment (hold position, swap
the goal quaternion): the object-goal quaternion drives ``command[:, :7]`` -- the target the frozen
low-level hand policy chases -- while the grasp anchor stays root-aligned, so the frozen hand policy
plus the bounded PID anchor correction physically execute the reorientation.

Pure ``torch`` so it can be unit-tested in isolation; the command term that consumes it lives in
:mod:`src.tasks.cupcake_on_plate.mdps.commands`.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.utils import (
    _as_pose_tensor,
    _with_normalized_quat,
    build_object_pose_sequence_from_keyframes,
)


# Plate placement pose (x, y, z, qw, qx, qy, qz) in the robot base frame. Matches the plate's scene
# init pose (z = table top 0.255, plate bottom flush on the table; plate top is 0.255 + 0.037 thick =
# 0.292). Only its x/y and z are used (placement orientation comes from ``upright_quat``).
DEFAULT_CUPCAKE_ON_PLATE_PLATE_POSE = (0.55, 0.0, 0.255, 1.0, 0.0, 0.0, 0.0)

# Canonical upright orientation of the cupcake in the robot base frame (top/frosting facing +z).
# The cupcake USD's identity orientation is upright, so identity is the flip target; the cupcake
# spawns rotated 180 deg (upside-down) and the reorient segment slerps the goal quaternion here.
DEFAULT_CUPCAKE_UPRIGHT_QUAT = (1.0, 0.0, 0.0, 0.0)

# Number of interpolation steps for each of the 6 segments: reach / lift / move / reorient / place /
# hold. The reach segment is 0-step: it adds no waypoints, so it just relabels the initial waypoint
# (the object held at its settled upside-down pose) as its own stage while the (open) hand reaches
# the grasp pose over the cupcake. The cupcake is lifted a little, carried (still upside-down) to
# above the plate, and only there flipped upright, so the reorient segment is interpolated gradually
# (the frozen hand policy has to physically rotate the cupcake between waypoints).
DEFAULT_CUPCAKE_ON_PLATE_SEGMENT_STEPS = (0, 1, 2, 2, 1, 1)

# Per-stage object position/orientation tolerances for advancing the command trajectory. One entry
# per segment: reach / lift / move / reorient / place / hold. The reorient stage keeps a loose
# orientation tolerance so the flip advances even before it is perfectly upright.
DEFAULT_CUPCAKE_ON_PLATE_STAGE_OBJECT_TOLERANCES = (
    StageObjTol(0.02, 0.3),  # reach (object held at its settled pose; advance is arm-gated, hand open)
    StageObjTol(0.02, 0.1),  # lift (goal lift_height above the object; the grip must lift it there)
    StageObjTol(0.02, 0.3),  # move (carry above the plate, still upside-down)
    StageObjTol(0.03, 0.8),  # reorient (in-hand flip to upright above the plate; loose orientation tol)
    StageObjTol(0.01, 0.2),  # place (lower onto the plate)
    StageObjTol(0.01, 0.2),  # hold
)


def build_cupcake_on_plate_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    plate_pose: torch.Tensor | Sequence[float] = DEFAULT_CUPCAKE_ON_PLATE_PLATE_POSE,
    upright_quat: torch.Tensor | Sequence[float] = DEFAULT_CUPCAKE_UPRIGHT_QUAT,
    segment_steps: Sequence[int] = DEFAULT_CUPCAKE_ON_PLATE_SEGMENT_STEPS,
    lift_height: float = 0.02,
    above_offset: float = 0.10,
    place_height: float = 0.03,
) -> torch.Tensor:
    """Build an interpolated cupcake pick->carry->reorient->place object-pose trajectory.

    Poses use ``(x, y, z, qw, qx, qy, qz)`` in the robot base frame. The returned sequence holds the
    cupcake at ``current_pose`` (reach, while the open hand moves to the grasp pose), raises the grasp
    goal ``lift_height`` above it (lift, where the grip lifts the cupcake to advance -- a grip-secured
    check), carries it -- still upside-down -- to ``above_offset`` above the plate (move), flips it
    upright in place there (reorient, slerp of the object-goal quaternion to ``upright_quat``), and
    lowers it onto the plate at ``place_height`` above the plate pose (place) -- 6 segments:
    reach / lift / move / reorient / place / hold. The reach segment is typically 0-step (see
    :data:`DEFAULT_CUPCAKE_ON_PLATE_SEGMENT_STEPS`), so it adds no waypoints and the cupcake never
    moves during it. The keyframes are interpolated by the shared
    :func:`~src.policy.high_level.utils.build_object_pose_sequence_from_keyframes` core.
    """
    if len(segment_steps) != 6:
        raise ValueError("segment_steps must contain 6 values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")

    current = _with_normalized_quat(_as_pose_tensor(current_pose))
    plate = _with_normalized_quat(
        _as_pose_tensor(plate_pose, dtype=current.dtype, device=current.device)
    )
    # Normalize the upright quaternion (build a throwaway pose so we can reuse the shared helper).
    upright_q = _with_normalized_quat(
        torch.cat((current.new_zeros(3), current.new_tensor(upright_quat)))
    )[3:7]

    # Lift keyframe: the cupcake's settled pose raised straight up by lift_height (same, still
    # upside-down, orientation). Reach holds the cupcake at its settled pose (goal == current) while
    # the open hand arrives; the lift segment then asks the grip to raise it to this pose.
    lifted = torch.cat(
        (
            torch.stack((current[0], current[1], current[2] + current.new_tensor(lift_height))),
            current[3:7],
        )
    )

    # Move keyframe: carried across to above the plate at above_offset, STILL upside-down (same
    # orientation as current) -- the flip happens later, over the plate.
    above_plate_grasped = torch.cat(
        (
            torch.stack((plate[0], plate[1], plate[2] + current.new_tensor(above_offset))),
            current[3:7],
        )
    )

    # Reorient keyframe: same (above-plate) position, goal orientation swapped to upright. The reorient
    # segment slerps from the grasped upside-down orientation to this -- the in-hand flip over the plate.
    above_plate_upright = torch.cat((above_plate_grasped[:3], upright_q))

    # Place keyframe: lowered onto the plate at place_height above the plate pose, upright.
    on_plate = torch.cat(
        (
            torch.stack((plate[0], plate[1], plate[2] + current.new_tensor(place_height))),
            upright_q,
        )
    )

    # Leading ``current`` is the reach keyframe: the object goal is held at its settled pose while the
    # open hand reaches the grasp pose; ``lifted`` then raises the grasp goal so the grip lifts the
    # cupcake; ``above_plate_grasped`` carries it (still upside-down) over the plate before the in-hand
    # flip; the trailing duplicate ``on_plate`` is the hold segment.
    key_poses = (current, current, lifted, above_plate_grasped, above_plate_upright, on_plate, on_plate)
    return build_object_pose_sequence_from_keyframes(key_poses, segment_steps)


__all__ = [
    "DEFAULT_CUPCAKE_ON_PLATE_PLATE_POSE",
    "DEFAULT_CUPCAKE_UPRIGHT_QUAT",
    "DEFAULT_CUPCAKE_ON_PLATE_SEGMENT_STEPS",
    "DEFAULT_CUPCAKE_ON_PLATE_STAGE_OBJECT_TOLERANCES",
    "build_cupcake_on_plate_object_pose_sequence",
]
