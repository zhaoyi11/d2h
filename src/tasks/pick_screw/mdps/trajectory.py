# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-screw object-pose trajectory builder.

The scripted reach->lift->move->align->approach->insert->twist->hold leg trajectory that is specific to
the pick-screw task, composed from the task-agnostic primitives in :mod:`src.policy.high_level.utils`.
It reproduces the pick-insert reach->...->insert peg trajectory and then adds a *twist* phase: once the
leg is inserted, the object-goal orientation accumulates yaw about the world/base +Z axis (the leg's
long/insertion axis, standing vertical in the socket) to screw it in.

The twist is a direct generalization of the cupcake-on-plate in-hand flip (hold position, drive the
object-goal quaternion): the object goal drives ``command[:, :7]`` -- the target the frozen low-level
hand policy chases -- while the grasp anchor stays root-aligned, so the frozen hand policy plus the
bounded PID anchor correction physically execute the rotation *in-hand*.

Because :func:`~src.policy.high_level.utils.slerp` always takes the short geodesic, a single keyframe
pair can encode at most ~180 deg of rotation. A full turn is therefore chained across ``twist_segments``
sub-segments each <= 90 deg (unambiguous), each composing an incremental yaw about +Z onto the inserted
orientation.

Pure ``torch`` so it can be unit-tested in isolation; the command term that consumes it lives in
:mod:`src.tasks.pick_screw.mdps.commands`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.utils import (
    _as_pose_tensor,
    _with_normalized_quat,
    build_object_pose_sequence_from_keyframes,
    compose_world_yaw,
)


# Socket pose (x, y, z, qw, qx, qy, qz) in the robot base frame. Matches the scene ``ReceptiveObject``
# (SquareTableTop) init pose; its identity orientation is the aligned/inserted (vertical-standing) leg
# orientation the twist rotates about.
DEFAULT_PICK_SCREW_RECEPTIVE_POSE = (0.55, 0.0, 0.271, 1.0, 0.0, 0.0, 0.0)

# Twist phase: total rotation about +Z and how many <= 90 deg sub-segments to chain it across (a full
# 2*pi turn needs >= 4). ``thread_pitch`` is the (scripted) z descent per full turn; 0 => pure in-place
# spin (the USDs have no real threads, so any descent is purely kinematic -- see the task plan).
DEFAULT_PICK_SCREW_TWIST_SEGMENTS = 4
DEFAULT_PICK_SCREW_TWIST_TOTAL_ANGLE = 2.0 * math.pi
DEFAULT_PICK_SCREW_THREAD_PITCH = 0.0

# Number of interpolation steps for each segment: reach / lift / move / align / approach / insert /
# (twist x twist_segments) / hold. The reach segment is 0-step: it adds no waypoints, so it just
# relabels the initial waypoint (the leg held at its settled pose) as its own stage while the (open)
# hand reaches the grasp pose over the leg. Length must equal ``7 + twist_segments``.
DEFAULT_PICK_SCREW_SEGMENT_STEPS = (0, 1, 2, 1, 1, 1) + (2, 2, 2, 2) + (1,)

# Per-stage object position/orientation tolerances for advancing the command trajectory. One entry per
# segment. The twist stages keep a loose orientation tolerance (like cupcake's reorient stage) so the
# stepper advances under partial progress instead of stalling on the in-hand rotation.
DEFAULT_PICK_SCREW_STAGE_OBJECT_TOLERANCES = (
    StageObjTol(0.02, 0.3),   # reach (object held at its settled pose; advance is arm-gated, hand open)
    StageObjTol(0.02, 0.1),   # lift (goal lift_height above the object; the grip must lift it there)
    StageObjTol(0.02, 0.3),   # move
    StageObjTol(0.02, 0.2),   # align
    StageObjTol(0.01, 0.2),   # approach
    StageObjTol(0.005, 0.1),  # insert
    StageObjTol(0.01, 0.8),   # twist_1 (in-hand yaw about +Z; loose orientation tol)
    StageObjTol(0.01, 0.8),   # twist_2
    StageObjTol(0.01, 0.8),   # twist_3
    StageObjTol(0.01, 0.8),   # twist_4
    StageObjTol(0.01, 0.3),   # hold (at the fully-screwed pose)
)


def build_pick_screw_object_pose_sequence(
    current_pose: torch.Tensor | Sequence[float],
    receptive_pose: torch.Tensor | Sequence[float] = DEFAULT_PICK_SCREW_RECEPTIVE_POSE,
    segment_steps: Sequence[int] = DEFAULT_PICK_SCREW_SEGMENT_STEPS,
    above_offset: float = 0.15,
    insertion_depth: float = 0.015,
    approach_height: float = 0.08,
    lift_height: float = 0.03,
    twist_total_angle: float = DEFAULT_PICK_SCREW_TWIST_TOTAL_ANGLE,
    twist_segments: int = DEFAULT_PICK_SCREW_TWIST_SEGMENTS,
    thread_pitch: float = DEFAULT_PICK_SCREW_THREAD_PITCH,
) -> torch.Tensor:
    """Build an interpolated pick->insert->twist leg object-pose trajectory.

    Poses use ``(x, y, z, qw, qx, qy, qz)`` in the robot base frame. The returned sequence holds the leg
    at ``current_pose`` (reach, while the open hand moves to the grasp pose), raises the grasp goal
    ``lift_height`` above it (lift, a grip-secured check), moves above the socket, aligns to the socket
    orientation, descends to the inserted pose, then *twists* it in by accumulating yaw about the +Z
    (insertion) axis across ``twist_segments`` sub-segments, and holds at the fully-screwed pose --
    ``7 + twist_segments`` segments: reach / lift / move / align / approach / insert / twist... / hold.
    The reach segment is typically 0-step (see :data:`DEFAULT_PICK_SCREW_SEGMENT_STEPS`), so it adds no
    waypoints and the leg never moves during it. The keyframes are interpolated by the shared
    :func:`~src.policy.high_level.utils.build_object_pose_sequence_from_keyframes` core.
    """
    if twist_segments < 1:
        raise ValueError("twist_segments must be >= 1.")
    if len(segment_steps) != 7 + twist_segments:
        raise ValueError(f"segment_steps must contain {7 + twist_segments} values.")
    if any(steps < 0 for steps in segment_steps):
        raise ValueError("segment_steps values must be non-negative.")
    d_theta = twist_total_angle / twist_segments
    if abs(d_theta) >= math.pi:
        raise ValueError(
            "per-sub-segment yaw (twist_total_angle / twist_segments) must be < pi so the slerp "
            "geodesic goes the right way; increase twist_segments."
        )

    current = _with_normalized_quat(_as_pose_tensor(current_pose))
    receptive = _with_normalized_quat(
        _as_pose_tensor(receptive_pose, dtype=current.dtype, device=current.device)
    )
    q_ins = receptive[3:7]  # aligned/inserted (vertical-standing) leg orientation; twist rotates it.

    # Lift keyframe: the leg's settled pose raised straight up by lift_height (same orientation).
    lifted = torch.cat(
        (
            torch.stack((current[0], current[1], current[2] + current.new_tensor(lift_height))),
            current[3:7],
        )
    )

    above_current = torch.cat(
        (
            torch.stack(
                (
                    receptive[0],
                    receptive[1],
                    receptive[2] + current.new_tensor(above_offset),
                )
            ),
            current[3:7],
        )
    )
    above_aligned = torch.cat((above_current[:3], q_ins))
    approach = torch.cat(
        (
            torch.stack(
                (
                    receptive[0],
                    receptive[1],
                    receptive[2] + current.new_tensor(insertion_depth + approach_height),
                )
            ),
            q_ins,
        )
    )
    inserted = torch.cat(
        (
            torch.stack(
                (
                    receptive[0],
                    receptive[1],
                    receptive[2] + current.new_tensor(insertion_depth),
                )
            ),
            q_ins,
        )
    )
    inserted_z = inserted[2]

    # Twist keyframes: accumulate yaw about +Z, optionally descend thread_pitch per full turn. Each
    # keyframe differs from the previous by exactly d_theta (< pi), so the slerp between them is a clean
    # d_theta yaw about Z in the correct direction.
    twist_poses = []
    for j in range(1, twist_segments + 1):
        angle_j = j * d_theta
        q_j = compose_world_yaw(q_ins, angle_j)
        z_j = inserted_z - current.new_tensor(thread_pitch * (angle_j / (2.0 * math.pi)))
        twist_poses.append(torch.cat((torch.stack((receptive[0], receptive[1], z_j)), q_j)))

    # Leading ``current`` is the reach keyframe: the object goal is held at its settled pose while the
    # open hand reaches the grasp pose; ``lifted`` then raises the grasp goal; the trailing duplicate of
    # the last twist keyframe is the hold segment (dwell at the fully-screwed pose).
    key_poses = (
        current,
        current,
        lifted,
        above_current,
        above_aligned,
        approach,
        inserted,
        *twist_poses,
        twist_poses[-1],
    )
    return build_object_pose_sequence_from_keyframes(key_poses, segment_steps)


__all__ = [
    "DEFAULT_PICK_SCREW_RECEPTIVE_POSE",
    "DEFAULT_PICK_SCREW_SEGMENT_STEPS",
    "DEFAULT_PICK_SCREW_STAGE_OBJECT_TOLERANCES",
    "DEFAULT_PICK_SCREW_THREAD_PITCH",
    "DEFAULT_PICK_SCREW_TWIST_SEGMENTS",
    "DEFAULT_PICK_SCREW_TWIST_TOTAL_ANGLE",
    "build_pick_screw_object_pose_sequence",
]
