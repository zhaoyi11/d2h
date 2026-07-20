# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-screw scripted-trajectory command term.

Thin task layer over the generic
:class:`~src.policy.high_level.trajectory_command.TrajectoryObjectAndHandBasePoseCommand`: it overrides
:meth:`_build_object_trajectories` with the pick-screw
reach->lift->move->align->approach->insert->twist->hold leg trajectory (built by
:func:`build_pick_screw_object_pose_sequence`). All the shared machinery (stepper, PI(D) anchor
correction, hand-base targeting, advance logic, drop recovery) is inherited.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.tasks.pick_screw.mdps.trajectory import (
    build_pick_screw_object_pose_sequence,
    DEFAULT_PICK_SCREW_RECEPTIVE_POSE,
    DEFAULT_PICK_SCREW_SEGMENT_STEPS,
    DEFAULT_PICK_SCREW_STAGE_OBJECT_TOLERANCES,
    DEFAULT_PICK_SCREW_THREAD_PITCH,
    DEFAULT_PICK_SCREW_TWIST_SEGMENTS,
    DEFAULT_PICK_SCREW_TWIST_TOTAL_ANGLE,
)


class PickScrewTrajectoryObjectAndHandBasePoseCommand(TrajectoryObjectAndHandBasePoseCommand):
    """Object and hand-base command that follows the pick-screw demo trajectory."""

    cfg: PickScrewTrajectoryObjectAndHandBasePoseCommandCfg

    def _build_object_trajectories(self, env_ids: torch.Tensor, current_pose_b: torch.Tensor) -> torch.Tensor:
        receptive_pose = torch.tensor(self.cfg.receptive_pose, dtype=current_pose_b.dtype, device=self.device)
        trajectories = [
            build_pick_screw_object_pose_sequence(
                current_pose_b[env_idx],
                receptive_pose=receptive_pose,
                segment_steps=self.cfg.trajectory_segment_steps,
                above_offset=self.cfg.above_offset,
                insertion_depth=self.cfg.insertion_depth,
                approach_height=self.cfg.approach_height,
                lift_height=self.cfg.lift_height,
                twist_total_angle=self.cfg.twist_total_angle,
                twist_segments=self.cfg.twist_segments,
                thread_pitch=self.cfg.thread_pitch,
            )
            for env_idx in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)


@configclass
class PickScrewTrajectoryObjectAndHandBasePoseCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    """Configuration for the pick-screw trajectory command."""

    class_type: type = PickScrewTrajectoryObjectAndHandBasePoseCommand

    receptive_pose: tuple[float, float, float, float, float, float, float] = DEFAULT_PICK_SCREW_RECEPTIVE_POSE
    """Target socket pose used by the pick-screw object trajectory."""

    trajectory_segment_steps: tuple[int, ...] = DEFAULT_PICK_SCREW_SEGMENT_STEPS
    """Interpolation samples for the reach, lift, move, align, approach, insert, twist, and hold
    segments. Must contain ``7 + twist_segments`` values."""

    above_offset: float = 0.10
    """Height above the socket for the initial move and orientation alignment."""

    insertion_depth: float = 0.06
    """Inserted object height offset above the socket pose."""

    approach_height: float = 0.01
    """Height above the insertion pose used before final descent."""

    lift_height: float = 0.03
    """Height (m) the lift-stage goal sits above the object's settled (reach) pose, so the grip must
    lift the object this far to advance out of the lift stage (an implicit grip-secured check)."""

    twist_total_angle: float = DEFAULT_PICK_SCREW_TWIST_TOTAL_ANGLE
    """Total rotation (rad) about the +Z insertion axis accumulated across the twist sub-segments."""

    twist_segments: int = DEFAULT_PICK_SCREW_TWIST_SEGMENTS
    """Number of <= 90 deg twist sub-segments the total twist is chained across (a full 2*pi turn needs
    >= 4, since slerp takes the short geodesic)."""

    thread_pitch: float = DEFAULT_PICK_SCREW_THREAD_PITCH
    """Scripted z descent (m) per full turn during the twist; 0 => pure in-place spin."""

    stage_object_tolerances: tuple[StageObjTol, ...] = DEFAULT_PICK_SCREW_STAGE_OBJECT_TOLERANCES
    """Per-stage object position and orientation tolerances for advancing the command trajectory.

    One entry is required for each trajectory segment: reach, lift, move, align, approach, insert, each
    twist sub-segment, and hold (``7 + twist_segments`` total)."""


__all__ = [
    "DEFAULT_PICK_SCREW_RECEPTIVE_POSE",
    "DEFAULT_PICK_SCREW_SEGMENT_STEPS",
    "DEFAULT_PICK_SCREW_STAGE_OBJECT_TOLERANCES",
    "DEFAULT_PICK_SCREW_THREAD_PITCH",
    "DEFAULT_PICK_SCREW_TWIST_SEGMENTS",
    "DEFAULT_PICK_SCREW_TWIST_TOTAL_ANGLE",
    "PickScrewTrajectoryObjectAndHandBasePoseCommand",
    "PickScrewTrajectoryObjectAndHandBasePoseCommandCfg",
]
