# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cupcake-on-plate scripted-trajectory command term.

Thin task layer over the generic
:class:`~src.policy.high_level.trajectory_command.TrajectoryObjectAndHandBasePoseCommand`: it overrides
:meth:`_build_object_trajectories` with the cupcake reach->lift->move->reorient->place->hold trajectory
(built by :func:`build_cupcake_on_plate_object_pose_sequence`). All the shared machinery (stepper,
PI(D) anchor correction, hand-base targeting, advance logic) is inherited.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.tasks.cupcake_on_plate.mdps.trajectory import (
    build_cupcake_on_plate_object_pose_sequence,
    DEFAULT_CUPCAKE_ON_PLATE_PLATE_POSE,
    DEFAULT_CUPCAKE_UPRIGHT_QUAT,
    DEFAULT_CUPCAKE_ON_PLATE_SEGMENT_STEPS,
    DEFAULT_CUPCAKE_ON_PLATE_STAGE_OBJECT_TOLERANCES,
)


class CupcakeOnPlateTrajectoryObjectAndHandBasePoseCommand(TrajectoryObjectAndHandBasePoseCommand):
    """Object and hand-base command that follows the cupcake pick->reorient->place trajectory."""

    cfg: CupcakeOnPlateTrajectoryObjectAndHandBasePoseCommandCfg

    def _build_object_trajectories(self, env_ids: torch.Tensor, current_pose_b: torch.Tensor) -> torch.Tensor:
        plate_pose = torch.tensor(self.cfg.plate_pose, dtype=current_pose_b.dtype, device=self.device)
        upright_quat = torch.tensor(self.cfg.upright_quat, dtype=current_pose_b.dtype, device=self.device)
        trajectories = [
            build_cupcake_on_plate_object_pose_sequence(
                current_pose_b[env_idx],
                plate_pose=plate_pose,
                upright_quat=upright_quat,
                segment_steps=self.cfg.trajectory_segment_steps,
                lift_height=self.cfg.lift_height,
                above_offset=self.cfg.above_offset,
                place_height=self.cfg.place_height,
            )
            for env_idx in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)


@configclass
class CupcakeOnPlateTrajectoryObjectAndHandBasePoseCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    """Configuration for the cupcake-on-plate trajectory command."""

    class_type: type = CupcakeOnPlateTrajectoryObjectAndHandBasePoseCommand

    plate_pose: tuple[float, float, float, float, float, float, float] = DEFAULT_CUPCAKE_ON_PLATE_PLATE_POSE
    """Target plate pose used by the cupcake-on-plate object trajectory (only x/y and z are used)."""

    upright_quat: tuple[float, float, float, float] = DEFAULT_CUPCAKE_UPRIGHT_QUAT
    """Canonical upright orientation the reorient segment flips the cupcake goal to."""

    trajectory_segment_steps: tuple[int, ...] = DEFAULT_CUPCAKE_ON_PLATE_SEGMENT_STEPS
    """Interpolation samples for the reach, lift, move, reorient, place, and hold segments."""

    lift_height: float = 0.02
    """Height (m) the lift-stage goal sits above the cupcake's settled (reach) pose, so the grip must
    lift the cupcake this far to advance out of the lift stage (an implicit grip-secured check). Kept
    small -- just enough to break the cupcake free of the table before it is carried across to the
    plate; the in-hand flip happens later, above the plate (see ``above_offset``)."""

    above_offset: float = 0.10
    """Height above the plate at which the cupcake is carried (move, still upside-down) and then
    flipped upright (reorient). Must clear the plate so the 180 deg in-hand flip does not collide."""

    place_height: float = 0.03
    """Height above the plate pose at which the cupcake is placed. Plate pose z is the plate bottom
    (on the table); at scale 0.8 the plate is ~0.03 m thick, so ~0.03 puts the upright cupcake's
    bottom on the plate top."""

    stage_object_tolerances: tuple[StageObjTol, ...] = DEFAULT_CUPCAKE_ON_PLATE_STAGE_OBJECT_TOLERANCES
    """Per-stage object position and orientation tolerances for advancing the command trajectory.

    One entry is required for each trajectory segment: reach, lift, move, reorient, place, and hold.
    """


__all__ = [
    "DEFAULT_CUPCAKE_ON_PLATE_PLATE_POSE",
    "DEFAULT_CUPCAKE_UPRIGHT_QUAT",
    "DEFAULT_CUPCAKE_ON_PLATE_SEGMENT_STEPS",
    "DEFAULT_CUPCAKE_ON_PLATE_STAGE_OBJECT_TOLERANCES",
    "CupcakeOnPlateTrajectoryObjectAndHandBasePoseCommand",
    "CupcakeOnPlateTrajectoryObjectAndHandBasePoseCommandCfg",
]
