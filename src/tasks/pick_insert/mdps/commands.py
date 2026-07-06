# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pick-insert scripted-trajectory command term.

Thin task layer over the generic
:class:`~src.policy.high_level.trajectory_command.TrajectoryObjectAndHandBasePoseCommand`: it overrides
:meth:`_build_object_trajectories` with the pick-insert move->align->approach->insert->hold peg
trajectory (built by :func:`build_pick_insert_object_pose_sequence`). All the shared machinery
(stepper, PI(D) anchor correction, hand-base targeting, pregrasp reach, advance logic) is inherited.
"""

from __future__ import annotations

import torch

from isaaclab.utils import configclass

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.tasks.pick_insert.mdps.trajectory import (build_pick_insert_object_pose_sequence,
                                                   DEFAULT_PICK_INSERT_RECEPTIVE_POSE,
                                                   DEFAULT_PICK_INSERT_SEGMENT_STEPS,
                                                   DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES)


class PickInsertTrajectoryObjectAndHandBasePoseCommand(TrajectoryObjectAndHandBasePoseCommand):
    """Object and hand-base command that follows the pick-insert demo trajectory."""

    cfg: PickInsertTrajectoryObjectAndHandBasePoseCommandCfg

    def _build_object_trajectories(self, env_ids: torch.Tensor, current_pose_b: torch.Tensor) -> torch.Tensor:
        receptive_pose = torch.tensor(self.cfg.receptive_pose, dtype=current_pose_b.dtype, device=self.device)
        trajectories = [
            build_pick_insert_object_pose_sequence(
                current_pose_b[env_idx],
                receptive_pose=receptive_pose,
                segment_steps=self.cfg.trajectory_segment_steps,
                above_offset=self.cfg.above_offset,
                insertion_depth=self.cfg.insertion_depth,
                approach_height=self.cfg.approach_height,
            )
            for env_idx in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)


@configclass
class PickInsertTrajectoryObjectAndHandBasePoseCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    """Configuration for the pick-insert trajectory command."""

    class_type: type = PickInsertTrajectoryObjectAndHandBasePoseCommand

    receptive_pose: tuple[float, float, float, float, float, float, float] = DEFAULT_PICK_INSERT_RECEPTIVE_POSE
    """Target receptacle pose used by the pick-insert object trajectory."""

    trajectory_segment_steps: tuple[int, int, int, int, int] = DEFAULT_PICK_INSERT_SEGMENT_STEPS
    """Interpolation samples for the move, align, approach, insert, and hold segments."""

    above_offset: float = 0.10
    """Height above the receptacle for the initial move and orientation alignment."""

    insertion_depth: float = 0.06
    """Inserted object height offset above the receptacle pose."""

    approach_height: float = 0.01
    """Height above the insertion pose used before final descent."""

    stage_object_tolerances: tuple[StageObjTol, ...] = DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES
    """Per-stage object position and orientation tolerances for advancing the command trajectory.

    One entry is required for each trajectory segment: move, align, approach, insert, and hold.
    """


__all__ = [
    "DEFAULT_PICK_INSERT_RECEPTIVE_POSE",
    "DEFAULT_PICK_INSERT_SEGMENT_STEPS",
    "DEFAULT_PICK_INSERT_STAGE_OBJECT_TOLERANCES",
    "PickInsertTrajectoryObjectAndHandBasePoseCommand",
    "PickInsertTrajectoryObjectAndHandBasePoseCommandCfg",
]
