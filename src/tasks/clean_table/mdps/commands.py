from __future__ import annotations

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.tasks.common.mdps.placement import PlacementCommand, PlacementCommandCfg
from src.tasks.clean_table.mdps.trajectory import (
    DEFAULT_BOX_ABOVE_OFFSET,
    DEFAULT_BOX_TARGET_OFFSET,
    DEFAULT_CLEAN_TABLE_SEGMENT_STEPS,
    DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES,
    build_clean_table_object_pose_sequence,
)


class CleanTableTrajectoryObjectAndHandBasePoseCommand(PlacementCommand):
    """Build the clean-table path using the live object and box poses."""

    def _build_object_trajectories(
        self, env_ids: torch.Tensor, current_pose_b: torch.Tensor
    ) -> torch.Tensor:
        box_pos_b, box_quat_b = subtract_frame_transforms(
            self.robot.data.root_pos_w[env_ids],
            self.robot.data.root_quat_w[env_ids],
            self.box.data.root_pos_w[env_ids],
            self.box.data.root_quat_w[env_ids],
        )
        box_pose_b = torch.cat((box_pos_b, box_quat_b), dim=1)
        trajectories = [
            build_clean_table_object_pose_sequence(
                current_pose_b[index],
                box_pose_b[index],
                segment_steps=self.cfg.trajectory_segment_steps,
                box_target_offset=self.cfg.box_target_offset,
                above_box_offset=self.cfg.above_box_offset,
            )
            for index in range(env_ids.numel())
        ]
        return torch.stack(trajectories, dim=0)


@configclass
class CleanTableTrajectoryObjectAndHandBasePoseCommandCfg(PlacementCommandCfg):
    class_type: type = CleanTableTrajectoryObjectAndHandBasePoseCommand
    trajectory_segment_steps: tuple[int, ...] = DEFAULT_CLEAN_TABLE_SEGMENT_STEPS
    stage_object_tolerances: tuple[StageObjTol, ...] = (
        DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES
    )
    box_target_offset: tuple[float, float, float] = DEFAULT_BOX_TARGET_OFFSET
    above_box_offset: tuple[float, float, float] = DEFAULT_BOX_ABOVE_OFFSET


__all__ = [
    "CleanTableTrajectoryObjectAndHandBasePoseCommand",
    "CleanTableTrajectoryObjectAndHandBasePoseCommandCfg",
]
