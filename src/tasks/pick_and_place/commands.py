"""Recorded carry with shared grasp, release and retreat gates."""

import numpy as np
import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

from src.policy.high_level.trajectory_stepper import StageObjTol
from src.tasks.common.mano_anchor import initialize_mano_anchor, update_mano_anchor_command
from src.tasks.common.mdps.placement import PlacementCommand, PlacementCommandCfg
from src.tasks.pick_and_place.trajectory import (
    DEFAULT_TRAJECTORY, build_pick_and_place_trajectory, load_carry_trajectory,
)


class PickAndPlaceTrajectoryCommand(PlacementCommand):
    def __init__(self, cfg, env):
        self._carry, frames, cfg.trajectory_segment_steps = load_carry_trajectory(
            cfg.trajectory_path, cfg.carry_start_frame, cfg.carry_end_frame, cfg.trajectory_stride
        )
        initialize_mano_anchor(self, cfg, env, frames)
        super().__init__(cfg, env)
        self._source_frames = torch.as_tensor(frames, device=self.device)
        self.metrics["pick_and_place_frame"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["pick_and_place_progress"] = torch.zeros(self.num_envs, device=self.device)

    def _update_hand_base_pose_command(self, env_ids=slice(None)):
        super()._update_hand_base_pose_command(env_ids)
        if self._mano_object_anchor is not None:
            ids = self._env_ids_tensor(env_ids)
            if getattr(self, "_stepper", None) is not None:
                ids = ids[self._current_stage(ids) < self._RETREAT_STAGE]
            update_mano_anchor_command(self, ids)

    def _build_object_trajectories(self, env_ids, current_pose_b):
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        pos_w, quat_w = combine_frame_transforms(
            root_pos, root_quat, current_pose_b[:, :3], current_pose_b[:, 3:]
        )
        current = torch.cat((pos_w, quat_w), dim=-1).detach().cpu().numpy()
        boxes = self.box.data.root_state_w[env_ids, :7].detach().cpu().numpy()
        trajectories = torch.as_tensor(
            np.stack([
                build_pick_and_place_trajectory(
                    self._carry, pose, box, box_target_offset=self.cfg.box_target_offset
                ) for pose, box in zip(current, boxes)
            ]), dtype=current_pose_b.dtype, device=self.device,
        )
        pos_b, quat_b = subtract_frame_transforms(
            root_pos[:, None].expand_as(trajectories[..., :3]),
            root_quat[:, None].expand_as(trajectories[..., 3:]),
            trajectories[..., :3], trajectories[..., 3:],
        )
        return torch.cat((pos_b, quat_b), dim=-1)

    def _update_keep_hand_open_metric(self, env_ids=slice(None)):
        super()._update_keep_hand_open_metric(env_ids)
        self.metrics["pick_and_place_frame"][env_ids] = self._source_frames[self._stepper.step[env_ids]].float()
        self.metrics["pick_and_place_progress"][env_ids] = (
            self._stepper.step[env_ids].float() / (self._stepper.length - 1)
        )


@configclass
class PickAndPlaceTrajectoryCommandCfg(PlacementCommandCfg):
    class_type: type = PickAndPlaceTrajectoryCommand
    trajectory_segment_steps: tuple[int, ...] = (0, 1, 1, 0, 1, 1)
    stage_object_tolerances: tuple[StageObjTol, ...] = (
        StageObjTol(0.04, 0.60),
        StageObjTol(0.04, 0.60),
        StageObjTol(0.04, 0.60),
        StageObjTol(0.04, 0.60),
        StageObjTol(0.04, 0.60),
        StageObjTol(0.04, 0.60),
    )
    # Required by the shared placement config; unused by this direct-to-drop task.
    above_box_offset: tuple[float, float, float] = (0.0, 0.0, 0.25)
    trajectory_path: str = str(DEFAULT_TRAJECTORY)
    anchor_calibration_path: str | None = None
    carry_start_frame: int = 320
    carry_end_frame: int = 400
    trajectory_stride: int = 20
    # The pig needs time to seat in the fingers before the arm starts lifting.
    grasp_contact_stable_steps: int = 15
    grasp_timeout_steps: int = 90
    box_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.15)
