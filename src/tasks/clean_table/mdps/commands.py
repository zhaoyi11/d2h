from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.policy.high_level.trajectory_stepper import StageObjTol
from src.tasks.common.mdps.rewards import contacts as good_object_contact
from src.tasks.clean_table.mdps.trajectory import (
    DEFAULT_BOX_ABOVE_OFFSET,
    DEFAULT_BOX_TARGET_OFFSET,
    DEFAULT_CLEAN_TABLE_SEGMENT_STEPS,
    DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES,
    build_clean_table_object_pose_sequence,
)


class CleanTableTrajectoryObjectAndHandBasePoseCommand(
    TrajectoryObjectAndHandBasePoseCommand
):
    """Follow a live-box pick and release trajectory, then return the hand home."""

    cfg: "CleanTableTrajectoryObjectAndHandBasePoseCommandCfg"

    _GRASP_STAGE = 1
    _RELEASE_STAGE = 4
    _RETREAT_STAGE = 5

    def __init__(self, cfg, env) -> None:
        if len(cfg.trajectory_segment_steps) != 6:
            raise ValueError("trajectory_segment_steps must contain 6 values.")
        if cfg.trajectory_segment_steps[self._RELEASE_STAGE] < 1:
            raise ValueError("the release stage must contain at least one step.")
        if cfg.grasp_contact_stable_steps < 1:
            raise ValueError("grasp_contact_stable_steps must be at least 1.")
        if cfg.grasp_timeout_steps < cfg.grasp_contact_stable_steps:
            raise ValueError("grasp_timeout_steps must be at least grasp_contact_stable_steps.")
        if cfg.success_stable_steps < 1:
            raise ValueError("success_stable_steps must be at least 1.")
        super().__init__(cfg, env)

        default_pos_b, default_quat_b = self._current_hand_base_pose_b()
        self._default_hand_base_pose_b = torch.cat(
            (default_pos_b, default_quat_b), dim=1
        ).detach().clone()

        self.box: RigidObject = env.scene[cfg.box_name]
        self.table: RigidObject = env.scene[cfg.table_name]
        self._grasp_contact_streak = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._grasp_phase_steps = torch.zeros_like(self._grasp_contact_streak)
        self._success_streak = torch.zeros_like(self._grasp_contact_streak)

        self.object_height_above_table = torch.zeros(self.num_envs, device=self.device)
        self.object_to_box_distance = torch.zeros(self.num_envs, device=self.device)
        self.object_pos_box = torch.zeros(self.num_envs, 3, device=self.device)
        self.lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.inside_box = torch.zeros_like(self.lifted)
        self.released = torch.zeros_like(self.lifted)
        self.hand_clear = torch.zeros_like(self.lifted)
        self.success = torch.zeros_like(self.lifted)

        self._box_target = torch.tensor(cfg.box_target_offset, device=self.device)
        self._box_min = torch.tensor(cfg.box_min, device=self.device)
        self._box_max = torch.tensor(cfg.box_max, device=self.device)
        self.metrics["object_height_above_table"] = self.object_height_above_table
        self.metrics["object_to_box_distance"] = self.object_to_box_distance
        self.metrics["inside_box"] = self.inside_box.float()
        self.metrics["released"] = self.released.float()
        self.metrics["hand_clear"] = self.hand_clear.float()
        self.metrics["success"] = self.success.float()

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

    def _update_hand_base_pose_command(self, env_ids=slice(None)) -> None:
        super()._update_hand_base_pose_command(env_ids)
        default_pose = getattr(self, "_default_hand_base_pose_b", None)
        if default_pose is None or getattr(self, "_stepper", None) is None:
            return
        retreating = self._current_stage(env_ids) >= self._RETREAT_STAGE
        self.hand_base_pose_command_b[env_ids] = torch.where(
            retreating.unsqueeze(-1),
            default_pose[env_ids],
            self.hand_base_pose_command_b[env_ids],
        )

    def _update_metrics(self) -> None:
        super()._update_metrics()
        if self.cfg.capture_goal_after_settle:
            self._trajectory_command_achieved &= self._grasp_goal_captured
        self._update_grasp_establish()

        table_top = self.table.data.root_pos_w[:, 2] + self.cfg.table_half_height
        self.object_height_above_table[:] = self.object.data.root_pos_w[:, 2] - table_top
        self.lifted[:] = self.object_height_above_table > self.cfg.lift_height

        object_pos_box, _ = subtract_frame_transforms(
            self.box.data.root_pos_w,
            self.box.data.root_quat_w,
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
        )
        self.object_pos_box[:] = object_pos_box
        self.object_to_box_distance[:] = torch.norm(object_pos_box - self._box_target, dim=1)
        self.inside_box[:] = ((object_pos_box >= self._box_min) & (object_pos_box <= self._box_max)).all(dim=1)

        stage = self._current_stage()
        self.released[:] = stage >= self._RELEASE_STAGE
        hand_base_pos_b, _ = self._current_hand_base_pose_b()
        object_pos_b, _ = self._current_object_pose_b()
        self.hand_clear[:] = (
            torch.norm(hand_base_pos_b - object_pos_b, dim=1)
            > self.cfg.hand_clear_distance
        )
        object_speed = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
        stable_success = (
            self.inside_box
            & (stage >= self._RETREAT_STAGE)
            & self._trajectory_command_achieved
            & self.hand_clear
            & (object_speed < self.cfg.success_settle_speed)
        )
        self._success_streak = torch.where(
            stable_success,
            self._success_streak + 1,
            torch.zeros_like(self._success_streak),
        )
        self.success[:] = self._success_streak >= self.cfg.success_stable_steps
        self.metrics["inside_box"] = self.inside_box.float()
        self.metrics["released"] = self.released.float()
        self.metrics["hand_clear"] = self.hand_clear.float()
        self.metrics["success"] = self.success.float()
        self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()

    def _update_grasp_establish(self) -> None:
        stage = self._current_stage()
        establishing = stage == self._GRASP_STAGE
        contact = good_object_contact(self._env, self.cfg.grasp_contact_force_threshold)
        self._grasp_phase_steps = torch.where(
            establishing,
            self._grasp_phase_steps + 1,
            torch.zeros_like(self._grasp_phase_steps),
        )
        self._grasp_contact_streak = torch.where(
            establishing & contact,
            self._grasp_contact_streak + 1,
            torch.zeros_like(self._grasp_contact_streak),
        )
        confirmed = self._grasp_contact_streak >= self.cfg.grasp_contact_stable_steps
        self._trajectory_command_achieved[establishing] = confirmed[establishing]
        timed_out = establishing & ~confirmed & (
            self._grasp_phase_steps >= self.cfg.grasp_timeout_steps
        )
        retry_ids = timed_out.nonzero().flatten()
        if retry_ids.numel() > 0:
            self._resample_command(retry_ids)
            self._steps_since_reset[retry_ids] = 0
            self._grasp_goal_captured[retry_ids] = False

    def _object_target_achieved(self) -> torch.Tensor:
        achieved = super()._object_target_achieved()
        return achieved | (self._current_stage() >= self._RELEASE_STAGE)

    def _update_keep_hand_open_metric(self, env_ids=slice(None)) -> None:
        stage = self._current_stage(env_ids)
        keep_open = (stage <= self.cfg.hand_open_until_stage) | (
            stage >= self._RELEASE_STAGE
        )
        self.metrics["keep_hand_open"][env_ids] = keep_open.float()

    def _apply_objanchor_correction(self, env_ids: torch.Tensor) -> None:
        stage = self._current_stage(env_ids)
        correction_active = (stage != self._GRASP_STAGE) & (
            stage < self._RELEASE_STAGE
        )
        super()._apply_objanchor_correction(env_ids[correction_active])

    def _anchor_correction(self, env_ids=slice(None)) -> torch.Tensor | None:
        correction = super()._anchor_correction(env_ids)
        if correction is None or getattr(self, "_stepper", None) is None:
            return correction
        stage = self._current_stage(env_ids)
        return torch.where(
            (stage < self._RELEASE_STAGE).unsqueeze(-1),
            correction,
            torch.zeros_like(correction),
        )

    def _update_drop_recovery(self) -> None:
        if not self.cfg.enable_drop_recovery:
            return
        stage = self._current_stage()
        recoverable = stage < self._RELEASE_STAGE
        in_transport = stage > self.cfg.recovery_arm_after_stage
        near = self.metrics["hand_base_object_error"] < self.cfg.drop_object_hand_distance
        object_speed = torch.norm(self.object.data.root_lin_vel_w, dim=-1)
        at_rest = object_speed < self.cfg.recovery_settle_speed
        settled = self._recovery.update(
            secured=recoverable & in_transport & near,
            dropped=recoverable & in_transport & ~near,
            at_rest=recoverable & at_rest,
        )
        if settled.numel() > 0:
            self._resample_command(settled)

    def _resample_command(self, env_ids) -> None:
        ids = self._env_ids_tensor(env_ids)
        super()._resample_command(ids)
        for name in ("_grasp_contact_streak", "_grasp_phase_steps", "_success_streak"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer[ids] = 0
        for name in ("inside_box", "released", "hand_clear", "success"):
            buffer = getattr(self, name, None)
            if buffer is not None:
                buffer[ids] = False

    def _current_stage(self, env_ids=slice(None)) -> torch.Tensor:
        return self._stepper.step_to_stage[self._stepper.step[env_ids]]


@configclass
class CleanTableTrajectoryObjectAndHandBasePoseCommandCfg(
    TrajectoryObjectAndHandBasePoseCommandCfg
):
    class_type: type = CleanTableTrajectoryObjectAndHandBasePoseCommand

    box_name: str = "receptive_object"
    table_name: str = "table"
    trajectory_segment_steps: tuple[int, ...] = DEFAULT_CLEAN_TABLE_SEGMENT_STEPS
    stage_object_tolerances: tuple[StageObjTol, ...] = (
        DEFAULT_CLEAN_TABLE_STAGE_OBJECT_TOLERANCES
    )
    lift_height: float = 0.08
    box_target_offset: tuple[float, float, float] = DEFAULT_BOX_TARGET_OFFSET
    above_box_offset: tuple[float, float, float] = DEFAULT_BOX_ABOVE_OFFSET
    box_min: tuple[float, float, float] = (-0.09, -0.15, 0.005)
    box_max: tuple[float, float, float] = (0.09, 0.15, 0.105)
    table_half_height: float = 0.02
    grasp_contact_force_threshold: float = 1.0
    grasp_contact_stable_steps: int = 3
    grasp_timeout_steps: int = 30
    hand_clear_distance: float = 0.10
    success_settle_speed: float = 0.05
    success_stable_steps: int = 5


__all__ = [
    "CleanTableTrajectoryObjectAndHandBasePoseCommand",
    "CleanTableTrajectoryObjectAndHandBasePoseCommandCfg",
]
