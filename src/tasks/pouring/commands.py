"""Pouring references and contact-latched bottle gravity for MPC + frozen hand RL."""

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_mul, subtract_frame_transforms

from src.policy.high_level.trajectory_command import (
    TrajectoryObjectAndHandBasePoseCommand,
    TrajectoryObjectAndHandBasePoseCommandCfg,
)
from src.policy.high_level.trajectory_stepper import StageObjTol
from src.policy.high_level.utils import hand_base_pose_from_object_command_b
from src.tasks.common.mdps.rewards import contacts
from src.tasks.pouring.trajectory import DEFAULT_TRAJECTORY, load_pouring_trajectory


class PouringTrajectoryCommand(TrajectoryObjectAndHandBasePoseCommand):
    cfg: "PouringTrajectoryCommandCfg"

    def __init__(self, cfg, env):
        poses, frames, segments = load_pouring_trajectory(
            cfg.trajectory_path, cfg.trajectory_stride, cfg.grasp_frame
        )
        if cfg.grasp_contact_stable_steps < 1 or cfg.success_stable_steps < 1:
            raise ValueError("Contact and success stability counts must be positive.")
        cfg.trajectory_segment_steps = segments
        super().__init__(cfg, env)
        self._demo_poses = torch.as_tensor(poses, device=self.device)
        self._source_frames = torch.as_tensor(frames, device=self.device)
        self._grasp_streak = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_streak = torch.zeros_like(self._grasp_streak)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.gravity_enabled = torch.zeros_like(self.success)
        self._gravity_reset_step = torch.full_like(self._grasp_streak, -1)
        self._bottle_contact = env.scene[cfg.bottle_contact_sensor]
        self._gravity = torch.tensor(env.cfg.sim.gravity, device=self.device)
        self._mass = self.object.data.default_mass.to(self.device)
        for name in ("pouring_frame", "pouring_progress", "pouring_success", "gravity_enabled", "bottle_hand_contact_force"):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

    def _build_object_trajectories(self, env_ids, current_pose_b):
        # A translation preserves the recording's object rotations and relative motion.
        # Environment origins are world offsets, not robot-root offsets.
        translation_w = self._demo_poses.new_tensor(self.cfg.initial_object_position) - self._demo_poses[0, :3]
        translation_w = translation_w + self._env.scene.env_origins[env_ids]
        positions_w = self._demo_poses[None, :, :3] + translation_w[:, None, :]
        quaternions_w = self._demo_poses[None, :, 3:].expand(len(env_ids), -1, -1)
        root_pos = self.robot.data.root_pos_w[env_ids, None, :]
        root_quat = self.robot.data.root_quat_w[env_ids, None, :].expand_as(quaternions_w)
        positions_b, quaternions_b = subtract_frame_transforms(
            root_pos, root_quat, positions_w, quaternions_w
        )
        return torch.cat((positions_b, quaternions_b), dim=-1)

    def _update_hand_base_pose_command(self, env_ids=slice(None)):
        ids = self._env_ids_tensor(env_ids)
        # Keep the object center; right-multiply to rotate +90 degrees about local X.
        offset = torch.zeros_like(self.pose_command_b[ids])
        rotation_x = offset.new_tensor((0.5**0.5, 0.5**0.5, 0.0, 0.0)).expand(len(ids), -1)
        offset[:, 3:] = quat_mul(self.pose_command_b[ids, 3:], rotation_x)
        hand, anchor = hand_base_pose_from_object_command_b(
            self.pose_command_b[ids], self._hand_base_to_anchor_pose[ids], offset,
            self._anchor_correction(ids),
        )
        self.hand_base_pose_command_b[ids] = hand
        self.anchor_pose_command_b[ids] = anchor

    def _update_keep_hand_open_metric(self, env_ids=slice(None)):
        super()._update_keep_hand_open_metric(env_ids)
        # The LEAP hand can touch earlier than the MANO grasp frame. Engage the policy
        # at that first contact instead of pushing a now gravity-loaded bottle while open.
        self.metrics["keep_hand_open"][env_ids] *= (~self.gravity_enabled[env_ids]).float()

    def _apply_objanchor_correction(self, env_ids):
        stage = self._stepper.step_to_stage[self._stepper.step[env_ids]]
        super()._apply_objanchor_correction(env_ids[stage == 2])

    def _update_metrics(self):
        super()._update_metrics()
        stage = self._stepper.step_to_stage[self._stepper.step]
        grasped = contacts(self._env, self.cfg.grasp_contact_force_threshold)
        self._grasp_streak = torch.where(
            (stage == 1) & grasped, self._grasp_streak + 1, 0
        )
        self._trajectory_command_achieved &= (stage != 1) | (
            self._grasp_streak >= self.cfg.grasp_contact_stable_steps
        )
        final = self._stepper.step == self._stepper.length - 1
        self._success_streak = torch.where(
            final & self._trajectory_command_achieved, self._success_streak + 1, 0
        )
        self.success[:] = self._success_streak >= self.cfg.success_stable_steps
        self.metrics["trajectory_command_achieved"] = self._trajectory_command_achieved.float()
        self.metrics["pouring_success"] = self.success.float()

    def _write_gravity_force(self):
        forces = self._mass[:, :, None] * self._gravity * self.gravity_enabled[:, None, None]
        # Write the full batch: IsaacLab's external-wrench enable flag is shared across envs.
        self.object.set_external_force_and_torque(forces, torch.zeros_like(forces), is_global=True)
        self.metrics["gravity_enabled"] = self.gravity_enabled.float()

    def compute(self, dt):
        force_matrix = self._bottle_contact.data.force_matrix_w_history
        contact = torch.linalg.vector_norm(force_matrix, dim=-1).flatten(1).amax(dim=1)
        self.metrics["bottle_hand_contact_force"] = contact
        fresh = self._env._sim_step_counter > self._gravity_reset_step
        self.gravity_enabled |= fresh & (contact > self.cfg.gravity_contact_force_threshold)
        self._write_gravity_force()
        super().compute(dt)
        self.metrics["pouring_frame"] = self._source_frames[self._stepper.step].float()
        self.metrics["pouring_progress"] = self._stepper.step.float() / (self._stepper.length - 1)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._grasp_streak[env_ids] = 0
        self._success_streak[env_ids] = 0
        self.success[env_ids] = False
        for name in ("pouring_frame", "pouring_progress", "pouring_success"):
            self.metrics[name][env_ids] = 0

    def reset(self, env_ids=None):
        ids = self._env_ids_tensor(slice(None) if env_ids is None else env_ids)
        extras = super().reset(ids)
        self.gravity_enabled[ids] = False
        self._gravity_reset_step[ids] = self._env._sim_step_counter
        self._update_keep_hand_open_metric(ids)
        self._write_gravity_force()
        return extras


@configclass
class PouringTrajectoryCommandCfg(TrajectoryObjectAndHandBasePoseCommandCfg):
    class_type: type = PouringTrajectoryCommand
    trajectory_path: str = str(DEFAULT_TRAJECTORY)
    trajectory_stride: int = 28
    grasp_frame: int = 0
    initial_object_position: tuple[float, float, float] = (0.55, 0.10, 0.40)
    # Derived from the selected NPZ in the command constructor, before creating its stepper.
    trajectory_segment_steps: tuple[int, ...] = ()
    stage_object_tolerances: tuple[StageObjTol, ...] = (StageObjTol(0.05, 0.60),) * 3
    hand_base_position_tolerance: float = 0.05
    hand_base_orientation_tolerance: float = 0.40
    hand_open_until_stage: int = -1
    bottle_contact_sensor: str = "bottle_hand_contact"
    gravity_contact_force_threshold: float = 0.05
    grasp_contact_force_threshold: float = 1.0
    grasp_contact_stable_steps: int = 3
    success_stable_steps: int = 5
