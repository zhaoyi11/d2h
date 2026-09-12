"""D2H simulation mirror. Import only after AppLauncher starts Isaac Sim."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms
from src.tasks.rotate_knob_distill.env_cfg import RotateKnobDistillEnvCfg_PLAY
from src.tasks.rotate_knob_distill.mdps.commands import yaw_from_quat


class SimMirror:
    def __init__(self, device="cpu", seed=0):
        cfg = RotateKnobDistillEnvCfg_PLAY()
        cfg.sim.device = device
        cfg.seed = seed
        cfg.events.reset_hand_pose.params["pose_range"] = {}
        self.env = ManagerBasedRLEnv(cfg)
        self.robot = self.env.scene["robot"]
        self.knob = self.env.scene["object"]
        self.action = self.env.action_manager.get_term("hand_action")
        self.joint_ids = self.action._joint_ids
        self.env.reset()

    def reset(self, joint_pos, angle, target, velocity=0.0):
        self.env.reset()
        q = joint_pos.to(self.env.device).reshape(1, 16)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=self.joint_ids)
        self.robot.set_joint_position_target(q, joint_ids=self.joint_ids)
        self.action._prev_applied_actions[:] = q
        state = self.knob.data.default_root_state.clone()
        state[:, :3] += self.env.scene.env_origins
        zero = torch.zeros(1, device=self.env.device)
        state[:, 3:7] = quat_from_euler_xyz(zero, zero, zero + angle)
        state[:, 7:] = 0
        state[:, 12] = velocity
        self.knob.write_root_state_to_sim(state)
        command = self.env.command_manager.get_term("object_pose")
        command.pose_command_w[:, :3] = state[:, :3]
        command.pose_command_w[:, 3:] = quat_from_euler_xyz(zero, zero, zero + target)
        pos, quat = subtract_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w,
                                             command.pose_command_w[:, :3], command.pose_command_w[:, 3:])
        command.pose_command_b[:] = torch.cat((pos, quat), dim=-1)
        self.env.scene.write_data_to_sim()
        self.env.scene.update(0.0)

    def read(self):
        return (self.robot.data.joint_pos[0, self.joint_ids].detach().cpu().float(),
                float(yaw_from_quat(self.knob.data.root_quat_w)[0]),
                float(self.knob.data.root_ang_vel_w[0, 2]))

    def step(self, applied_target):
        # Targets already include deployment EMA and slew limiting. Do not process them again.
        target = applied_target.to(self.env.device).reshape(1, 16)
        self.robot.set_joint_position_target(target, joint_ids=self.joint_ids)
        self.action._prev_applied_actions[:] = target
        for _ in range(self.env.cfg.decimation):
            self.env._sim_step_counter += 1
            self.env.scene.write_data_to_sim()
            self.env.sim.step(render=False)
            self.env.scene.update(self.env.physics_dt)
        if self.env.sim.has_gui():
            self.env.sim.render()
        return self.read()

    def render(self):
        if self.env.sim.has_gui():
            self.env.sim.render()

    def disconnect(self):
        self.env.close()
