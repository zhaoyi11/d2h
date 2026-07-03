"""RSL-RL adapter for the chunk-level HRL environment wrapper."""

from __future__ import annotations

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.env import VecEnv

from .wrapper import DirectLowLevelEnvWrapper


class HrlRslRlVecEnvWrapper(VecEnv):
    """Minimal RSL-RL VecEnv wrapper that respects HRL action dimensions."""

    def __init__(self, env: DirectLowLevelEnvWrapper, clip_actions: float | None = None) -> None:
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = env.num_envs
        self.device = env.device
        self.max_episode_length = env.max_episode_length
        self.num_actions = env.num_actions
        self._modify_action_space()
        self.env.reset()

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self.env.action_space

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.env.reset()
        return TensorDict(obs_dict, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        return TensorDict(self.env.get_observations(), batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras

    def close(self) -> None:
        return self.env.close()

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return
        self.env.single_action_space = gym.spaces.Box(
            low=-self.clip_actions,
            high=self.clip_actions,
            shape=(self.num_actions,),
        )
        self.env.action_space = gym.vector.utils.batch_space(self.env.single_action_space, self.num_envs)
