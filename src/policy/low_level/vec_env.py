"""RSL-RL VecEnv that drives a frozen hand policy while the env's MPC drives the arm."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from tensordict import TensorDict
from torch import Tensor

from rsl_rl.env import VecEnv

from .policy import LowLevelRslRlPolicy


class FrozenHandVecEnv(VecEnv):
    """Expose an rsl-rl VecEnv whose only actuation is a frozen RSL-RL hand policy.

    The Franka arm is governed by the env's cuRobo MPC action term (which consumes zero action
    dims and tracks the trajectory command), so the high-level policy has no action to emit
    (``num_actions == 0``). Each step this wrapper computes the 16-DOF LEAP hand action from the
    frozen low-level policy and steps the underlying env with it.
    """

    def __init__(
        self,
        env: gym.Env,
        low_level_policy: LowLevelRslRlPolicy,
        low_level_obs_group: str = "low_level",
    ) -> None:
        self.env = env
        self.low_level_policy = low_level_policy
        self.low_level_obs_group = low_level_obs_group

        self.unwrapped = env.unwrapped
        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self.unwrapped.device
        self.max_episode_length = int(getattr(self.unwrapped, "max_episode_length", 1_000_000))
        self.num_actions = 0  # high-level policy has no action; MPC owns the arm

        # Safety check: the env's only real action dims are the frozen hand policy's.
        env_action_dim = int(getattr(self.unwrapped, "num_actions", env.single_action_space.shape[0]))
        if env_action_dim != self.low_level_policy.action_dim:
            raise ValueError(
                f"Expected env action dim {self.low_level_policy.action_dim}, got {env_action_dim}."
            )

        self.single_observation_space = self.unwrapped.single_observation_space
        self.observation_space = self.unwrapped.observation_space
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.render_mode = getattr(env, "render_mode", None)
        self._last_obs: dict[str, Tensor] | None = None
        self.reset()

    @property
    def cfg(self) -> object:
        return self.unwrapped.cfg

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

    def reset(self) -> tuple[TensorDict, dict[str, Any]]:
        obs, extras = self.env.reset()
        self._last_obs = obs
        return TensorDict(obs, batch_size=[self.num_envs]), extras

    def get_observations(self) -> TensorDict:
        return TensorDict(self._compute_obs(), batch_size=[self.num_envs])

    def step(
        self, actions: Tensor
    ) -> tuple[TensorDict, Tensor, Tensor, dict[str, Any]]:  # actions carry no arm command
        obs = self._last_obs if self._last_obs is not None else self._compute_obs()
        hand_action = self.low_level_policy.act(self._low_level_state(obs))
        obs_dict, reward, terminated, truncated, extras = self.env.step(hand_action)
        self._last_obs = obs_dict
        dones = (terminated | truncated).to(dtype=torch.long)
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return TensorDict(obs_dict, batch_size=[self.num_envs]), reward, dones, extras

    def close(self) -> None:
        return self.env.close()

    def _compute_obs(self) -> dict[str, Tensor]:
        if hasattr(self.unwrapped, "observation_manager"):
            return self.unwrapped.observation_manager.compute()
        if hasattr(self.unwrapped, "_get_observations"):
            return self.unwrapped._get_observations()
        raise AttributeError("Wrapped env does not expose observations")

    def _low_level_state(self, obs: dict[str, Tensor]) -> Tensor:
        if self.low_level_obs_group not in obs:
            raise KeyError(f"Observation group {self.low_level_obs_group!r} is missing from wrapped env observations.")
        state = obs[self.low_level_obs_group].to(device=self.device)
        state = state.reshape(self.num_envs, -1)
        if state.shape[-1] != self.low_level_policy.obs_dim:
            raise ValueError(f"Expected low-level observation dim {self.low_level_policy.obs_dim}, got {state.shape[-1]}.")
        return state
