"""HRL wrapper that delegates hand actions to a direct frozen BC policy."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from torch import Tensor

from .policy import DirectBcLowLevelPolicy


class DirectLowLevelEnvWrapper:
    """Expose a wrist-only HRL action interface over a direct BC hand policy."""

    def __init__(
        self,
        env: gym.Env,
        low_level_policy: DirectBcLowLevelPolicy,
        low_level_obs_group: str = "low_level",
        wrist_action_dim: int = 6,
    ) -> None:
        self.env = env
        self.low_level_policy = low_level_policy
        self.low_level_obs_group = low_level_obs_group
        self.wrist_action_dim = int(wrist_action_dim)
        if self.wrist_action_dim < 1:
            raise ValueError("wrist_action_dim must be positive")

        self.unwrapped = env.unwrapped
        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self.unwrapped.device
        self.max_episode_length = int(getattr(self.unwrapped, "max_episode_length", 1_000_000))
        self.num_actions = self.wrist_action_dim

        self.single_observation_space = self.unwrapped.single_observation_space
        self.observation_space = self.unwrapped.observation_space
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.render_mode = getattr(env, "render_mode", None)

        self._last_obs: dict[str, Tensor] | None = None

    def reset(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        obs, extras = self.env.reset()
        self._last_obs = obs
        return obs, extras

    def get_observations(self) -> dict[str, Tensor]:
        if hasattr(self.unwrapped, "observation_manager"):
            obs = self.unwrapped.observation_manager.compute()
        elif hasattr(self.unwrapped, "_get_observations"):
            obs = self.unwrapped._get_observations()
        else:
            raise AttributeError("Wrapped env does not expose observations")
        self._last_obs = obs
        return obs

    def step(self, actions: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        wrist_action = actions.to(device=self.device)
        if wrist_action.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected high-level action shape {(self.num_envs, self.num_actions)}, got {actions.shape}.")
        if self._last_obs is None:
            self._last_obs = self.get_observations()

        low_level_state = self._low_level_state(self._last_obs)
        hand_action = self.low_level_policy.act(low_level_state)
        if hand_action.shape != (self.num_envs, self.low_level_policy.action_dim):
            raise ValueError(
                "Expected direct low-level action shape "
                f"{(self.num_envs, self.low_level_policy.action_dim)}, got {hand_action.shape}."
            )

        env_action = torch.cat((wrist_action, hand_action.to(device=self.device, dtype=wrist_action.dtype)), dim=-1)
        obs, reward, terminated, truncated, extras = self.env.step(env_action)
        self._last_obs = obs
        extras = dict(extras)
        extras["low_level_steps"] = 1
        return obs, reward, terminated, truncated, extras

    def close(self) -> None:
        return self.env.close()

    def _low_level_state(self, obs: dict[str, Tensor]) -> Tensor:
        if self.low_level_obs_group not in obs:
            raise KeyError(f"Observation group {self.low_level_obs_group!r} is missing from wrapped env observations.")
        state = obs[self.low_level_obs_group].to(device=self.device)
        state = state.reshape(self.num_envs, -1)
        if state.shape[-1] != self.low_level_policy.obs_dim:
            raise ValueError(f"Expected direct BC state dim {self.low_level_policy.obs_dim}, got {state.shape[-1]}.")
        return state
