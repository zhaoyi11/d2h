"""Direct low-level env wrapper for hierarchical dexterous manipulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from torch import Tensor

from .low_level import LowLevelRslRlPolicy


class DirectLowLevelEnvWrapper:
    """Expose high-level actions while a frozen RSL-RL policy controls the hand."""

    def __init__(
        self,
        env: gym.Env,
        low_level_policy: LowLevelRslRlPolicy,
        low_level_obs_group: str = "low_level",
        wrist_action_dim: int = 0,
    ) -> None:
        self.env = env
        self.low_level_policy = low_level_policy
        self.low_level_obs_group = low_level_obs_group
        self.wrist_action_dim = int(wrist_action_dim)
        if self.wrist_action_dim < 0:
            raise ValueError("wrist_action_dim must be non-negative")

        self.unwrapped = env.unwrapped
        self.num_envs = int(self.unwrapped.num_envs)
        self.device = self.unwrapped.device
        self.max_episode_length = int(getattr(self.unwrapped, "max_episode_length", 1_000_000))
        self.num_actions = self.wrist_action_dim

        expected_env_action_dim = self.wrist_action_dim + self.low_level_policy.action_dim
        actual_env_action_dim = int(getattr(self.unwrapped, "num_actions", env.single_action_space.shape[0]))
        if actual_env_action_dim != expected_env_action_dim:
            raise ValueError(
                f"Expected wrapped env action dim {expected_env_action_dim}, got {actual_env_action_dim}."
            )

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
            return self.unwrapped.observation_manager.compute()
        if hasattr(self.unwrapped, "_get_observations"):
            return self.unwrapped._get_observations()
        raise AttributeError("Wrapped env does not expose observations")

    def step(self, actions: Tensor) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        actions = actions.to(device=self.device)
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected high-level action shape {(self.num_envs, self.num_actions)}, got {actions.shape}.")
        obs = self._last_obs if self._last_obs is not None else self.get_observations()
        hand_action = self.low_level_policy.act(self._low_level_state(obs))
        if self.wrist_action_dim > 0:
            env_action = torch.cat((actions, hand_action), dim=-1)
        else:
            env_action = hand_action

        final_obs, reward, terminated, truncated, final_extras = self.env.step(env_action)
        self._last_obs = final_obs
        final_extras = dict(final_extras)
        final_extras["low_level_steps"] = 1
        return final_obs, reward, terminated, truncated, final_extras

    def close(self) -> None:
        return self.env.close()

    def _low_level_state(self, obs: dict[str, Tensor]) -> Tensor:
        if self.low_level_obs_group not in obs:
            raise KeyError(f"Observation group {self.low_level_obs_group!r} is missing from wrapped env observations.")
        state = obs[self.low_level_obs_group].to(device=self.device)
        state = state.reshape(self.num_envs, -1)
        if state.shape[-1] != self.low_level_policy.obs_dim:
            raise ValueError(f"Expected low-level observation dim {self.low_level_policy.obs_dim}, got {state.shape[-1]}.")
        return state
