"""High-level chunk wrapper for hierarchical dexterous manipulation."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch
from torch import Tensor

from .low_level import LowLevelVAEPolicy


class HierarchicalChunkEnvWrapper:
    """Expose a chunk-level HRL action interface over a low-level IsaacLab env.

    High-level actions are ordered as ``[wrist_delta_6d, z]``. One high-level
    step decodes one low-level hand-action chunk and advances the wrapped env
    until the chunk ends or any vectorized environment terminates.
    """

    def __init__(
        self,
        env: gym.Env,
        low_level_policy: LowLevelVAEPolicy,
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
        self.num_actions = self.wrist_action_dim + self.low_level_policy.latent_dim

        self.single_observation_space = self.unwrapped.single_observation_space
        self.observation_space = self.unwrapped.observation_space
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,))
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)
        self.render_mode = getattr(env, "render_mode", None)

        self._state_history: Tensor | None = None
        self._action_history: Tensor | None = None

    def reset(self) -> tuple[dict[str, Tensor], dict[str, Any]]:
        obs, extras = self.env.reset()
        self._reset_history(obs)
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
        if self._state_history is None or self._action_history is None:
            self._reset_history(self.get_observations())

        wrist_delta = actions[:, : self.wrist_action_dim]
        z = actions[:, self.wrist_action_dim :]
        wrist_step = wrist_delta / float(self.low_level_policy.chunk_length)
        hand_chunk = self.low_level_policy.decode(self._context(), z)

        total_reward = torch.zeros(self.num_envs, device=self.device, dtype=hand_chunk.dtype)
        terminated_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        truncated_total = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        final_obs: dict[str, Tensor] | None = None
        final_extras: dict[str, Any] = {}
        low_level_steps = 0

        for chunk_idx in range(self.low_level_policy.chunk_length):
            hand_action = hand_chunk[:, chunk_idx, :]
            env_action = torch.cat((wrist_step, hand_action), dim=-1)
            final_obs, reward, terminated, truncated, final_extras = self.env.step(env_action)
            low_level_steps += 1
            total_reward += reward
            terminated_total |= terminated.bool()
            truncated_total |= truncated.bool()
            self._append_history(final_obs, hand_action)
            if bool((terminated_total | truncated_total).any()):
                break

        if final_obs is None:
            raise RuntimeError("No low-level steps were executed")
        final_extras = dict(final_extras)
        final_extras["low_level_steps"] = low_level_steps
        return final_obs, total_reward, terminated_total, truncated_total, final_extras

    def close(self) -> None:
        return self.env.close()

    def _reset_history(self, obs: dict[str, Tensor]) -> None:
        state = self._low_level_state(obs)
        self._state_history = state.unsqueeze(1).repeat(1, self.low_level_policy.config.past_length, 1)
        self._action_history = torch.zeros(
            self.num_envs,
            self.low_level_policy.config.past_length,
            self.low_level_policy.action_dim,
            device=self.device,
            dtype=state.dtype,
        )

    def _append_history(self, obs: dict[str, Tensor], hand_action: Tensor) -> None:
        assert self._state_history is not None
        assert self._action_history is not None
        state = self._low_level_state(obs)
        self._state_history = torch.cat((self._state_history[:, 1:, :], state.unsqueeze(1)), dim=1)
        self._action_history = torch.cat((self._action_history[:, 1:, :], hand_action.unsqueeze(1)), dim=1)

    def _context(self) -> Tensor:
        assert self._state_history is not None
        assert self._action_history is not None
        context = torch.cat((self._state_history, self._action_history), dim=-1)
        return context.reshape(self.num_envs, -1)

    def _low_level_state(self, obs: dict[str, Tensor]) -> Tensor:
        if self.low_level_obs_group not in obs:
            raise KeyError(f"Observation group {self.low_level_obs_group!r} is missing from wrapped env observations.")
        state = obs[self.low_level_obs_group].to(device=self.device)
        state = state.reshape(self.num_envs, -1)
        if state.shape[-1] != self.low_level_policy.state_dim:
            raise ValueError(f"Expected low-level state dim {self.low_level_policy.state_dim}, got {state.shape[-1]}.")
        return state
