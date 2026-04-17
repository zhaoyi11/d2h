"""Adapter that exposes a D2H Isaac Lab gym env as a FlashSAC-compatible VectorEnv.

FlashSAC's ``flash_rl.envs.IsaacLabVectorEnv`` launches its own ``AppLauncher`` and
``parse_env_cfg`` — neither fits D2H's flow, which already uses Hydra to build
``env_cfg`` and ``gym.make(task, cfg=env_cfg)`` from ``scripts/flash_sac/train.py``.
This adapter takes the already-constructed gym env and presents the same contract
FlashSAC consumes (see ``FlashSAC_dex/flash_rl/envs.py:IsaacLabVectorEnv``).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Sequence, Union, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

NDArray = npt.NDArray[Any]
F32NDArray = npt.NDArray[np.float32]
Tensor = Union[NDArray, torch.Tensor]


def _recursive_to_numpy(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, dict):
        return {k: _recursive_to_numpy(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_recursive_to_numpy(v) for v in data)
    return data


def _concat_groups(obs_dict: dict[str, torch.Tensor], groups: Sequence[str]) -> torch.Tensor:
    tensors = [obs_dict[g] for g in groups]
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=-1)


class FlashSACIsaacLabAdapter(
    VectorEnv[Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray], Union[torch.Tensor, F32NDArray]]
):
    """Wrap a live Isaac Lab gym env in the interface FlashSAC's trainer expects.

    Observation groups: D2H env configs expose multiple named groups in
    ``observations`` (e.g. ``policy``, ``perception``, ``proprio``). We mirror
    RSL-RL's ``obs_groups`` convention — a dict with ``policy`` and optionally
    ``critic`` keys, each a list of group names to concatenate along the last
    dim. When ``critic`` differs from ``policy`` the adapter enables FlashSAC's
    asymmetric-observation path (critic sees a wider vector than actor).
    """

    def __init__(
        self,
        env: gym.Env,
        device: str,
        action_bounds: float = 1.0,
        obs_groups: dict[str, list[str]] | None = None,
        to_numpy: bool = True,
    ) -> None:
        self._env = env
        self.device = device
        self.to_numpy = to_numpy
        self.action_bounds = action_bounds

        unwrapped: Any = env.unwrapped
        self.num_envs = int(unwrapped.num_envs)
        self.max_episode_steps = int(getattr(unwrapped, "max_episode_length", 1_000_000))

        single_obs_space = unwrapped.single_observation_space
        available_groups = list(single_obs_space.keys())
        if obs_groups is None:
            obs_groups = {"policy": ["policy"]}
        self._actor_groups: list[str] = list(obs_groups["policy"])
        self._critic_groups: list[str] = list(obs_groups.get("critic", self._actor_groups))
        for g in self._actor_groups + self._critic_groups:
            if g not in available_groups:
                raise KeyError(
                    f"obs_group '{g}' not found in env.single_observation_space "
                    f"(available: {available_groups})"
                )

        actor_dim = sum(int(single_obs_space[g].shape[-1]) for g in self._actor_groups)
        critic_dim = sum(int(single_obs_space[g].shape[-1]) for g in self._critic_groups)
        self.obs_size: tuple[int, ...] = (actor_dim,)
        self.asymmetric_obs = self._critic_groups != self._actor_groups

        if self.asymmetric_obs:
            self.critic_obs_size: tuple[int, ...] = (critic_dim,)
            self.single_observation_space = gym.spaces.Box(
                low=0.0,
                high=0.0,
                shape=(actor_dim + critic_dim,),
                dtype=np.float32,
            )
        else:
            self.critic_obs_size = (0,)
            self.single_observation_space = gym.spaces.Box(
                low=0.0, high=0.0, shape=(actor_dim,), dtype=np.float32
            )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

        self.action_size = tuple(unwrapped.single_action_space.shape)
        self.single_action_space = gym.spaces.Box(
            low=-1.0 * action_bounds,
            high=1.0 * action_bounds,
            shape=self.action_size,
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        # Per-env episodic return / length trackers. FlashSAC's loop gates wandb
        # reward logging on infos["episode_info"] being present; upstream's
        # IsaacLabVectorEnv relies on evaluate() to produce returns, but that
        # path is unavailable under Isaac Lab's single-SimulationApp constraint.
        # We compute rolling train-time returns here instead.
        self._ep_return = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._ep_length = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        fifo_cap = max(self.num_envs * 4, 64)
        self._ep_return_fifo: deque[float] = deque(maxlen=fifo_cap)
        self._ep_length_fifo: deque[int] = deque(maxlen=fifo_cap)

    def _build_obs(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        actor_obs = _concat_groups(obs_dict, self._actor_groups)
        if self.asymmetric_obs:
            critic_obs = _concat_groups(obs_dict, self._critic_groups)
            full = torch.cat((actor_obs, critic_obs), dim=-1)
            return full, critic_obs
        return actor_obs, None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        random_start_init: bool = True,
    ) -> tuple[Tensor, dict[str, Any]]:
        obs_dict, infos = self._env.reset()
        obs, _ = self._build_obs(obs_dict)

        unwrapped: Any = self._env.unwrapped
        if random_start_init and hasattr(unwrapped, "episode_length_buf"):
            unwrapped.episode_length_buf = torch.randint_like(
                unwrapped.episode_length_buf, high=int(self.max_episode_steps)
            )

        self._ep_return.zero_()
        self._ep_length.zero_()

        if self.to_numpy:
            obs = obs.cpu().numpy()
            infos = _recursive_to_numpy(infos)
        infos.update({"actor_observation_size": self.obs_size, "asymmetric_obs": self.asymmetric_obs})
        return obs, infos

    def step(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        if isinstance(actions, torch.Tensor):
            torch_actions = actions.to(self.device)
        else:
            torch_actions = torch.from_numpy(np.asarray(actions)).to(self.device)
        torch_actions = torch.clamp(torch_actions, -1.0, 1.0) * self.action_bounds

        obs_dict, rew, terminations, truncations, extras = cast(Any, self._env.step(torch_actions))
        obs, critic_obs = self._build_obs(obs_dict)

        # NOTE: IsaacLab auto-resets terminated envs inside step(), so obs here is
        # already the post-reset state for those envs. Off-policy training needs the
        # true terminal observation for the replay buffer. There is no clean way to
        # recover it from IsaacLab (see issue isaac-sim/IsaacLab#1362), so we mirror
        # FlashSAC's upstream workaround and use the current obs as final_obs.
        infos: dict[str, Any] = {
            "time_outs": truncations,
            "observations": {"critic": critic_obs},
            "final_obs": obs,
        }

        self._ep_return += rew
        self._ep_length += 1
        done_mask = terminations | truncations
        episode_info = self._build_episode_info(done_mask, extras)
        if episode_info:
            infos["episode_info"] = episode_info

        if self.to_numpy:
            obs = obs.cpu().numpy()
            rew = rew.cpu().numpy()
            terminations = terminations.cpu().numpy()
            truncations = truncations.cpu().numpy()
            infos = _recursive_to_numpy(infos)
        return obs, rew, terminations, truncations, infos

    def _build_episode_info(self, done_mask: torch.Tensor, extras: dict[str, Any]) -> dict[str, float]:
        """Produce the dict FlashSAC's train loop logs under ``infos["episode_info"]``.

        Combines two sources:
        - Per-env running return / length we track here → ``episode/return``,
          ``episode/length``. Only emitted on steps where at least one env reset,
          since those are the only steps that produce fresh samples.
        - IsaacLab's ``extras["log"]`` — per-reset-batch mean of
          ``Episode_Reward/{term}`` and ``Episode_Termination/{term}`` entries,
          populated by ``RewardManager.reset()``. NaN entries (no reset this
          step) are filtered out.
        """
        info: dict[str, float] = {}

        if bool(done_mask.any()):
            finished_returns = self._ep_return[done_mask].detach().cpu().tolist()
            finished_lengths = self._ep_length[done_mask].detach().cpu().tolist()
            self._ep_return_fifo.extend(finished_returns)
            self._ep_length_fifo.extend(int(x) for x in finished_lengths)
            self._ep_return[done_mask] = 0.0
            self._ep_length[done_mask] = 0

        if self._ep_return_fifo:
            info["episode/return"] = float(np.mean(self._ep_return_fifo))
            info["episode/length"] = float(np.mean(self._ep_length_fifo))

        log = extras.get("log") if isinstance(extras, dict) else None
        if isinstance(log, dict):
            for key, value in log.items():
                if not (key.startswith("Episode_Reward/") or key.startswith("Episode_Termination/")):
                    continue
                scalar = value.item() if isinstance(value, torch.Tensor) else float(value)
                if math.isnan(scalar):
                    continue
                info[key] = scalar

        unwrapped: Any = self._env.unwrapped
        cmd_mgr = getattr(unwrapped, "command_manager", None)
        if cmd_mgr is not None:
            try:
                term = cmd_mgr.get_term("object_pose")
                success = term.metrics.get("consecutive_success")
                if isinstance(success, torch.Tensor):
                    info["episode/consecutive_success"] = float(success.float().mean().item())
            except (KeyError, AttributeError, ValueError):
                pass

        return info

    def close(self, **kwargs: Any) -> None:
        # Leave the underlying env + simulation_app alive; the trainer script owns
        # lifecycle (matches FlashSAC's upstream IsaacLabVectorEnv.close no-op).
        return

    def render(self) -> None:
        raise NotImplementedError("Rendering is not supported; use --video at the gym layer instead.")
