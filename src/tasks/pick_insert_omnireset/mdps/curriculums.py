"""Adaptive reset-state curriculum for pick-insert OmniReset."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ManagerTermBase
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _env_ids_tensor(env_ids, num_envs: int, device: str | torch.device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, slice):
        start = 0 if env_ids.start is None else env_ids.start
        stop = num_envs if env_ids.stop is None else env_ids.stop
        step = 1 if env_ids.step is None else env_ids.step
        return torch.arange(start, stop, step, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


class ResetStateCurriculum(ManagerTermBase):
    """Adjust the active reset suffix from batches of completed-episode success."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.num_stages = int(cfg.params.get("num_stages", 10))
        self.current_stage = int(cfg.params.get("initial_stage", 0))
        self._completed_count = 0
        self._success_count = 0
        self.latest_success_rate = 0.0
        self._success_term = env.reward_manager.get_term_cfg("success").func

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        initial_stage: int = 0,
        num_stages: int = 10,
        evaluation_batch_size: int = 4096,
        promote_threshold: float = 0.7,
        demote_threshold: float = 0.5,
    ) -> dict[str, float]:
        env_ids_t = _env_ids_tensor(env_ids, env.num_envs, env.device)
        finished_ids = env_ids_t[env.episode_length_buf[env_ids_t] > 0]
        if finished_ids.numel() > 0:
            self._completed_count += int(finished_ids.numel())
            self._success_count += int(
                self._success_term.episode_succeeded[finished_ids].sum().item()
            )

        if self._completed_count >= evaluation_batch_size:
            self.latest_success_rate = self._success_count / self._completed_count
            if self.latest_success_rate >= promote_threshold:
                self.current_stage = min(self.current_stage + 1, num_stages - 1)
            elif self.latest_success_rate <= demote_threshold:
                self.current_stage = max(self.current_stage - 1, 0)
            self._completed_count = 0
            self._success_count = 0

        return {
            "stage": float(self.current_stage),
            "active_fraction": (self.current_stage + 1) / num_stages,
            "batch_progress": self._completed_count / evaluation_batch_size,
            "success_rate": self.latest_success_rate,
        }


@configclass
class CurriculumCfg:
    reset_state = CurrTerm(
        func=ResetStateCurriculum,
        params={
            "initial_stage": 0,
            "num_stages": 10,
            "evaluation_batch_size": 4096,
            "promote_threshold": 0.7,
            "demote_threshold": 0.3,
        },
    )


__all__ = ["CurriculumCfg", "ResetStateCurriculum"]
