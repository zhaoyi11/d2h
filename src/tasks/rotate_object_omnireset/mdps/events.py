"""IsaacLab reset event for recorded rotate-object states and goals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase

from .reset_dataset import (
    _sample_curriculum_state_indices,
    _sample_hardest_state_indices,
    load_reset_state_pool,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids_tensor(env_ids, num_envs: int, device: str | torch.device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, slice):
        start = 0 if env_ids.start is None else env_ids.start
        stop = num_envs if env_ids.stop is None else env_ids.stop
        step = 1 if env_ids.step is None else env_ids.step
        return torch.arange(start, stop, step, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


class ResetSceneFromInstantDexterity(ManagerTermBase):
    """Restore one sampled scene and its recorded goal for each reset environment."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pool = load_reset_state_pool(
            env.cfg.reset_dataset_dir,
            env.scene["robot"].joint_names,
            env.device,
        )

    def __call__(self, env: ManagerBasedEnv, env_ids, hardest_only: bool = False) -> None:
        env_ids_t = _env_ids_tensor(env_ids, env.num_envs, env.device)
        if hardest_only:
            state_indices = _sample_hardest_state_indices(
                self._pool.progress,
                count=env_ids_t.numel(),
                device=env.device,
            )
        else:
            curriculum = env.curriculum_manager.cfg.reset_state.func
            state_indices = _sample_curriculum_state_indices(
                self._pool.progress,
                count=env_ids_t.numel(),
                stage=curriculum.current_stage,
                num_stages=curriculum.num_stages,
                device=env.device,
            )
        env.scene.reset_to(
            self._pool.scene_state(state_indices),
            env_ids=env_ids_t,
            is_relative=True,
        )
        command = env.command_manager.get_term("object_pose")
        command.set_pending_goals(
            env_ids_t,
            self._pool.goal_pose_b.index_select(0, state_indices),
            self._pool.hand_base_command_b.index_select(0, state_indices),
        )


__all__ = ["ResetSceneFromInstantDexterity"]
