"""IsaacLab reset event for a pick-insert Instant Dexterity archive."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase

from src.tasks.common.mdps.events import _env_ids_tensor

from .asset_geometry import insertion_geometry_from_assets
from .reset_dataset import (
    _sample_curriculum_state_indices,
    load_reset_state_pool,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetSceneFromInstantDexterity(ManagerTermBase):
    """Restore independently sampled full-scene poses with zero velocities."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        geometry = insertion_geometry_from_assets(
            env.scene["object"],
            env.scene["receptive_object"],
            env.device,
        )
        self._pool = load_reset_state_pool(
            env.cfg.reset_dataset_path,
            env.scene["robot"].joint_names,
            env.device,
            geometry,
        )

    def __call__(self, env: ManagerBasedEnv, env_ids) -> None:
        env_ids_t = _env_ids_tensor(env_ids, env.num_envs, env.device)
        curriculum = env.curriculum_manager.cfg.reset_state.func
        state_indices = _sample_curriculum_state_indices(
            num_states=self._pool.num_states,
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


__all__ = ["ResetSceneFromInstantDexterity"]
