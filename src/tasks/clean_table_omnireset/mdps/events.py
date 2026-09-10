"""IsaacLab reset event for Instant Dexterity scene-state datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase

from src.tasks.common.mdps.events import _env_ids_tensor

from .geometry import outside_box_state_indices
from .reset_dataset import load_reset_state_pool
from .task_mdps import BOX_MAX, BOX_MIN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetSceneFromInstantDexterity(ManagerTermBase):
    """Restore independently sampled full-scene poses with zero velocities."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._pool = load_reset_state_pool(
            env.cfg.reset_dataset_dir,
            env.scene["robot"].joint_names,
            env.device,
        )
        self._outside_box_state_indices = outside_box_state_indices(
            self._pool.object_root_pose,
            self._pool.receptive_object_root_pose,
            BOX_MIN,
            BOX_MAX,
        )

    def __call__(self, env: ManagerBasedEnv, env_ids) -> None:
        env_ids_t = _env_ids_tensor(env_ids, env.num_envs, env.device)
        sample_indices = torch.randint(
            self._outside_box_state_indices.numel(),
            (env_ids_t.numel(),),
            device=env.device,
        )
        state_indices = self._outside_box_state_indices.index_select(0, sample_indices)
        env.scene.reset_to(
            self._pool.scene_state(state_indices),
            env_ids=env_ids_t,
            is_relative=True,
        )


__all__ = ["ResetSceneFromInstantDexterity"]
