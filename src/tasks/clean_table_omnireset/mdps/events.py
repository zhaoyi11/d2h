"""IsaacLab reset event for Instant Dexterity scene-state datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import EventTermCfg, ManagerTermBase

from .geometry import outside_box_state_indices
from .reset_dataset import load_reset_state_pool
from .task_mdps import BOX_MAX, BOX_MIN

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _env_ids_tensor(env_ids, num_envs: int, device: torch.device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(num_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, slice):
        start = 0 if env_ids.start is None else env_ids.start
        stop = num_envs if env_ids.stop is None else env_ids.stop
        step = 1 if env_ids.step is None else env_ids.step
        return torch.arange(start, stop, step, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


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
