"""Task-specific manager terms for physical unscrewing."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from src.tasks.common.mdps.rewards import contacts as good_object_contact


# Collision bounds authored in the UW square-leg USD, scaled by the task's 1.5 asset scale.
BOLT_AABB_MIN = (-0.018282, -0.017355, -0.084987)
BOLT_AABB_MAX = (0.018281, 0.019207, -0.043598)
SOCKET_TOP_Z = 0.0233475


def _aabb_corners(
    lower: Sequence[float], upper: Sequence[float], *, device: str
) -> torch.Tensor:
    """Return the eight corners of an axis-aligned box as an ``(8, 3)`` tensor."""
    lo = torch.tensor(lower, dtype=torch.float32, device=device)
    hi = torch.tensor(upper, dtype=torch.float32, device=device)
    return torch.stack(
        [
            torch.stack((x, y, z))
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
            for z in (lo[2], hi[2])
        ]
    )


class StableUnscrewSuccess(ManagerTermBase):
    """Terminate after the bolt is clear and a stable two-finger grasp is retained."""

    def __init__(self, cfg: TerminationTermCfg, env) -> None:
        super().__init__(cfg, env)
        object_cfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        receptive_cfg = cfg.params.get("receptive_cfg", SceneEntityCfg("receptive_object"))
        self._object: RigidObject = env.scene[object_cfg.name]
        self._receptive: RigidObject = env.scene[receptive_cfg.name]
        self._bolt_corners = _aabb_corners(BOLT_AABB_MIN, BOLT_AABB_MAX, device=env.device)
        self._stable_counter = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._stable_counter[slice(None) if env_ids is None else env_ids] = 0

    def __call__(
        self,
        env,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
        clearance_margin: float = 0.030,
        force_threshold: float = 1.0,
        stable_steps: int = 5,
    ) -> torch.Tensor:
        if clearance_margin < 0.0:
            raise ValueError("clearance_margin must be non-negative.")
        if force_threshold < 0.0:
            raise ValueError("force_threshold must be non-negative.")
        if stable_steps < 1:
            raise ValueError("stable_steps must be at least 1.")

        object_pos_r, object_quat_r = subtract_frame_transforms(
            self._receptive.data.root_pos_w,
            self._receptive.data.root_quat_w,
            self._object.data.root_pos_w,
            self._object.data.root_quat_w,
        )
        num_envs = object_pos_r.shape[0]
        corner_quat = object_quat_r[:, None, :].expand(-1, 8, -1).reshape(-1, 4)
        local_corners = self._bolt_corners[None, :, :].expand(num_envs, -1, -1).reshape(-1, 3)
        corner_pos_r = quat_apply(corner_quat, local_corners).reshape(num_envs, 8, 3)
        corner_pos_r = corner_pos_r + object_pos_r[:, None, :]
        clear = corner_pos_r[:, :, 2].amin(dim=1) >= SOCKET_TOP_Z + clearance_margin
        grasped = good_object_contact(env, force_threshold)
        valid = clear & grasped
        self._stable_counter = torch.where(
            valid, self._stable_counter + 1, torch.zeros_like(self._stable_counter)
        )
        return self._stable_counter >= stable_steps


__all__ = [
    "BOLT_AABB_MAX",
    "BOLT_AABB_MIN",
    "SOCKET_TOP_Z",
    "StableUnscrewSuccess",
]
