"""Task-specific manager terms for physical unscrewing."""

from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import subtract_frame_transforms

from src.tasks.common.mdps.rewards import contacts as good_object_contact
from src.tasks.unscrew.geometry import (
    BOLT_AABB_MAX,
    BOLT_AABB_MIN,
    SOCKET_TOP_Z,
    bolt_aabb_corners,
    bolt_bottom_clearance,
)


class StableUnscrewSuccess(ManagerTermBase):
    """Terminate after the bolt is clear and a stable two-finger grasp is retained."""

    def __init__(self, cfg: TerminationTermCfg, env) -> None:
        super().__init__(cfg, env)
        object_cfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        receptive_cfg = cfg.params.get("receptive_cfg", SceneEntityCfg("receptive_object"))
        self._object: RigidObject = env.scene[object_cfg.name]
        self._receptive: RigidObject = env.scene[receptive_cfg.name]
        self._bolt_corners = bolt_aabb_corners(env.device)

    def __call__(
        self,
        env,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        receptive_cfg: SceneEntityCfg = SceneEntityCfg("receptive_object"),
        clearance_margin: float = 0.030,
        force_threshold: float = 1.0,
    ) -> torch.Tensor:
        if clearance_margin < 0.0:
            raise ValueError("clearance_margin must be non-negative.")
        if force_threshold < 0.0:
            raise ValueError("force_threshold must be non-negative.")

        object_pos_r, object_quat_r = subtract_frame_transforms(
            self._receptive.data.root_pos_w,
            self._receptive.data.root_quat_w,
            self._object.data.root_pos_w,
            self._object.data.root_quat_w,
        )
        clear = (
            bolt_bottom_clearance(object_pos_r, object_quat_r, self._bolt_corners)
            >= clearance_margin
        )
        grasped = good_object_contact(env, force_threshold)
        valid = clear & grasped
        return valid


__all__ = [
    "BOLT_AABB_MAX",
    "BOLT_AABB_MIN",
    "SOCKET_TOP_Z",
    "StableUnscrewSuccess",
    "bolt_aabb_corners",
    "bolt_bottom_clearance",
]
