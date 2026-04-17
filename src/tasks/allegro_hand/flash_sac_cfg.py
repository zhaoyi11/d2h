"""FlashSAC training config for Isaac-Repose-Cube-Allegro-Direct-v0."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tasks.common.flash_sac_cfg_base import FlashSacCfgBase


@dataclass
class AllegroHandFlashSacCfg(FlashSacCfgBase):
    experiment_name: str = "allegro_repose_cube_flash_sac"
    # DirectRL env exposes a single "policy" observation group (124-dim).
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"policy": ["policy"]}
    )
    action_bounds: float = 1.0
