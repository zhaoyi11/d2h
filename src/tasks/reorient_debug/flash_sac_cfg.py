"""FlashSAC training config for Reorient-v0 (Leap-hand in-hand re-orientation)."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.tasks.common.flash_sac_cfg_base import FlashSacCfgBase


@dataclass
class LeapObjectFlashSacCfg(FlashSacCfgBase):
    experiment_name: str = "reorient_flash_sac"
    # Match the RSL-RL config for Reorient-v0: policy + perception for actor, plus privileged critic terms.
    # See src/tasks/reorient/rsl_rl_ppo_cfg.py.
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {
            "policy": ["policy", "perception"],
            "critic": ["policy", "perception", "privileged"],
        }
    )
    # Reorient uses EMAJointPositionToLimitsActionCfg; actions land in [-1, 1].
    action_bounds: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Reorient's obs is ~1060-dim (kinematic + 5-frame 64-point PC). The default
        # FlashSAC buffer_max_length (10M) would need ~42 GB on GPU. Cap to 1M which
        # fits comfortably on consumer GPUs while still giving SAC enough capacity.
        self.agent.buffer_max_length = 10_000_000
