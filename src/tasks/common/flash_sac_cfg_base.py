"""Base dataclass for per-task FlashSAC training configs.

RSL-RL configs live under ``src/tasks/common/rsl_rl_ppo_cfg_base.py``; this is the
parallel file for the FlashSAC runner. Each task registers a ``flash_sac_cfg_entry_point``
that points to a subclass of ``FlashSacCfgBase``.

Design note: FlashSAC's own config uses plain ``@dataclass`` (``flash_rl.config.Config``)
with mutable defaults gated by ``field(default_factory=...)``. To keep subclassing
ergonomic — D2H tasks typically tweak only two or three fields — this base uses
``@dataclass`` as well. Task configs override fields via ``__post_init__`` or by
assignment in a subclass ``__post_init__`` after calling ``super().__post_init__()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flash_rl.config import AgentConfig, RunnerConfig


@dataclass
class FlashSacCfgBase:
    """Top-level FlashSAC training config for a D2H task.

    Analogue of ``flash_rl.config.Config`` plus the extra fields needed to bridge
    an Isaac Lab multi-group observation env into FlashSAC's vector-env interface.
    """

    # Identity / logging
    experiment_name: str = "flash_sac"
    run_name: str = ""
    project_name: str = "FlashSAC-D2H"
    entity_name: str | None = None
    group_name: str = "dex"

    # Training-loop scheduling
    seed: int = 42
    num_env_steps: int = 50_000_000
    num_eval_episodes: int = 0
    num_record_episodes: int = 0

    # Env / adapter knobs
    obs_groups: dict[str, list[str]] = field(
        default_factory=lambda: {"policy": ["policy"], "critic": ["policy"]}
    )
    action_bounds: float = 1.0
    device: str = "cuda:0"

    # Nested FlashSAC configs (mutable defaults → default_factory)
    agent: AgentConfig = field(default_factory=AgentConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)

    def __post_init__(self) -> None:
        # Propagate seed into nested configs, matching flash_rl.config.Config.__post_init__.
        self.agent.seed = self.seed
        # Keep runner.num_env_steps aligned with the top-level field so FlashSAC's
        # built-in scheduling math (evaluation / logging intervals) stays consistent.
        self.runner.num_env_steps = self.num_env_steps

    def to_flash_rl_config(self, num_train_envs: int, save_path: str) -> Any:
        """Assemble a ``flash_rl.config.Config`` instance ready to hand to FlashSAC.

        The training loop in ``FlashSAC_dex/train.py`` reads ``cfg.env_cfg.num_train_envs``
        to compute ``env_step`` and uses ``cfg.runner_cfg.*_per_interaction_step`` for
        eval/log cadence. We synthesize ``env_cfg`` here (FlashSAC's own env creator is
        bypassed — D2H builds the Isaac Lab env directly).
        """
        from flash_rl.config import Config, EnvConfig

        env_cfg = EnvConfig(
            env_type="isaaclab",
            env_name=self.experiment_name,
            seed=self.seed,
            num_train_envs=num_train_envs,
        )
        cfg = Config(
            project_name=self.project_name,
            entity_name=self.entity_name or "aria-ic",
            group_name=self.group_name,
            exp_name=self.experiment_name,
            seed=self.seed,
            mode="train",
            num_env_steps=self.num_env_steps,
            num_eval_episodes=self.num_eval_episodes,
            num_record_episodes=self.num_record_episodes,
            env_cfg=env_cfg,
            agent_cfg=self.agent,
            runner_cfg=self.runner,
        )
        # Override the default ``models/…`` save_path with D2H's logs/flash_sac/… layout.
        cfg.save_path = save_path
        return cfg
