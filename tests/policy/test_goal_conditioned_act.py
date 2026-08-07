from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from src.policy.goal_conditioned_act import (
    MODEL_TYPE,
    GoalConditionedACTConfig,
    GoalConditionedACTPolicy,
    train_goal_conditioned_act,
)


def _write_metadata(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "metadata.json").write_text(json.dumps({"obs_groups": ["policy"]}))


def _write_episode(path: Path, length: int, state_dim: int = 8, action_dim: int = 2) -> None:
    states = np.arange((length + 1) * state_dim, dtype=np.float32).reshape(length + 1, state_dim)
    actions = np.arange(length * action_dim, dtype=np.float32).reshape(length, action_dim)
    np.savez_compressed(
        path,
        **{
            "observation.policy": states,
            "action": actions,
        },
    )


def _config() -> GoalConditionedACTConfig:
    return GoalConditionedACTConfig(
        state_dim=8,
        action_dim=2,
        past_length=4,
        future_length=3,
        condition_dim=8,
        hidden_dim=16,
        transformer_layers=1,
        transformer_heads=4,
        transformer_feedforward_dim=32,
        transformer_dropout=0.0,
    )


def test_model_forward_shape_and_loss_are_finite() -> None:
    config = _config()
    model = GoalConditionedACTPolicy(config)
    context = torch.randn(3, config.context_dim)
    target_actions = torch.randn(3, config.future_length, config.action_dim)

    pred = model(context)
    loss = F.mse_loss(pred, target_actions)

    assert pred.shape == (3, config.future_length, config.action_dim)
    assert torch.isfinite(loss)


def test_context_path_uses_transformer_encoder_and_condition_token() -> None:
    config = _config()
    model = GoalConditionedACTPolicy(config)
    context = torch.randn(3, config.context_dim)

    condition, context_memory = model.context_encoder(context)
    condition_token = model.decoder.condition_memory_proj(condition).unsqueeze(1)
    memory = torch.cat([condition_token, context_memory], dim=1)

    assert isinstance(model.context_encoder.transformer, torch.nn.TransformerEncoder)
    assert model.decoder.query_embed.num_embeddings == config.target_length
    assert not hasattr(model, "goal_encoder")
    assert memory.shape == (3, config.past_length + 2, config.hidden_dim)


def test_train_goal_conditioned_act_writes_checkpoint_log_and_config() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = root / "dataset"
        output_dir = root / "checkpoint"
        _write_metadata(dataset_dir)
        _write_episode(dataset_dir / "0000000000.npz", length=10)
        _write_episode(dataset_dir / "0000000001.npz", length=11)

        train_goal_conditioned_act(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            past_length=4,
            future_length=3,
            condition_dim=8,
            hidden_dim=16,
            transformer_layers=1,
            transformer_heads=4,
            transformer_feedforward_dim=32,
            transformer_dropout=0.0,
            batch_size=2,
            epochs=1,
            lr=1e-3,
            seed=0,
            device="cpu",
            train_windows_per_epoch=2,
            val_windows=1,
            val_ratio=0.5,
            wandb_mode="disabled",
        )

        assert (output_dir / "config.json").exists()
        assert (output_dir / "model.pt").exists()
        assert (output_dir / "train_log.jsonl").exists()
        assert (output_dir / "last.pt").exists()
        assert (output_dir / "best.pt").exists()

        with (output_dir / "config.json").open() as f:
            config_payload = json.load(f)
        assert config_payload["model_type"] == MODEL_TYPE
        assert config_payload["training"]["transformer_layers"] == 1
        assert config_payload["training"]["transformer_heads"] == 4
        assert config_payload["training"]["transformer_feedforward_dim"] == 32
        assert config_payload["training"]["transformer_dropout"] == 0.0
        assert "goal_slice" not in config_payload["training"]

        with (output_dir / "train_log.jsonl").open() as f:
            row = json.loads(f.readline())
        assert "global_step" in row
        assert "loss" in row
        assert "val_loss" in row
