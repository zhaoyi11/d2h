from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.policy.bc_base import (
    MODEL_TYPE,
    BehaviorCloningChunkConfig,
    BehaviorCloningChunkPolicy,
    BehaviorCloningWindowDataset,
    _compute_behavior_cloning_normalization_stats,
    discover_behavior_cloning_specs,
    train,
)


def _write_episode(path: Path, states: np.ndarray, actions: np.ndarray) -> None:
    np.savez_compressed(
        path,
        **{
            "observation.policy": states.astype(np.float32),
            "action": actions.astype(np.float32),
        },
    )


def test_behavior_cloning_dataset_uses_current_obs_and_next_actions(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    states = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    actions = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)
    _write_episode(dataset_dir / "0000000000.npz", states, actions)

    specs, state_keys, state_dim, action_dim = discover_behavior_cloning_specs(
        dataset_dir,
        future_length=4,
        state_keys=("observation.policy",),
    )
    dataset = BehaviorCloningWindowDataset(
        specs,
        state_keys,
        future_length=4,
        windows=[(0, 1)],
    )
    sample = dataset[0]

    assert state_dim == 3
    assert action_dim == 2
    assert dataset.state_dim == 3
    assert dataset.action_dim == 2
    torch.testing.assert_close(sample["obs"], torch.from_numpy(states[1]))
    torch.testing.assert_close(sample["target_actions"], torch.from_numpy(actions[1:5]))


def test_behavior_cloning_policy_predicts_action_chunk_and_validates_obs_dim() -> None:
    config = BehaviorCloningChunkConfig(
        state_dim=3,
        action_dim=2,
        future_length=4,
        hidden_dim=8,
        num_layers=3,
    )
    model = BehaviorCloningChunkPolicy(config)

    linear_layers = [layer for layer in model.net if isinstance(layer, torch.nn.Linear)]
    pred = model(torch.zeros(5, config.state_dim))

    assert len(linear_layers) == config.num_layers + 1
    assert pred.shape == (5, config.future_length, config.action_dim)
    with pytest.raises(ValueError, match="Expected obs dim 3"):
        model(torch.zeros(5, config.state_dim + 1))


def test_behavior_cloning_normalization_stats_use_selected_specs(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    train_states = np.array(
        [
            [1.0, 2.0, 7.0],
            [3.0, 4.0, 7.0],
            [5.0, 6.0, 7.0],
            [100.0, 100.0, 100.0],
        ],
        dtype=np.float32,
    )
    train_actions = np.array(
        [
            [1.0, 10.0],
            [3.0, 14.0],
            [5.0, 18.0],
        ],
        dtype=np.float32,
    )
    val_states = np.full((3, 3), 1000.0, dtype=np.float32)
    val_actions = np.full((3, 2), 1000.0, dtype=np.float32)
    _write_episode(dataset_dir / "0000000000.npz", train_states, train_actions)
    _write_episode(dataset_dir / "0000000001.npz", val_states, val_actions)

    specs, state_keys, _, _ = discover_behavior_cloning_specs(
        dataset_dir,
        future_length=2,
        state_keys=("observation.policy",),
    )
    stats = _compute_behavior_cloning_normalization_stats(
        [specs[0]],
        state_keys,
        normalization_eps=1e-6,
    )

    expected_states = train_states[: train_actions.shape[0]]
    np.testing.assert_allclose(stats.obs_mean, expected_states.mean(axis=0))
    np.testing.assert_allclose(stats.obs_std[:2], expected_states.std(axis=0)[:2], rtol=1e-6)
    assert stats.obs_std[2] == 1.0
    np.testing.assert_allclose(stats.action_mean, train_actions.mean(axis=0))
    np.testing.assert_allclose(stats.action_std, train_actions.std(axis=0), rtol=1e-6)


def test_behavior_cloning_policy_normalizes_and_returns_raw_actions() -> None:
    config = BehaviorCloningChunkConfig(
        state_dim=3,
        action_dim=2,
        future_length=4,
        hidden_dim=8,
        num_layers=1,
        normalization_eps=1e-6,
    )
    model = BehaviorCloningChunkPolicy(config)
    model.set_normalization(
        obs_mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        obs_std=np.array([2.0, 4.0, 5.0], dtype=np.float32),
        action_mean=np.array([10.0, -1.0], dtype=np.float32),
        action_std=np.array([3.0, 0.5], dtype=np.float32),
    )
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            torch.nn.init.zeros_(module.bias)

    obs = torch.tensor([[3.0, 6.0, 8.0]], dtype=torch.float32)
    actions = torch.tensor([[[13.0, -0.5], [7.0, -1.5]]], dtype=torch.float32)
    expected_norm_obs = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    expected_raw = torch.tensor([10.0, -1.0]).view(1, 1, 2).expand(2, config.future_length, 2)

    torch.testing.assert_close(model.normalize_obs(obs), expected_norm_obs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(model.denormalize_actions(model.normalize_actions(actions)), actions)
    torch.testing.assert_close(model(torch.zeros(2, config.state_dim)), expected_raw)


def test_behavior_cloning_policy_rejects_invalid_num_layers() -> None:
    config = BehaviorCloningChunkConfig(state_dim=3, action_dim=2, num_layers=0)

    with pytest.raises(ValueError, match="num_layers must be at least 1"):
        BehaviorCloningChunkPolicy(config)


def test_train_uses_wandb_and_saves_behavior_cloning_checkpoints(tmp_path: Path) -> None:
    class FakeWandb:
        def __init__(self) -> None:
            self.logs = []
            self.saved = []
            self.finished = False

        def init(self, **kwargs):
            self.init_kwargs = kwargs
            return self

        def log(self, payload, step=None):
            self.logs.append((payload, step))

        def save(self, path):
            self.saved.append(Path(path).name)

        def finish(self):
            self.finished = True

    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "checkpoint"
    dataset_dir.mkdir()
    (dataset_dir / "metadata.json").write_text(json.dumps({"obs_groups": ["policy"]}))
    for idx in range(4):
        states = np.arange(6 * 3, dtype=np.float32).reshape(6, 3) + idx
        actions = np.arange(6 * 2, dtype=np.float32).reshape(6, 2) + idx
        _write_episode(dataset_dir / f"{idx:010d}.npz", states, actions)

    fake_wandb = FakeWandb()
    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        best_path = train(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            future_length=4,
            hidden_dim=8,
            num_layers=2,
            normalization_eps=1e-6,
            batch_size=2,
            epochs=1,
            lr=1e-3,
            seed=0,
            device="cpu",
            train_windows_per_epoch=2,
            val_windows=2,
            val_ratio=0.5,
            wandb_mode="offline",
            wandb_project="test-project",
            wandb_run_name="bc-run",
        )

    config_payload = json.loads((output_dir / "config.json").read_text())
    assert config_payload["model_type"] == MODEL_TYPE
    assert config_payload["config"]["future_length"] == 4
    assert config_payload["config"]["num_layers"] == 2
    assert config_payload["config"]["normalization_eps"] == 1e-6
    assert config_payload["training"]["num_layers"] == 2
    assert config_payload["training"]["normalization"] == {
        "enabled": True,
        "eps": 1e-6,
        "source": "train_split",
    }
    assert best_path == output_dir / "best.pt"
    assert best_path.exists()
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    for key in ("obs_mean", "obs_std", "action_mean", "action_std"):
        assert key in checkpoint["model_state_dict"]
    assert fake_wandb.init_kwargs["project"] == "test-project"
    assert fake_wandb.init_kwargs["mode"] == "offline"
    assert fake_wandb.init_kwargs["name"] == "bc-run"
    assert fake_wandb.saved.count("best.pt") == 1
    assert any(name.startswith("final_") and name.endswith(".pt") for name in fake_wandb.saved)
    assert any("val/loss" in payload for payload, _ in fake_wandb.logs)
    assert any("val/raw_mse" in payload for payload, _ in fake_wandb.logs)
    assert fake_wandb.finished
