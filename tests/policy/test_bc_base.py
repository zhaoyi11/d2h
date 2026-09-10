from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from dataets.bc import (
    MODEL_TYPE,
    BehaviorCloningConfig,
    BehaviorCloningPolicy,
    _episode_windows,
    _normalization,
    load_behavior_cloning_data,
    train,
)


def _write_episode(path: Path, states: np.ndarray, actions: np.ndarray) -> None:
    np.savez_compressed(
        path,
        **{
            "observation.bc": states.astype(np.float32),
            "action": actions.astype(np.float32),
        },
    )


def test_behavior_cloning_windows_use_current_obs_and_next_actions(tmp_path: Path) -> None:
    states = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    actions = np.arange(6 * 2, dtype=np.float32).reshape(6, 2)

    obs, targets = _episode_windows([(tmp_path / "episode.npz", states, actions)], future_length=4)

    np.testing.assert_array_equal(obs[1], states[1])
    np.testing.assert_array_equal(targets[1], actions[1:5])


def test_behavior_cloning_policy_predicts_normalized_action_chunk() -> None:
    config = BehaviorCloningConfig(
        state_dim=3,
        action_dim=2,
        future_length=4,
        hidden_dim=8,
        num_layers=2,
    )
    model = BehaviorCloningPolicy(config)
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
    prediction = model(torch.zeros(5, config.state_dim))

    assert prediction.shape == (5, config.future_length, config.action_dim)
    torch.testing.assert_close(model.normalize_obs(obs), torch.ones_like(obs))
    torch.testing.assert_close(model.denormalize_actions(model.normalize_actions(actions)), actions)
    torch.testing.assert_close(
        prediction,
        torch.tensor([10.0, -1.0]).view(1, 1, 2).expand_as(prediction),
    )
    with pytest.raises(ValueError, match="Expected obs dim 3"):
        model(torch.zeros(5, config.state_dim + 1))


def test_behavior_cloning_data_splits_episodes_and_normalizes_train_windows(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for index in range(4):
        states = np.arange(6 * 3, dtype=np.float32).reshape(6, 3) + index
        states[:, 2] = 7.0
        actions = np.arange(6 * 2, dtype=np.float32).reshape(6, 2) + index
        _write_episode(dataset_dir / f"{index:010d}.npz", states, actions)

    data = load_behavior_cloning_data(dataset_dir, future_length=4, val_ratio=0.25, seed=0)
    obs_mean, obs_std, action_mean, action_std = _normalization(data, 1e-6)

    assert len(data.train_episodes) == 3
    assert len(data.val_episodes) == 1
    assert set(data.train_episodes).isdisjoint(data.val_episodes)
    np.testing.assert_allclose(obs_mean, data.train_obs.mean(axis=0), rtol=1e-6)
    assert obs_std[2] == 1.0
    np.testing.assert_allclose(action_mean, data.train_actions.reshape(-1, 2).mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(action_std, data.train_actions.reshape(-1, 2).std(axis=0), rtol=1e-6)


def test_behavior_cloning_data_rejects_non_finite_values(tmp_path: Path) -> None:
    states = np.zeros((4, 3), dtype=np.float32)
    actions = np.zeros((4, 2), dtype=np.float32)
    _write_episode(tmp_path / "0000000000.npz", states, actions)
    states[0, 0] = np.nan
    _write_episode(tmp_path / "0000000001.npz", states, actions)

    with pytest.raises(ValueError, match="non-finite"):
        load_behavior_cloning_data(tmp_path)


def test_behavior_cloning_policy_rejects_invalid_num_layers() -> None:
    config = BehaviorCloningConfig(state_dim=3, action_dim=2, num_layers=0)

    with pytest.raises(ValueError, match="num_layers must be at least 1"):
        BehaviorCloningPolicy(config)


def test_train_logs_metrics_and_saves_loadable_checkpoint(tmp_path: Path) -> None:
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
    for index in range(4):
        states = np.arange(6 * 3, dtype=np.float32).reshape(6, 3) + index
        actions = np.arange(6 * 2, dtype=np.float32).reshape(6, 2) + index
        _write_episode(dataset_dir / f"{index:010d}.npz", states, actions)

    fake_wandb = FakeWandb()
    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        best_path = train(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            future_length=4,
            hidden_dim=8,
            num_layers=2,
            batch_size=2,
            epochs=1,
            lr=1e-3,
            seed=0,
            device="cpu",
            val_ratio=0.5,
            wandb_mode="offline",
            save_every_steps=2,
        )

    config_payload = json.loads((output_dir / "config.json").read_text())
    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)

    assert config_payload["model_type"] == MODEL_TYPE
    assert config_payload["config"]["future_length"] == 4
    assert config_payload["training"]["save_every_steps"] == 2
    assert config_payload["training"]["normalization"] == {
        "enabled": True,
        "eps": 1e-6,
        "source": "train_split",
    }
    assert best_path == output_dir / "best.pt"
    periodic_path = output_dir / "step_000000002.pt"
    assert periodic_path.exists()
    periodic_checkpoint = torch.load(periodic_path, map_location="cpu", weights_only=True)
    assert periodic_checkpoint["global_step"] == 2
    for key in ("obs_mean", "obs_std", "action_mean", "action_std"):
        assert key in checkpoint["model_state_dict"]
    assert fake_wandb.init_kwargs["project"] == "d2h-bc"
    assert fake_wandb.init_kwargs["mode"] == "offline"
    assert fake_wandb.saved == ["step_000000002.pt", "best.pt"]
    assert any("val/loss" in payload for payload, _ in fake_wandb.logs)
    assert any("val/raw_mse" in payload for payload, _ in fake_wandb.logs)
    assert fake_wandb.finished
    with pytest.raises(FileExistsError, match="already contains"):
        train(dataset_dir=dataset_dir, output_dir=output_dir, epochs=1, device="cpu")
