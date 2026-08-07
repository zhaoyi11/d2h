from __future__ import annotations

from dataclasses import asdict
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.policy.goal_conditioned import (
    MODEL_TYPE,
    GoalConditionedChunkConfig,
    GoalConditionedChunkPolicy,
    GoalConditionedWindowDataset,
    discover_goal_conditioned_specs,
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


def test_goal_conditioned_dataset_splits_goal_from_context(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    states = np.arange(6 * 26, dtype=np.float32).reshape(6, 26)
    actions = np.arange(5 * 2, dtype=np.float32).reshape(5, 2)
    _write_episode(dataset_dir / "0000000000.npz", states, actions)

    specs, state_keys, _, _ = discover_goal_conditioned_specs(
        dataset_dir,
        past_length=2,
        future_length=2,
        state_keys=("observation.policy",),
    )
    dataset = GoalConditionedWindowDataset(
        specs,
        state_keys,
        past_length=2,
        future_length=2,
        windows=[(0, 1)],
    )
    sample = dataset[0]

    assert dataset.full_state_dim == 26
    assert dataset.state_dim == 19
    assert dataset.goal_dim == 7
    torch.testing.assert_close(sample["goal"], torch.from_numpy(states[2, 3:10]))
    expected_context_states = np.concatenate((states[1:3, :3], states[1:3, 10:26]), axis=-1)
    expected_context = np.concatenate((expected_context_states, actions[1:3]), axis=-1).reshape(-1)
    torch.testing.assert_close(sample["context"], torch.from_numpy(expected_context))
    torch.testing.assert_close(sample["target_actions"], torch.from_numpy(actions[3:5]))


def test_goal_conditioned_policy_and_loader_freeze_model(tmp_path: Path) -> None:
    from src.policy.low_level.policy import load_low_level_goal_conditioned

    config = GoalConditionedChunkConfig(
        state_dim=3,
        action_dim=2,
        goal_dim=7,
        past_length=2,
        future_length=4,
        condition_dim=5,
        hidden_dim=6,
    )
    model = GoalConditionedChunkPolicy(config)
    checkpoint_path = tmp_path / "goal.pt"
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    policy = load_low_level_goal_conditioned(checkpoint_path, device="cpu", expected_action_dim=2)

    assert policy.config == config
    assert policy.chunk_length == 4
    assert policy.action_dim == 2
    assert policy.goal_dim == 7
    assert not any(param.requires_grad for param in policy.model.parameters())
    decoded = policy.predict(torch.zeros(3, config.context_dim), torch.zeros(3, config.goal_dim))
    assert decoded.shape == (3, config.future_length, config.action_dim)


def test_goal_conditioned_train_uses_wandb_and_saves_checkpoints() -> None:
    class FakeWandb:
        def __init__(self):
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

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = root / "dataset"
        output_dir = root / "checkpoint"
        dataset_dir.mkdir()
        (dataset_dir / "metadata.json").write_text(json.dumps({"obs_groups": ["policy"]}))
        for idx in range(4):
            states = np.arange(8 * 10, dtype=np.float32).reshape(8, 10) + idx
            actions = np.arange(7 * 2, dtype=np.float32).reshape(7, 2) + idx
            np.savez_compressed(
                dataset_dir / f"{idx:010d}.npz",
                **{
                    "observation.policy": states,
                    "action": actions,
                },
            )

        fake_wandb = FakeWandb()
        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            train(
                dataset_dir=dataset_dir,
                output_dir=output_dir,
                past_length=2,
                future_length=2,
                condition_dim=8,
                hidden_dim=16,
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
                wandb_run_name="goal-run",
            )

        assert fake_wandb.init_kwargs["project"] == "test-project"
        assert fake_wandb.init_kwargs["mode"] == "offline"
        assert fake_wandb.init_kwargs["name"] == "goal-run"
        assert fake_wandb.saved.count("best.pt") == 1
        assert any(name.startswith("final_") and name.endswith(".pt") for name in fake_wandb.saved)
        assert fake_wandb.logs
        assert any("val/loss" in payload for payload, _ in fake_wandb.logs)
        assert fake_wandb.finished
