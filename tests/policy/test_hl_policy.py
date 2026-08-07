from __future__ import annotations

from pathlib import Path

import torch


def _make_rsl_rl_actor(obs_dim: int = 3, action_dim: int = 2) -> torch.nn.Sequential:
    return torch.nn.Sequential(torch.nn.Linear(obs_dim, 4), torch.nn.ELU(), torch.nn.Linear(4, action_dim))


def _write_rsl_rl_checkpoint(path: Path, obs_dim: int = 3, action_dim: int = 2) -> None:
    actor = _make_rsl_rl_actor(obs_dim=obs_dim, action_dim=action_dim)
    state_dict = {f"actor.{key}": value for key, value in actor.state_dict().items()}
    state_dict["actor_obs_normalizer._mean"] = torch.zeros(1, obs_dim)
    state_dict["actor_obs_normalizer._std"] = torch.ones(1, obs_dim)
    torch.save({"model_state_dict": state_dict}, path)


class _ExportedRslRlPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int = 3, action_dim: int = 2) -> None:
        super().__init__()
        self.actor = _make_rsl_rl_actor(obs_dim=obs_dim, action_dim=action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


def _write_torchscript_rsl_rl_policy(path: Path, obs_dim: int = 3, action_dim: int = 2) -> None:
    model = _ExportedRslRlPolicy(obs_dim=obs_dim, action_dim=action_dim)
    exported = torch.jit.trace(model, torch.zeros(1, obs_dim))
    exported.save(str(path))


def test_low_level_rsl_rl_loader_supports_training_checkpoint(tmp_path: Path) -> None:
    from src.policy.low_level.policy import load_low_level_rsl_rl_policy

    checkpoint_path = tmp_path / "model.pt"
    _write_rsl_rl_checkpoint(checkpoint_path, obs_dim=3, action_dim=2)

    policy = load_low_level_rsl_rl_policy(
        checkpoint_path,
        device="cpu",
        expected_obs_dim=3,
        expected_action_dim=2,
    )

    assert policy.obs_dim == 3
    assert policy.action_dim == 2
    assert policy.obs_mean is not None
    assert policy.obs_std is not None
    assert not any(param.requires_grad for param in policy.model.parameters())
    assert not policy.model.training
    assert policy.act(torch.zeros(5, 3)).shape == (5, 2)


def test_low_level_rsl_rl_loader_supports_torchscript_policy(tmp_path: Path) -> None:
    from src.policy.low_level.policy import load_low_level_rsl_rl_policy

    checkpoint_path = tmp_path / "policy.pt"
    _write_torchscript_rsl_rl_policy(checkpoint_path, obs_dim=3, action_dim=2)

    policy = load_low_level_rsl_rl_policy(
        checkpoint_path,
        device="cpu",
        expected_obs_dim=3,
        expected_action_dim=2,
    )

    assert policy.obs_dim == 3
    assert policy.action_dim == 2
    assert policy.obs_mean is None
    assert policy.obs_std is None
    assert not any(param.requires_grad for param in policy.model.parameters())
    assert not policy.model.training
    assert policy.act(torch.zeros(5, 3)).shape == (5, 2)


def test_low_level_rsl_rl_loader_validates_torchscript_dims(tmp_path: Path) -> None:
    from src.policy.low_level.policy import load_low_level_rsl_rl_policy

    checkpoint_path = tmp_path / "policy.pt"
    _write_torchscript_rsl_rl_policy(checkpoint_path, obs_dim=3, action_dim=2)

    try:
        load_low_level_rsl_rl_policy(checkpoint_path, device="cpu", expected_obs_dim=4)
    except ValueError as exc:
        assert "Expected low-level observation dim 4, checkpoint has 3" in str(exc)
    else:
        raise AssertionError("Expected wrong observation dim to fail")

    try:
        load_low_level_rsl_rl_policy(checkpoint_path, device="cpu", expected_action_dim=3)
    except ValueError as exc:
        assert "Expected low-level action dim 3, checkpoint has 2" in str(exc)
    else:
        raise AssertionError("Expected wrong action dim to fail")


def test_low_level_rsl_rl_loader_rejects_unsupported_checkpoint(tmp_path: Path) -> None:
    from src.policy.low_level.policy import load_low_level_rsl_rl_policy

    checkpoint_path = tmp_path / "bad.pt"
    torch.save({"not_model_state_dict": {}}, checkpoint_path)

    try:
        load_low_level_rsl_rl_policy(checkpoint_path, device="cpu")
    except ValueError as exc:
        assert "Unsupported RSL-RL checkpoint format" in str(exc)
    else:
        raise AssertionError("Expected unsupported checkpoint to fail")
