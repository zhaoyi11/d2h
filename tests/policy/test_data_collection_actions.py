from __future__ import annotations

import pathlib
from types import SimpleNamespace
from typing import Any

import torch


def _load_function(name: str) -> Any:
    source_path = pathlib.Path(__file__).resolve().parents[2] / "src" / "policy" / "data_collection.py"
    source = source_path.read_text()
    marker = f"def {name}("
    start = source.find(marker)
    assert start >= 0, f"{name} is missing from data_collection.py"
    next_def = source.find("\ndef ", start + len(marker))
    next_section = source.find("\n# --------------------------------------------------------------------------- #", start + len(marker))
    ends = [idx for idx in (next_def, next_section) if idx >= 0]
    end = min(ends) if ends else len(source)
    namespace = {"Any": Any, "torch": torch}
    exec(source[start:end], namespace)
    return namespace[name]


class _PolicyNN:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action

    def act_inference(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.action.clone()


class _Policy:
    def __init__(self, action: torch.Tensor) -> None:
        self.action = action

    def __call__(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.action.clone()


def test_zero_noise_std_per_env_uses_clean_action_for_env_action() -> None:
    select_actions = _load_function("_select_clean_and_env_actions")
    clean_expected = torch.zeros(2, 3)
    action_noise_std = torch.zeros(2)

    clean, env_action = select_actions(
        obs={"policy": torch.zeros(2, 4)},
        policy=_Policy(torch.ones(2, 3)),
        policy_nn=_PolicyNN(clean_expected),
        deterministic=True,
        action_noise_std=action_noise_std,
    )

    torch.testing.assert_close(clean, clean_expected)
    torch.testing.assert_close(env_action, clean_expected)


def test_per_env_action_noise_std_adds_noise_only_to_env_action() -> None:
    select_actions = _load_function("_select_clean_and_env_actions")
    clean_expected = torch.zeros(2, 3)
    action_noise_std = torch.tensor([0.0, 0.3])
    torch.manual_seed(0)

    clean, env_action = select_actions(
        obs={"policy": torch.zeros(2, 4)},
        policy=_Policy(torch.ones(2, 3)),
        policy_nn=_PolicyNN(clean_expected),
        deterministic=True,
        action_noise_std=action_noise_std,
    )

    torch.testing.assert_close(clean, clean_expected)
    torch.testing.assert_close(env_action[0], clean[0])
    assert not torch.equal(env_action[1], clean[1])


def test_stochastic_clean_action_still_uses_policy_callable() -> None:
    select_actions = _load_function("_select_clean_and_env_actions")
    stochastic_expected = torch.full((2, 3), 2.0)

    clean, env_action = select_actions(
        obs={"policy": torch.zeros(2, 4)},
        policy=_Policy(stochastic_expected),
        policy_nn=_PolicyNN(torch.zeros(2, 3)),
        deterministic=False,
        action_noise_std=torch.zeros(2),
    )

    torch.testing.assert_close(clean, stochastic_expected)
    torch.testing.assert_close(env_action, stochastic_expected)


def test_sample_action_noise_std_resamples_only_done_envs() -> None:
    sample_noise_std = _load_function("_sample_action_noise_std")
    current = torch.tensor([0.1, 0.2, 0.3])
    done = torch.tensor([False, True, False])
    generator = torch.Generator(device="cpu").manual_seed(0)

    updated = sample_noise_std(
        num_envs=3,
        device=torch.device("cpu"),
        max_std=0.5,
        current=current,
        done=done,
        generator=generator,
    )

    assert updated.shape == (3,)
    torch.testing.assert_close(updated[0], current[0])
    torch.testing.assert_close(updated[2], current[2])
    assert 0.0 <= float(updated[1]) <= 0.5
    assert not torch.equal(updated[1], current[1])


def test_metadata_payload_records_action_noise_std_range() -> None:
    build_metadata = _load_function("_build_metadata_payload")
    args = SimpleNamespace(
        action_noise_std_max=0.5,
        deterministic=True,
        num_envs=4,
        num_episodes=100,
        seed=0,
        success_key=None,
        task="Reorient_Play-v0",
    )

    metadata = build_metadata(
        args_cli=args,
        groups=["policy"],
        group_dims={"policy": [151]},
        action_dim=16,
        resume_path="/tmp/model.pt",
        git_sha="abc123",
        timestamp="2026-05-03T12:00:00",
    )

    assert metadata["action_noise_std_range"] == [0.0, 0.5]
