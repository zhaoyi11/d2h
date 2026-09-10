# Unscrew OmniReset Train/Play Resets Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train `Unscrew_OmniReset-v0` from saved trajectory states using the existing reverse curriculum while evaluating `Unscrew_OmniReset_Play-v0` from the fixed hardest reset.

**Architecture:** The environment ID selects the reset contract. The training config owns the reset dataset, `ResetSceneFromInstantDexterity`, and curriculum. A dedicated `_PLAY` config disables the dataset reset and curriculum and enables IsaacLab's fixed default-scene reset. The generic RSL-RL play launcher remains task-agnostic.

**Tech Stack:** Python 3.11, IsaacLab manager-based events, RSL-RL, pytest.

---

### Task 1: Specify the train/play reset contract

**Files:**
- Modify: `tests/tasks/unscrew/test_omnireset_trajectory_observations.py`

1. Update the task-config test to require the repository-local `dataets/unscrew_bc` dataset, `ResetSceneFromInstantDexterity`, and the existing reset curriculum for training.
2. Add a test that requires `Unscrew_OmniReset_Play-v0` to use a dedicated `_PLAY` config with the hardest reset, while `scripts/rsl_rl/play.py` remains task-agnostic.
3. Run the focused tests and verify they fail before implementation.

### Task 2: Restore trajectory curriculum resets for training

**Files:**
- Modify: `src/tasks/unscrew_omnireset/env_cfg.py`

1. Import the task-local reset event and define a repository-relative reset dataset path.
2. Replace `reset_scene_to_default` with `ResetSceneFromInstantDexterity` in the training event config.
3. Add `reset_dataset_dir` to the environment config without changing observations, rewards, actions, or physics.
4. Run the focused task test and verify the training assertions pass.

### Task 3: Register the hardest-reset Play environment

**Files:**
- Modify: `src/tasks/unscrew_omnireset/env_cfg.py`
- Modify: `src/tasks/__init__.py`
- Restore: `scripts/rsl_rl/play.py`

1. Add `DexsuiteFrankaLeapUnscrewOmniResetEnvCfg_PLAY`, disabling the dataset reset and curriculum and enabling `reset_scene_to_default`.
2. Register it as `Unscrew_OmniReset_Play-v0` with the same RSL-RL agent config.
3. Remove task-specific reset logic from `play.py`.
4. Run the focused play-contract test and verify it passes.

### Task 4: Verify

**Files:**
- Verify: `tests/tasks/unscrew/test_omnireset_trajectory_observations.py`
- Verify: `src/tasks/unscrew_omnireset/env_cfg.py` and `src/tasks/__init__.py`

1. Run both focused test files in `env_isaaclab`.
2. Run `python -m py_compile` on the changed Python files.
3. Run `git diff --check` and inspect the diff to confirm the existing reward comments remain untouched.
