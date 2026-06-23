# Reorient Gravity Curriculum Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a task-local gravity curriculum for `src/tasks/reorient` that mirrors the DexSuite pattern without importing the reference implementation.

**Architecture:** Add a task-local `src/tasks/reorient/curriculum.py` that exposes `CurriculumCfg` with an `adr` scheduler and a `gravity_adr` curriculum term that rewrites `events.variable_gravity.params.gravity_distribution_params`. Re-export `CurriculumCfg` through `src/tasks/reorient/mdps.py`, enable it only in the main reorient environment config, and keep grasp-generation behavior unchanged.

**Tech Stack:** Python, Isaac Lab config classes, `pytest`, AST/text-based tests

---

### Task 1: Add the failing test

**Files:**
- Create: `tests/test_reorient_curriculum_cfg.py`
- Test: `tests/test_reorient_curriculum_cfg.py`

**Step 1: Write the failing test**

Add tests that assert:
- `src/tasks/reorient/_mdps/curriculum.py` defines `CurriculumCfg`
- `CurriculumCfg.gravity_adr` targets `events.variable_gravity.params.gravity_distribution_params`
- `src/tasks/reorient/env_cfg.py` enables `task_mdp.CurriculumCfg()` and updates `rot_tol` from `orientation_success_threshold`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_reorient_curriculum_cfg.py -q`
Expected: FAIL because `CurriculumCfg` and/or env wiring do not exist yet.

### Task 2: Implement the curriculum

**Files:**
- Create: `src/tasks/reorient/curriculum.py`
- Modify: `src/tasks/reorient/mdps.py`
- Modify: `src/tasks/reorient/env_cfg.py`

**Step 1: Write minimal implementation**

Add the missing `CurrTerm` and `configclass` imports, then define `CurriculumCfg` with:
- `adr = CurrTerm(func=DifficultyScheduler, params={...})`
- `gravity_adr = CurrTerm(func=mdp.modify_term_cfg, params={...})`

Wire `curriculum: task_mdp.CurriculumCfg | None = task_mdp.CurriculumCfg()` into `InHandObjectEnvCfg`, then set `self.curriculum.adr.params["rot_tol"]` from `self.commands.object_pose.orientation_success_threshold` in `__post_init__`.

**Step 2: Run test to verify it passes**

Run: `pytest tests/test_reorient_curriculum_cfg.py -q`
Expected: PASS.

### Task 3: Regression check

**Files:**
- Test: `tests/test_reorient_mdps.py`

**Step 1: Run targeted tests**

Run: `pytest tests/test_reorient_curriculum_cfg.py tests/test_reorient_mdps.py -q`
Expected: PASS.
