# Unscrew Reset Collision-Clearance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep the full-scale leg upright and visibly seated without a large PhysX contact impulse, while preserving the physical distinction between helical unscrewing and straight pulling.

**Architecture:** Keep the authored installed transform fixed and apply a small symmetric negative PhysX rest offset to the leg and socket colliders. Tune only that offset against the reset-stability validator, then confirm that the realistic verifier still allows the helix and blocks straight pulling.

**Tech Stack:** Python 3.11, IsaacLab/Isaac Sim PhysX, PyTorch, pytest.

---

### Task 1: Restore the seated full-scale pose

**Files:**
- Modify: `src/tasks/unscrew/env_cfg.py`
- Test: `tests/tasks/test_unscrew_registration_and_config.py`

1. Restore the original installed position and identity orientation.
2. Keep the leg and socket scales equal to `ASSET_SCALE`.
3. Add the same explicit contact and rest offsets to both mating assets.

### Task 2: Find the minimum stable contact allowance

**Files:**
- Modify: `src/tasks/unscrew/env_cfg.py`
- Test: `tests/tasks/test_unscrew_registration_and_config.py`
- Modify if needed: `scripts/validate_unscrew_reset.py`

1. Update exact pose and collision-offset expectations in the focused config test.
2. Run the reset validator with eight environments for two simulated seconds.
3. Increase only the magnitude of the symmetric negative rest offset until every environment remains below 0.01 m/s linear and 0.1 rad/s angular speed.

### Task 3: Verify both stability and unscrew feasibility

**Files:**
- Verify: `scripts/validate_unscrew_reset.py`
- Verify: `scripts/verify_unscrew_trajectory_physics.py`

1. Run the reset-stability diagnostic under HRL gravity.
2. Run the realistic helix-versus-straight-pull verifier without changing forces, torque, or gains.
3. Run focused pytest, `py_compile`, and `git diff --check`.
4. Report measured reset speeds, physical classification, and any remaining limitation.
