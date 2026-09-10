# Clean-Table Six-Stage Retreat Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce the clean-table trajectory to six stages and return the hand base to its configured default pose during retreat.

**Architecture:** Merge the lift and carry object goals into one diagonal segment ending above the box. Keep release and retreat as distinct stages, but make retreat task-specific in the clean-table command by replacing the derived hand-base target with the pose captured from the robot's configured default state.

**Tech Stack:** Python, PyTorch, IsaacLab manager commands, pytest.

---

### Task 1: Six-stage object trajectory

**Files:**
- Modify: `tests/tasks/test_clean_table_trajectory.py`
- Modify: `src/tasks/clean_table/mdps/trajectory.py`

1. Update trajectory tests to require six segments and a direct combined lift/carry waypoint above the box.
2. Run `pytest -q tests/tasks/test_clean_table_trajectory.py` and verify the old seven-stage implementation fails.
3. Change the default segment steps and tolerances to six entries, remove the separate lifted keyframe, and retain the above-box, deposited, release-hold, and final duplicate keyframes.
4. Run `pytest -q tests/tasks/test_clean_table_trajectory.py` and verify it passes.

### Task 2: Reindex command stages and return home

**Files:**
- Modify: `tests/tasks/test_clean_table_command.py`
- Modify: `src/tasks/clean_table/mdps/commands.py`

1. Update command tests for release stage 4 and retreat stage 5, and add a test showing only retreat-stage environments receive the captured default hand-base pose.
2. Run `pytest -q tests/tasks/test_clean_table_command.py` and verify the old indices and missing retreat override fail.
3. Reindex the stage constants, validate six segment entries, capture the initial default hand-base pose, override the derived hand-base goal only during retreat, and require both that goal and physical hand-to-object clearance before success accumulates.
4. Run `pytest -q tests/tasks/test_clean_table_command.py` and verify it passes.

### Task 3: Align configuration and verify

**Files:**
- Modify: `src/tasks/clean_table/env_cfg.py`
- Test: `tests/tasks/test_clean_table_hrl_config.py`

1. Change `recovery_arm_after_stage` from 2 to 1 so the merged lift/carry stage is treated as transport, and retain the transport tolerance for the merged stage.
2. Run the clean-table trajectory, command, and HRL configuration tests.
3. Run Python compilation and `git diff --check`, then inspect the final diff to confirm the user's 35 cm box pose and unrelated edits remain intact.
