# Cupcake Yaw Reward and Training Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `Cupcake_on_Plate-v0` reward tracking its active yaw command and provide a runnable
RSL-RL training launcher.

**Architecture:** The cupcake trajectory command publishes whether its yaw stage is active. A
task-local reward reads that flag and computes the live cake quaternion error against the command's
current object goal, applying a gated tanh kernel; the environment reward config supplies the command
name and scale. A shell launcher invokes the existing RSL-RL entry point and task registration without
duplicating training logic.

**Tech Stack:** Python, PyTorch, IsaacLab manager terms/configs, RSL-RL, Bash, pytest.

---

### Task 1: Specify the command-conditioned reward

**Files:**
- Modify: `tests/tasks/test_cupcake_z_axis.py`

1. Extend the command lifecycle test to require a `yaw_target_active` metric derived from the current trajectory stage.
2. Add a pure unit test requiring `trajectory_yaw_tracking` to ignore stale command metrics, return
   zero during reach, and use `1 - tanh(error / std)` during yaw tracking.
3. Add AST checks requiring `RewardsCfg.yaw_tracking` to use command `object_pose`, `std=0.5`, and
   weight `4.0`.
4. Run `pytest tests/tasks/test_cupcake_z_axis.py -q` and confirm the new expectations fail.

### Task 2: Implement the reward

**Files:**
- Modify: `src/tasks/cupcake_on_plate/mdps/commands.py`
- Modify: `src/tasks/cupcake_on_plate/mdps/task_mdps.py`
- Modify: `src/tasks/cupcake_on_plate/mdps/trajectory.py`
- Modify: `src/tasks/cupcake_on_plate/env_cfg.py`

1. Keep stage tolerances primitive for Hydra serialization and convert them on a runtime command-config
   copy.
2. Initialize and update `metrics["yaw_target_active"]` in the command term.
3. Add `trajectory_yaw_tracking(env, command_name="object_pose", std=0.5)` using the live goal and
   object quaternions.
4. Configure the reward as `RewTerm(func=mdp.trajectory_yaw_tracking, weight=4.0,
   params={"command_name": "object_pose", "std": 0.5})`.
5. Run the focused test and confirm it passes.

### Task 3: Add the training launcher

**Files:**
- Create: `scripts/train_cupcake_z_axis.sh`
- Modify: `tests/tasks/test_cupcake_z_axis.py`

1. Test that the launcher selects `Cupcake_on_Plate-v0`, uses `scripts/rsl_rl/train.py`, activates
   `env_isaaclab`, and defaults to 4096 environments and 15,000 iterations.
2. Add a strict Bash launcher that changes to the repository root and forwards additional arguments.
3. Run `bash -n scripts/train_cupcake_z_axis.sh` and the focused test suite.

### Task 4: Verify and review

**Files:**
- Verify all files above.

1. Run `python -m compileall -q src/tasks/cupcake_on_plate`.
2. Run `pytest tests/tasks -q`; accept only the five user-approved baseline failures.
3. Run an independent code review focused on reward timing, command lifecycle, and launcher correctness.
4. Commit the verified changes on `feat/cupcake-z-axis`.
