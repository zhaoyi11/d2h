# Screw-In Calibration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Physically screw the full-scale leg into the socket, report its settled pose, and save the measured motion reversed as a reusable unscrew trajectory.

**Architecture:** Add one standalone IsaacLab script that reuses the realistic verifier's bounded wrench controller and existing unscrew trajectory builder. Spawn the leg three thread pitches above the installed pose (the minimum integer-pitch rise that clears the authored socket bounds), track the helix in reverse, settle without an applied wrench, and write one compressed NumPy artifact containing world-frame and socket-relative measurements plus explicit metadata.

**Tech Stack:** Python 3.11, IsaacLab/Isaac Sim PhysX, PyTorch, NumPy, pytest.

---

### Task 1: Add pure recording helpers

**Files:**
- Create: `src/tasks/unscrew/trajectory_recording.py`
- Create: `scripts/record_screw_in_calibration.py`
- Create: `tests/tasks/unscrew/test_screw_in_calibration.py`

1. Add pure NumPy helpers that reverse the measured screw-in samples and prepend the settled pose.
2. Add helpers that validate and save the documented `.npz` schema without importing IsaacLab.
3. Test sample ordering, timestamps, shapes, and metadata without launching Isaac Sim.

### Task 2: Add the physical screw-in rollout

**Files:**
- Modify: `scripts/record_screw_in_calibration.py`

1. Preserve the `AppLauncher`-before-IsaacLab import order.
2. Build a one-environment scene from `src.tasks.unscrew.env_cfg.SceneCfg` with the robot removed.
3. Spawn the object three 15 mm pitches above `INSTALLED_OBJECT_POS`, about 7.5 mm clear of the socket.
4. Hold the initial pose, then track the verified ideal unscrew helix in reverse with a bounded PD wrench plus an insertion-only downward preload.
5. Release the wrench, settle, and measure pose drift and terminal velocities.
6. Reject non-finite, insufficient-rotation, or unstable recordings.

### Task 3: Record and verify the artifact

**Files:**
- Create at runtime: `outputs/unscrew/screw_in_calibration.npz`

1. Run the focused unit tests and compile the script.
2. Run the script headlessly with GPU access.
3. Load the saved file independently and verify every required array.
4. Run `git diff --check` and report the measured stable world/socket-relative pose and artifact path.
