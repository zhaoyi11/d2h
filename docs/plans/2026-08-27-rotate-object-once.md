# Standalone Rotate Object Once Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an independent `rotate_object_once` IsaacLab task that resets successfully after one yaw target while preserving the existing `rotate_object` task.

**Architecture:** Copy the current rotate-object package and rename every task-local import and public symbol so the new package has no dependency on `src.tasks.rotate_object`. Keep the reach-to-yaw command flow, latch final yaw achievement, and expose it through a named IsaacLab success termination that resets on the following control step.

**Tech Stack:** Python, PyTorch, IsaacLab manager-based environments, Gymnasium registration, pytest.

---

## Implementation

1. Duplicate `src/tasks/rotate_object` as `src/tasks/rotate_object_once`, excluding caches, and rename task-specific trajectory, command, environment, and PPO symbols.
2. Rewrite all task-local imports to `src.tasks.rotate_object_once`; continue sharing project-wide policies, common MDP utilities, robot configuration, and assets.
3. Change the copied command term so reach achievement advances normally, while final yaw achievement remains latched and never resamples another target.
4. Add `yaw_target_achieved()` and configure a named `success` termination without adding a terminal reward bonus.
5. Register `Rotate_Object_Once-v0` and `Rotate_Object_Once_HRL-v0`, preserving existing registrations and user changes.
6. Add focused tests for package independence, final-success latching, termination configuration, registrations, and PPO naming.

## Public Interfaces

- Gym IDs: `Rotate_Object_Once-v0` and `Rotate_Object_Once_HRL-v0`.
- Environment configs: `DexsuiteFrankaLeapRotateObjectOnceEnvCfg` and `DexsuiteFrankaLeapRotateObjectOnceHrlEnvCfg`.
- Trainer config: `RotateObjectOnceRslRlPpoCfg` with `experiment_name = "rotate_object_once"`.
- Task-local termination helper: `yaw_target_achieved(env, command_name="object_pose")`.

## Verification

- Run `pytest -q tests/tasks/test_rotate_object_once.py`.
- Run the original continuous-resampling regression test to prove `rotate_object` remains unchanged.
- Confirm the new package contains no imports from `src.tasks.rotate_object`.
- Confirm `git diff -- src/tasks/rotate_object` is empty.
- Do not fix the two pre-existing anchor-offset and knob-scale failures in the complete original test file.

## Assumptions

- Success resets through normal IsaacLab ordering on the following 60 Hz control step.
- The play configuration class is copied but no separate Play Gym ID is added.
- `rotate_object_omnireset` is not duplicated or modified.
- No dedicated launcher script or terminal reward bonus is added.
