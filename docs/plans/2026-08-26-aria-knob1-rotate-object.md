# ARIA Knob1 Rotate-Object Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the rotate-object task's selected VisDex object with a repository-local ARIA knob1
handle aligned to the table baseline.

**Architecture:** Copy the original ARIA USD composition and add a stronger handle-only USDA layer
that removes articulation topology while preserving geometry, materials, and collisions. Keep the
existing `RigidObject` and prestartup Z-axis joint pipeline, changing only the asset path, scale,
baseline position, and contact prim.

**Tech Stack:** Python, pytest, USD/pxr, IsaacLab scene configs and spawners.

---

### Task 1: Specify the knob asset contract

**Files:**
- Modify: `tests/tasks/test_rotate_object_z_axis.py`

1. Replace the fixed-VisDex test with assertions for the ARIA handle asset path, scale `1.0`, root
   position `(0.55, 0.20, 0.255)`, and contact path `/Object/handle`.
2. Add a USD topology test asserting that the derived asset opens successfully, contains exactly
   one rigid body named `handle`, contains no physics joints or articulation-root API, and has its
   lower Z bound at zero.
3. Run the two tests and confirm they fail before the asset/config changes.

### Task 2: Copy and adapt knob1

**Files:**
- Add: `src/assets/aria/knob1/knob1.usd`
- Add: `src/assets/aria/knob1/knob1/`
- Add: `src/assets/aria/knob1/knob1_handle.usda`

1. Copy `knob1.usd` and the complete `knob1/` dependency directory from the local ARIA checkout.
2. Add `knob1_handle.usda`, referencing `/knob`, deleting the articulation APIs, and deactivating
   `base` and `Joints`.
3. Run the USD topology test and confirm it passes.

### Task 3: Use knob1 in the task

**Files:**
- Modify: `src/tasks/rotate_object/env_cfg.py`
- Modify: `src/tasks/rotate_object/mdps/contact_filters.py`
- Modify: `tests/tasks/test_rotate_object_z_axis.py`

1. Remove the VisDex path helper/index and point `UsdFileCfg` at the derived knob asset.
2. Set scale to `(1.0, 1.0, 1.0)` and place the authored baseline at table-top Z `0.255`.
3. Change the object contact-filter path to `/Object/handle`.
4. Run the full rotate-object test file and evaluator marker regression tests.

### Task 4: Validate the real scene

1. Spawn four cloned environments with the current `SceneCfg`.
2. Verify each object resolves to one rigid body, the handle baseline is on the table, and the
   external Z-axis joint attaches to `/Object/handle`.
3. Verify the goal marker uses the copied knob asset.
4. Run compilation and scoped diff checks without changing the user's existing staging state.
