# Actual Goal Object Marker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the instant-dexterity goal marker use the USD geometry actually sampled in each environment.

**Architecture:** Resolve the selected reference from every spawned object prim once, build one marker prototype per unique selected USD, and retain a per-environment prototype-index list. Pass that list on every goal marker pose update while leaving single-USD and procedural marker behavior unchanged.

**Tech Stack:** Python, IsaacLab `VisualizationMarkers`, USD prim reference metadata, pytest.

---

### Task 1: Specify per-environment marker behavior

**Files:**
- Modify: `tests/policy/test_instant_dexterity_recording.py`

**Step 1: Add failing behavioral tests**

Extract the relevant top-level functions from `scripts/instant_dexterity.py` with `ast`, execute them against fake stage and marker objects, and assert:

- three environment prims resolve to their three authored USD references,
- duplicate selected paths produce one shared prototype and indices such as `[0, 1, 0]`,
- `_update_command_markers` passes the object marker indices to `visualize`,
- the source no longer selects `usd_path[0]`.

**Step 2: Run the focused tests and confirm red**

```bash
pytest tests/policy/test_instant_dexterity_recording.py -q
```

Expected: the new tests fail because the evaluator currently selects the first configured USD and never supplies marker indices.

### Task 2: Resolve and display actual sampled objects

**Files:**
- Modify: `scripts/instant_dexterity.py`

**Step 1: Add selected-reference resolution**

Add `_selected_object_usd_paths(env)` that reads the single authored reference from every matching spawned object prim when `usd_path` is a list. Validate the prim count and reference count with descriptive runtime errors. Repeat a configured string path for single-USD scenes.

**Step 2: Build unique prototypes and indices**

Add `_marker_paths_and_indices(paths)` and update `_make_target_object_marker` to return `(marker, marker_indices)`. Create one `UsdFileCfg` per unique selected path; procedural fallback uses one cuboid and all-zero indices.

**Step 3: Propagate marker indices**

Thread `target_object_marker_indices` through `_update_command_markers` and both call sites, passing it as `marker_indices=` only for the object marker.

**Step 4: Run focused verification**

```bash
python -m compileall -q scripts/instant_dexterity.py
pytest tests/policy/test_instant_dexterity_recording.py -q
```

Expected: all tests pass.

### Task 3: Validate and commit

**Files:**
- Modify: `scripts/instant_dexterity.py`
- Modify: `tests/policy/test_instant_dexterity_recording.py`
- Add: `docs/plans/2026-08-26-actual-goal-object-marker-design.md`
- Add: `docs/plans/2026-08-26-actual-goal-object-marker.md`

**Step 1: Run a real-stage smoke check**

Spawn four randomized VisDex objects, verify four selected reference paths are recovered, and verify the marker prototype indices select the corresponding goal geometry.

**Step 2: Validate the diff**

```bash
git diff --check
git status --short
```

**Step 3: Commit**

```bash
git add scripts/instant_dexterity.py tests/policy/test_instant_dexterity_recording.py docs/plans/2026-08-26-actual-goal-object-marker-design.md docs/plans/2026-08-26-actual-goal-object-marker.md
git commit -m "fix: visualize each sampled goal object"
```
