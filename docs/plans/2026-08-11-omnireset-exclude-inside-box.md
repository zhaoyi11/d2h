# Clean Table OmniReset Outside-Box Reset Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Exclude reset states whose object is already inside the receptive box.

**Architecture:** A simulator-free task-local geometry helper will classify pool poses in the box's local frame using IsaacLab's wxyz inverse-quaternion convention. The reset event will pass the existing task bounds, classify once, then uniformly sample only retained outside-box indices while restoring original full-scene states.

**Tech Stack:** Python, PyTorch, IsaacLab manager terms, pytest.

---

### Task 1: Filter the reset sampling pool

**Files:**
- Create: `src/tasks/clean_table_omnireset/mdps/geometry.py`
- Modify: `src/tasks/clean_table_omnireset/mdps/events.py`
- Modify: `tests/tasks/test_clean_table_omnireset.py`

**Step 1: Write the failing test**

Add `test_reset_pool_excludes_inside_box_states`. Build two object poses relative
to a box rotated 90 degrees around z: local x positions `0.08` and `0.11` with
z `0.05`. Assert the classifier returns only index `1`. Pass the inside state
alone and assert it raises `ValueError` containing `outside-box`.

**Step 2: Run the test to verify it fails**

Run:

```bash
env PYTHONPATH=. /home/yizhao/miniconda3/envs/env_isaaclab/bin/pytest -q \
  tests/tasks/test_clean_table_omnireset.py::test_reset_pool_excludes_inside_box_states
```

Expected: FAIL because `mdps.geometry` does not exist.

**Step 3: Implement the minimal filter**

Create `outside_box_state_indices(object_root_pose, box_root_pose, box_min,
box_max)`. Compute the object offset, apply the wxyz inverse-quaternion formula,
apply inclusive bounds, and return `torch.nonzero(~inside).squeeze(-1)`. Raise a
clear `ValueError` when the result is empty.

In `ResetSceneFromInstantDexterity.__init__`, call the helper once with the
loaded object and receptive-object poses plus `BOX_MIN` and `BOX_MAX`. In
`__call__`, sample offsets into this retained tensor and use the selected dataset
indices in `scene_state`.

**Step 4: Run focused tests**

Run:

```bash
env PYTHONPATH=. /home/yizhao/miniconda3/envs/env_isaaclab/bin/pytest -q \
  tests/tasks/test_clean_table_omnireset.py
```

Expected: 8 passed.

**Step 5: Verify the real dataset and static checks**

Load `/home/yizhao/yi/D2H/dataets/clean_table_bc` and run the helper on CPU.
Verify it retains 497 of 1,200 states. Compile the changed modules and run
`git diff --check`.

**Step 6: Commit**

```bash
git add src/tasks/clean_table_omnireset/mdps/geometry.py \
  src/tasks/clean_table_omnireset/mdps/events.py \
  tests/tasks/test_clean_table_omnireset.py \
  docs/plans/2026-08-11-omnireset-exclude-inside-box-design.md \
  docs/plans/2026-08-11-omnireset-exclude-inside-box.md
git commit -m "fix: exclude completed OmniReset states"
```
