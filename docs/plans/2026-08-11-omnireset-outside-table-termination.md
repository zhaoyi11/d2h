# Clean Table OmniReset Outside-Table Termination Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Terminate OmniReset episodes when the object center leaves the tabletop footprint or falls below its surface.

**Architecture:** A task-local MDP predicate will transform object position into the loaded table frame and compare it against the configured cuboid half extents and top surface. The environment termination config will replace its broad world-frame bounds with this table-relative predicate.

**Tech Stack:** Python, PyTorch, IsaacLab manager terms and frame transforms, pytest.

---

### Task 1: Add the table-relative termination

**Files:**
- Modify: `src/tasks/clean_table_omnireset/mdps/task_mdps.py`
- Modify: `src/tasks/clean_table_omnireset/env_cfg.py:281-287`
- Modify: `tests/tasks/test_clean_table_omnireset.py`

**Step 1: Write the failing test**

Extend the fake table asset with `root_quat_w` and make the fake frame transform
respect wxyz rotation. Add `test_object_outside_table_uses_loaded_table_frame`
covering an inside lifted pose, horizontal boundary crossing, below-surface
pose, and a 90-degree-rotated table.

**Step 2: Run the test to verify it fails**

Run:

```bash
env PYTHONPATH=. /home/yizhao/miniconda3/envs/env_isaaclab/bin/pytest -q \
  tests/tasks/test_clean_table_omnireset.py::test_object_outside_table_uses_loaded_table_frame
```

Expected: FAIL because `object_outside_table` does not exist.

**Step 3: Implement the minimal predicate and config**

Add `object_outside_table(env, table_half_extents=(0.4, 0.75),
table_half_height=0.02, object_cfg=..., table_cfg=...)`. Use
`subtract_frame_transforms` to obtain object position in the table frame and
return horizontal-outside OR below-tabletop. Export it and configure
`TerminationsCfg.object_outside_table` to use the object and table entities.

**Step 4: Run focused verification**

Run the complete focused test file, compile the changed Python files, and run
`git diff --check`. Expected: 9 tests pass and all static checks exit zero.

**Step 5: Commit**

```bash
git add src/tasks/clean_table_omnireset/mdps/task_mdps.py \
  src/tasks/clean_table_omnireset/env_cfg.py \
  tests/tasks/test_clean_table_omnireset.py \
  docs/plans/2026-08-11-omnireset-outside-table-termination-design.md \
  docs/plans/2026-08-11-omnireset-outside-table-termination.md
git commit -m "fix: terminate OmniReset objects outside table"
```
