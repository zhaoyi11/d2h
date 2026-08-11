# Clean Table OmniReset Frozen Viewer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a standalone Isaac Sim viewer that samples one clean-table OmniReset state and freezes it without sending actions.

**Architecture:** A task-specific script will follow the existing AppLauncher-first startup pattern, build `Clean_Table_OmniReset-v0`, reset once, pause the Omniverse timeline, and update only the application viewport. The existing task, reset event, and EMA hand controller remain unchanged.

**Tech Stack:** Python, IsaacLab `AppLauncher`, Gymnasium, Omniverse timeline API, pytest.

---

### Task 1: Add the frozen reset viewer

**Files:**
- Create: `scripts/visualize_clean_table_omnireset.py`
- Modify: `tests/tasks/test_clean_table_omnireset.py`

**Step 1: Write the failing source-level test**

Add a test that parses `scripts/visualize_clean_table_omnireset.py` and verifies:

```python
def test_frozen_reset_viewer_resets_and_pauses_without_actions() -> None:
    viewer_path = REPO_ROOT / "scripts/visualize_clean_table_omnireset.py"
    source = viewer_path.read_text()
    tree = ast.parse(source)

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [
        node.func.attr
        for node in calls
        if isinstance(node.func, ast.Attribute)
    ]
    assert "reset" in attributes
    assert "pause" in attributes
    assert "update" in attributes
    assert "step" not in attributes
    assert "reset_dataset_dir" in source
    assert 'Clean_Table_OmniReset-v0' in source
```

**Step 2: Run the test to verify it fails**

Run:

```bash
PYTHONPATH=. /home/yizhao/miniconda3/envs/env_isaaclab/bin/pytest -q \
  tests/tasks/test_clean_table_omnireset.py::test_frozen_reset_viewer_resets_and_pauses_without_actions
```

Expected: FAIL because the viewer script does not exist.

**Step 3: Implement the minimal viewer**

Create a script with this control flow:

```python
parser = argparse.ArgumentParser(...)
parser.add_argument("--num_envs", type=int, default=1, ...)
parser.add_argument("--reset_dataset_dir", type=str, default=None, ...)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Gym, IsaacLab tasks, omni.timeline, and src.tasks only after launch.

env_cfg = parse_env_cfg(
    "Clean_Table_OmniReset-v0",
    device=args_cli.device,
    num_envs=args_cli.num_envs,
    use_fabric=not args_cli.disable_fabric,
)
if args_cli.reset_dataset_dir is not None:
    env_cfg.reset_dataset_dir = args_cli.reset_dataset_dir

env = gym.make("Clean_Table_OmniReset-v0", cfg=env_cfg)
try:
    env.reset()
    omni.timeline.get_timeline_interface().pause()
    print("[INFO]: Reset state frozen. Close Isaac Sim to exit.")
    while simulation_app.is_running():
        simulation_app.update()
finally:
    env.close()
```

Close `simulation_app` after `main()` returns. Do not call `env.step()` and do not construct an action tensor.

**Step 4: Run focused tests**

Run:

```bash
PYTHONPATH=. /home/yizhao/miniconda3/envs/env_isaaclab/bin/pytest -q \
  tests/tasks/test_clean_table_omnireset.py
```

Expected: 7 passed.

**Step 5: Run static verification**

Run:

```bash
/home/yizhao/miniconda3/envs/env_isaaclab/bin/python -m py_compile \
  scripts/visualize_clean_table_omnireset.py
git diff --check
```

Expected: both commands exit successfully with no output.

**Step 6: Run an Isaac startup smoke check**

Launch the viewer headlessly with a bounded external timeout:

```bash
PYTHONPATH=. timeout --signal=INT 25s \
  /home/yizhao/miniconda3/envs/env_isaaclab/bin/python \
  scripts/visualize_clean_table_omnireset.py --headless
```

Expected before timeout: environment setup completes and the script prints `Reset state frozen`. No action-manager step is executed.

**Step 7: Commit**

```bash
git add scripts/visualize_clean_table_omnireset.py \
  tests/tasks/test_clean_table_omnireset.py \
  docs/plans/2026-08-11-clean-table-omnireset-viewer.md
git commit -m "feat: add frozen OmniReset state viewer"
```
