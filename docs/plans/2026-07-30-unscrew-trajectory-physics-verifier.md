# Unscrew Trajectory Physics Verifier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a bounded-wrench Isaac Sim test that determines whether the authored three-turn helix physically clears the square-leg socket while an otherwise identical straight pull remains blocked.

**Architecture:** Put deterministic tensor math and result classification in a small task-local module so it can be unit-tested without launching Isaac Sim. Add one standalone `AppLauncher` script that creates two robotless unscrew scenes, generates dense targets from the existing trajectory builder, applies only bounded external wrenches to the dynamic objects, steps physics, and classifies the measured results.

**Tech Stack:** Python 3.11, PyTorch, Isaac Lab `InteractiveScene`/`RigidObject`, Isaac Sim PhysX, pytest.

---

### Task 1: Add tested bounded-wrench and classification primitives

**Files:**
- Create: `src/tasks/unscrew/mdps/physics_verifier.py`
- Create: `tests/tasks/unscrew/test_physics_verifier.py`
- Modify: `.gitignore:70-73`

**Step 1: Make the new test path trackable**

Extend the existing tests allow-list without changing its treatment of any other test directory:

```gitignore
tests/*
!tests/policy/
!tests/policy/test_vae.py
!tests/tasks/
tests/tasks/*
!tests/tasks/unscrew/
tests/tasks/unscrew/*
!tests/tasks/unscrew/test_physics_verifier.py
```

**Step 2: Write failing unit tests**

Create tests that load `physics_verifier.py` directly with `importlib.util.spec_from_file_location`, avoiding package registration and simulator startup. Cover:

```python
def test_bounded_pd_wrench_tracks_pose_and_clamps_norms(): ...
def test_straight_pull_targets_keep_initial_orientation(): ...
def test_accumulate_world_yaw_integrates_angular_velocity(): ...
def test_bolt_bottom_clearance_uses_rotated_aabb_corners(): ...
def test_classify_pass_requires_helix_clear_and_pull_blocked(): ...
def test_classify_detects_invalid_collision_model(): ...
def test_classify_marks_saturated_failed_trial_inconclusive(): ...
def test_classify_marks_unsaturated_failed_trial_infeasible(): ...
```

Use identity and 90-degree Z quaternions with hand-computable expectations. For classification, construct `TrialSummary` values directly so each branch has one isolated cause.

**Step 3: Run tests to verify they fail**

Run:

```bash
/home/yizhao/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/tasks/unscrew/test_physics_verifier.py -q
```

Expected: FAIL because `src/tasks/unscrew/mdps/physics_verifier.py` does not exist.

**Step 4: Implement the minimal pure-tensor module**

Create these public types and functions:

```python
class VerificationResult(str, Enum):
    PASS = "PASS"
    INVALID_PHYSICS_MODEL = "INVALID PHYSICS MODEL"
    TRAJECTORY_INFEASIBLE = "TRAJECTORY INFEASIBLE"
    CONTROLLER_INCONCLUSIVE = "CONTROLLER INCONCLUSIVE"

@dataclass(frozen=True)
class WrenchCommand:
    force_w: torch.Tensor
    torque_w: torch.Tensor
    force_saturated: torch.Tensor
    torque_saturated: torch.Tensor

@dataclass(frozen=True)
class TrialSummary:
    finite: bool
    held_clear: bool
    accumulated_yaw: float
    final_position_error: float
    final_orientation_error: float
    max_lateral_drift: float
    force_saturation_fraction: float
    torque_saturation_fraction: float
    peak_force: float
    peak_torque: float
    final_clearance: float

def bounded_pd_wrench(
    current_pose_w, target_pose_w, linear_velocity_w, angular_velocity_w,
    mass, gravity_w, *, position_stiffness, position_damping,
    rotation_stiffness, rotation_damping, max_force, max_torque,
) -> WrenchCommand: ...

def straight_pull_targets(helical_targets: torch.Tensor) -> torch.Tensor: ...

def accumulate_world_yaw(accumulated_yaw, angular_velocity_w, dt): ...

def bolt_bottom_clearance(
    object_pos_r, object_quat_r, bolt_corners, socket_top_z,
) -> torch.Tensor: ...

def classify_verification(
    helix, straight_pull, *, expected_yaw, yaw_tolerance=0.35,
    position_tolerance=0.005, orientation_tolerance=0.35,
    lateral_tolerance=0.005, saturation_inconclusive_fraction=0.90,
) -> VerificationResult: ...
```

Implementation constraints:

- Compute quaternion error as `target * conjugate(current)` in `(w, x, y, z)` order and convert the shortest representative to an axis-angle vector.
- Add gravity compensation `-mass * gravity_w` before norm clamping the force.
- Clamp force and torque by vector norm, not per axis.
- Mark saturation before clamping.
- `straight_pull_targets` must clone the helix and replace every quaternion with the first quaternion while retaining every XYZ target.
- `bolt_bottom_clearance` must rotate all eight supplied local AABB corners before taking the minimum Z relative to the socket top.
- Classification order: non-finite is inconclusive; a clear straight-pull trial invalidates the physics model; a clear, accurately tracked helix with the expected accumulated yaw passes; a failed helix with persistent saturation is inconclusive; all other finite failures are infeasible.

**Step 5: Run unit tests**

Run the Task 1 pytest command again.

Expected: all eight tests PASS.

**Step 6: Commit Task 1**

```bash
git add .gitignore src/tasks/unscrew/mdps/physics_verifier.py \
  tests/tasks/unscrew/test_physics_verifier.py
git commit -m "test: add unscrew physics verifier primitives"
```

---

### Task 2: Add the standalone physical feasibility runner

**Files:**
- Create: `scripts/verify_unscrew_trajectory_physics.py`
- Test: `tests/tasks/unscrew/test_physics_verifier.py`

**Step 1: Add failing source-contract tests**

Add an AST/text test for the standalone runner that verifies:

- `AppLauncher` is constructed before runtime Isaac Lab imports.
- The runner calls `set_external_force_and_torque`, `scene.write_data_to_sim`, `sim.step`, and `scene.update`.
- The runner never calls `write_root_pose_to_sim` or `write_root_velocity_to_sim`.
- It builds targets through `build_unscrew_object_pose_sequence` and uses `straight_pull_targets` for environment 1.
- It uses exactly two environments and Earth gravity `(0.0, 0.0, -9.81)`.

**Step 2: Run the new test to verify it fails**

Run the Task 1 pytest command.

Expected: FAIL because `scripts/verify_unscrew_trajectory_physics.py` does not exist.

**Step 3: Implement the standalone runner**

Follow the existing startup pattern in `scripts/visualize_unscrew_object_trajectory.py`:

1. Parse `--settle_duration` (1.0 s), `--turn_duration` (12.0 s), `--extraction_duration` (2.0 s), `--hold_duration` (1.0 s), and standard `AppLauncher` arguments.
2. Launch the app before importing `isaaclab.sim`, `InteractiveScene`, task configs, or Isaac Lab math.
3. Create `SimulationCfg(dt=1/120, gravity=(0, 0, -9.81), device=args_cli.device)`.
4. Instantiate `SceneCfg(num_envs=2, env_spacing=3.0, replicate_physics=False)` with `robot=None`.
5. Reset, settle both dynamic legs with zero external wrench, and capture their live world poses.
6. Compute dense segment counts:

```python
turn_steps_per_segment = round(turn_duration / 12 / dt)
extract_steps = round(extraction_duration / dt)
hold_steps = round(hold_duration / dt)
segment_steps = (0, 0) + (turn_steps_per_segment,) * 12 + (extract_steps, hold_steps)
```

7. Build each environment's helical targets from its live pose; replace environment 1's quaternion sequence with `straight_pull_targets`.
8. At every target sample, read the live object pose and velocities, compute `bounded_pd_wrench`, store it with `set_external_force_and_torque(..., is_global=True)`, call `scene.write_data_to_sim()`, `sim.step(render=not args_cli.headless)`, then `scene.update(dt)`.
9. Use these agreed realistic defaults:

```python
MAX_FORCE = 10.0       # N
MAX_TORQUE = 0.2       # N m
POSITION_STIFFNESS = 200.0
POSITION_DAMPING = 4.0
ROTATION_STIFFNESS = 0.5
ROTATION_DAMPING = 0.01
CLEARANCE_MARGIN = 0.030
```

10. Accumulate actual yaw from `object.data.root_ang_vel_w[:, 2] * dt`; do not infer completed turns from the final quaternion.
11. Transform the object into the receptive-object frame and compute clearance with `BOLT_AABB_MIN`, `BOLT_AABB_MAX`, and `SOCKET_TOP_Z` from `task_mdps.py`.
12. Track consecutive clear samples during the hold and construct one `TrialSummary` per environment.
13. Print target settings plus both summaries, print the classification, and return exit code zero only for `PASS`.
14. In `finally`, clear the external wrench, close the scene/app cleanly, and preserve the existing `AppLauncher` lifecycle.

**Step 4: Run unit and contract tests**

Run:

```bash
/home/yizhao/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/tasks/unscrew/test_physics_verifier.py -q
/home/yizhao/miniconda3/envs/env_isaaclab/bin/python -m compileall -q \
  src/tasks/unscrew/mdps/physics_verifier.py \
  scripts/verify_unscrew_trajectory_physics.py
```

Expected: all tests PASS and compileall exits zero.

**Step 5: Commit Task 2**

```bash
git add scripts/verify_unscrew_trajectory_physics.py \
  tests/tasks/unscrew/test_physics_verifier.py
git commit -m "feat: add physical unscrew trajectory verifier"
```

---

### Task 3: Run and validate the real PhysX experiment

**Files:**
- Modify only if evidence requires it: `scripts/verify_unscrew_trajectory_physics.py`
- Modify only if a corrected contract needs coverage: `tests/tasks/unscrew/test_physics_verifier.py`

**Step 1: Run the headless integration experiment**

Run:

```bash
conda activate env_isaaclab
python scripts/verify_unscrew_trajectory_physics.py --headless
```

Expected: both UW-Lab USD assets load, the simulation advances for settle + turn + extraction + hold, and complete per-trial diagnostics plus one defined classification are printed.

**Step 2: Interpret evidence before changing anything**

- `PASS`: retain gains and thresholds.
- `INVALID PHYSICS MODEL`: do not tune the controller; report that straight pulling also clears.
- `TRAJECTORY INFEASIBLE`: retain the agreed realistic bounds and report the measured stall.
- `CONTROLLER INCONCLUSIVE`: inspect non-finite state, lateral drift, tracking error, and saturation fractions. Change only one PD gain at a time, never `MAX_FORCE` or `MAX_TORQUE`, and rerun after each change.

No tuning may convert an invalid collision model into a pass, and no direct pose/velocity write may be introduced.

**Step 3: Re-run all verification**

Run:

```bash
/home/yizhao/miniconda3/envs/env_isaaclab/bin/python -m pytest \
  tests/tasks/unscrew/test_physics_verifier.py -q
/home/yizhao/miniconda3/envs/env_isaaclab/bin/python -m compileall -q src scripts
python scripts/verify_unscrew_trajectory_physics.py --headless
git diff --check
```

Expected: unit/contract tests and compile checks pass; the physical run emits a trustworthy classification; `git diff --check` is clean.

**Step 4: Commit evidence-based tuning, if any**

Skip this commit if Task 3 required no code change. Otherwise:

```bash
git add scripts/verify_unscrew_trajectory_physics.py \
  tests/tasks/unscrew/test_physics_verifier.py
git commit -m "fix: stabilize unscrew physics verification"
```

**Step 5: Request review and finish the branch**

Use `superpowers:requesting-code-review`, address only verified findings, then use
`superpowers:verification-before-completion` and `superpowers:finishing-a-development-branch`.
