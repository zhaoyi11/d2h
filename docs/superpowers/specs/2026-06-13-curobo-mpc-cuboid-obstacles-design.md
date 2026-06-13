# cuRobo MPC — static + dynamic cuboid obstacles

**Date:** 2026-06-13
**Status:** Approved (design), pending implementation
**Component:** `CommandHandBaseCuroboMpcAction` / `pick_insert demo`

## Goal

Add two cuboid obstacles to the `pick_insert demo` environment for the cuRobo MPC arm
controller to avoid:

1. **One static obstacle** — a fixed pillar the arm must route around.
2. **One dynamic obstacle** — a box following a scripted (kinematic) path that periodically
   sweeps across the hand's route, so the MPC visibly re-plans around a moving object.

Both are real, visible Isaac Lab scene props. The dynamic one's live pose is mirrored into the
cuRobo collision world every control step.

## Architecture (Approach A — generic action + scripted event)

Chosen over the "all-in-the-action" alternative for separation of concerns and reusability. The
cost is a one-control-step lag between the prop's visual pose and the pose the planner sees,
which is invisible at ~60 Hz.

Three independent pieces:

1. **Scene props** (in `SceneCfg`) — declarative geometry, the single source of truth for each
   obstacle's pose.
2. **Generic obstacle tracking** (in `CommandHandBaseCuroboMpcAction`) — "mirror these named
   scene assets into the cuRobo world each step." Knows nothing about *how* obstacles move.
3. **Scripted motion** (a `task_mdps` event func + per-step `EventTerm`) — drives the dynamic
   prop's pose. Knows nothing about cuRobo.

### Frame note

The cuRobo world is defined in the **robot base frame** (`panda_link0`). The existing `table`
obstacle entry uses the same pose `(0.55, 0, 0.235)` as the table scene prop's world pose, which
confirms the robot base sits at the world origin in this env. We still convert each tracked
asset's world pose into the robot base frame with `subtract_frame_transforms(root_pos_w,
root_quat_w, obj_pos_w, obj_quat_w)` so the code is correct regardless of where the robot is
placed.

## Components

### 1. Scene — two kinematic cuboid props

In `SceneCfg` ([env_cfg.py](../../../src/tasks/pick_insert%20demo/env_cfg.py)), spawned exactly
like the existing `table` prop (`CuboidCfg`, `RigidBodyPropertiesCfg(kinematic_enabled=True)`,
`CollisionPropertiesCfg()`, `visible=True`) so physics never moves them and we set their pose
explicitly.

- **`static_obstacle`** — `prim_path="{ENV_REGEX_NS}/StaticObstacle"`, size `(0.05, 0.05, 0.25)`,
  init pos `(0.5, -0.25, 0.38)`, identity rot. A pillar standing on the table off to the side of
  the peg→hole path.
- **`dynamic_obstacle`** — `prim_path="{ENV_REGEX_NS}/DynamicObstacle"`, size `(0.06, 0.06,
  0.06)`, init pos `(0.4, 0.05, 0.45)`, identity rot. A box that oscillates in Y across the
  hand's route.

All dimensions/positions are tunable defaults; they are chosen to force the arm to route around
the obstacles without blocking pick (peg at `(0.45, 0.2, 0.3)`) or insert (hole at
`(0.35, 0.0, 0.275)`).

### 2. cuRobo world — pre-declare both cuboids

Extend the action's `obstacle_cuboids` so each obstacle gets a permanent buffer slot at build
time (poses in robot base frame, matching the scene props' world poses):

```python
obstacle_cuboids={
    "table":            {"dims": [0.8, 1.5, 0.04],  "pose": [0.55,  0.0,  0.235, 1, 0, 0, 0]},
    "static_obstacle":  {"dims": [0.05, 0.05, 0.25], "pose": [0.5,  -0.25, 0.38,  1, 0, 0, 0]},
    "dynamic_obstacle": {"dims": [0.06, 0.06, 0.06], "pose": [0.4,   0.05, 0.45,  1, 0, 0, 0]},
}
```

`static_obstacle` is never touched again after build. `dynamic_obstacle`'s slot is rewritten each
step via `update_obstacle_pose`, which is an **in-place tensor write**
(`data_cuboid.py:update_pose` → `self.inv_pose[env_idx, idx, :7] = ...`) and therefore
**CUDA-graph safe** with `use_cuda_graph=True`.

### 3. Action — new `dynamic_obstacle_assets` field

Add to `CommandHandBaseCuroboMpcActionCfg`:

```python
dynamic_obstacle_assets: dict[str, str] = {}
"""Map of cuRobo-cuboid-name -> scene-asset-name. Each control step the action reads the scene
asset's world pose, converts to the robot base frame, and rewrites that cuRobo cuboid's pose.
The named cuRobo cuboids MUST also be declared in `obstacle_cuboids` (so a buffer slot exists).
Empty (default) reproduces the current behavior."""
```

For the demo: `dynamic_obstacle_assets={"dynamic_obstacle": "dynamic_obstacle"}`.

In `CommandHandBaseCuroboMpcAction.__init__`: resolve each scene asset
(`self._dyn_obstacle_assets = {curobo_name: env.scene[asset_name] for ...}`).

In `process_actions`, **before** `optimize_next_action`, for each `(curobo_name, asset)`:

```python
pos_b, quat_b = subtract_frame_transforms(
    self._asset.data.root_pos_w, self._asset.data.root_quat_w,
    asset.data.root_pos_w, asset.data.root_quat_w,
)
pose = self._Pose(position=pos_b[:1].contiguous(), quaternion=quat_b[:1].contiguous())
self._mpc.scene_collision_checker.update_obstacle_pose(curobo_name, pose, env_idx=0)
```

(`env_idx=0` only — see Known Limitation. `[:1]` because the shared world holds a single pose.)

### 4. Scripted-motion event

A new func in `task_mdps`, e.g.:

```python
def move_dynamic_obstacle(env, env_ids, asset_cfg, center, axis, amplitude, freq, phase=0.0):
    """Kinematically drive a prop along a sinusoid: pos = center + axis * amplitude * sin(2*pi*freq*t + phase)."""
```

- Time base: a global sim phase so all envs and the cuRobo world agree. Use
  `env.sim.current_time` (or `env.common_step_counter * env.step_dt`) — **not** per-env
  `episode_length_buf`, which would desync envs from the single shared cuRobo world.
- Writes the prop's root pose for every env via `write_root_pose_to_sim` (world poses =
  `env.scene.env_origins + center + axis*amplitude*sin(...)`, broadcast across envs).
- Registered as an `EventTerm(mode="interval", interval_range_s=(0.0, 0.0), ...)` in `EventCfg` —
  the same per-step pattern already used by `gravity_compensation_assist` in this demo.

Defaults: `center=(0.4, 0.05, 0.45)`, `axis=(0, 1, 0)`, `amplitude=0.20`, `freq=0.25` Hz.

## Data flow (per control step)

```
EventTerm.move_dynamic_obstacle  --writes-->  dynamic_obstacle prop root pose (sim)   [step t-1]
        |
        v  (read next step)
action.process_actions:
    read dynamic_obstacle.data.root_pose_w  ->  subtract_frame_transforms -> base frame
    mpc.scene_collision_checker.update_obstacle_pose("dynamic_obstacle", pose, env_idx=0)
    mpc.update_goal_tool_poses(goal)
    mpc.optimize_next_action(ref_js)  -> cmd_pos
action.apply_actions: set_joint_position_target(cmd_pos)
```

## Error handling / edge cases

- **Missing slot:** if a `dynamic_obstacle_assets` key is not also in `obstacle_cuboids`,
  `update_obstacle_pose` raises (obstacle not found). Validate in `__init__` and raise a clear
  error early rather than per-step.
- **Empty map:** default `{}` ⇒ no tracking, behavior identical to today (backward compatible).
- **Reset:** obstacles are kinematic and pose-driven; nothing extra needed on `reset`. The
  scripted motion is continuous in global sim time (it does not restart per episode — acceptable
  and arguably nicer for a demo).

## Known limitation — `multi_env=False`

The MPC uses `multi_env=False`, i.e. **one shared cuRobo collision world** for all `num_envs`
(default 4096, typically overridden small for the demo). The dynamic obstacle is synced from
`env_idx=0` only, so the planner uses env-0's obstacle pose for every env. This is correct for a
single-/few-env demo where the scripted motion is identical (in base frame) across envs. For
per-env-distinct obstacles, switch to `multi_env=True` and loop `update_obstacle_pose` over each
`env_idx` (more memory/compute). Out of scope unless requested.

## Testing / verification

- Launch the demo with a small `num_envs` and confirm in the viewer: the static pillar is fixed;
  the dynamic box oscillates in Y; the arm routes around both and the peg still reaches the bore.
- Sanity-check no CUDA-graph error at runtime with `use_cuda_graph=True` (validates the in-place
  pose update path).
- Regression: with `dynamic_obstacle_assets={}` the controller behaves exactly as before.

## Out of scope

- Per-env distinct obstacles (`multi_env=True`).
- Non-cuboid obstacle geometry (sphere/mesh) — cuboids only.
- Physics-driven (free rigid body) or externally-teleoped obstacle motion.
- Making the obstacles part of any RL reward/observation.
