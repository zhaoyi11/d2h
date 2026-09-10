# Pairwise Collision Filtering Between object_table and object

## Problem

You want:

- `object_table` vs `object`: **no collision** (the leg should pass through the table top during insertion)
- `object_table` vs `robot`: **collision enabled** (robot should interact with the table surface)

Using `collision_enabled=False` on `object_table` disables ALL its collisions, which is too broad. Isaac Lab's `collision_group` (0 / -1) only controls inter-environment filtering, not intra-environment pairwise filtering.

## Solution: `UsdPhysics.FilteredPairsAPI`

PhysX supports pairwise collision filtering via the USD schema `UsdPhysics.FilteredPairsAPI`. This lets you mark two specific prims as a "filtered pair" so they ignore each other, while still colliding with everything else. This has **precedence over CollisionGroup filtering**.

Isaac Lab does not have a built-in helper for this, but it can be applied via a custom `startup` event that runs after scene creation.

## Changes

### 1. Fix `object_table` collision_props in [`twist_env_cfg.py`](src/env/tasks/twist_ufactory850/twist_env_cfg.py)

Remove the broken `disable_collision=True` line and keep collisions **enabled** (either remove `collision_props` entirely or use `CollisionPropertiesCfg()` with defaults):

```python
# Line 127 - remove entirely or replace with:
collision_props=sim_utils.CollisionPropertiesCfg(),
```

### 2. Add a new event function in [`events.py`](src/env/tasks/twist_ufactory850/mdps/events.py)

Write a `startup`-mode event function that applies `FilteredPairsAPI` between the `object_table` and `object` prims in every environment:

```python
def apply_filtered_collision_pairs(
    env: ManagerBasedEnv,
    asset_cfg_a: SceneEntityCfg,
    asset_cfg_b: SceneEntityCfg,
):
    """Disable collision between two specific assets using FilteredPairsAPI."""
    from pxr import UsdPhysics
    import omni.isaac.core.utils.stage as stage_utils

    stage = stage_utils.get_current_stage()
    asset_a = env.scene[asset_cfg_a.name]
    asset_b = env.scene[asset_cfg_b.name]

    # Resolve all env prim paths (e.g. /World/envs/env_0/ObjectTable, ...)
    paths_a = sim_utils.find_matching_prim_paths(asset_a.cfg.prim_path)
    paths_b = sim_utils.find_matching_prim_paths(asset_b.cfg.prim_path)

    for path_a, path_b in zip(paths_a, paths_b):
        prim_a = stage.GetPrimAtPath(path_a)
        filtered_api = UsdPhysics.FilteredPairsAPI.Apply(prim_a)
        filtered_api.CreateFilteredPairsRel().AddTarget(path_b)
```

The `FilteredPairsAPI` applied at the rigid body root is sufficient -- PhysX propagates the filter to all child collision shapes.

### 3. Register the event in [`twist_env_cfg.py`](src/env/tasks/twist_ufactory850/twist_env_cfg.py) EventCfg

```python
filter_object_table_collision = EventTerm(
    func=mdp.apply_filtered_collision_pairs,
    mode="startup",
    params={
        "asset_cfg_a": SceneEntityCfg("object_table"),
        "asset_cfg_b": SceneEntityCfg("object"),
    },
)
```

## Why This Works

- `FilteredPairsAPI` tells PhysX to skip collision detection for just this pair.
- `object_table` still has its collision meshes active, so the robot will collide with it normally.
- The `object` (table leg) will pass through `object_table` (table top) freely.
- Works with `replicate_physics=False` since we iterate over all env prim paths.

## Alternate Simpler Approach (if FilteredPairsAPI has issues)

If `FilteredPairsAPI` causes problems with the cloner or GPU pipeline, you can also use the Isaac Sim utility directly:

```python
from omni.physx.scripts.utils import addPairFilter
addPairFilter(stage, [path_a, path_b])
```

This does the same thing under the hood but is the "official" Isaac Sim helper.

## Todos

- [ ] Fix object_table collision_props: remove `disable_collision=True`, keep collision enabled with default `CollisionPropertiesCfg()`
- [ ] Add `apply_filtered_collision_pairs` function to `events.py` using `UsdPhysics.FilteredPairsAPI`
- [ ] Add `filter_object_table_collision` EventTerm (startup mode) to `EventCfg` in `twist_env_cfg.py`
