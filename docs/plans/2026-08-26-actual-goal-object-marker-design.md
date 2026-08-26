# Actual Goal Object Marker Design

## Goal

Show each environment's actual sampled object geometry at its commanded goal pose in `scripts/instant_dexterity.py`.

## Design

When a scene uses `MultiUsdFileCfg`, inspect each spawned object prim once and read its authored USD reference. Deduplicate the selected paths into `VisualizationMarkers` prototypes, and retain one prototype index per environment. Every marker update supplies those indices together with the existing goal positions and orientations.

Single-USD and procedural objects retain their current behavior with prototype index zero. Missing object prims, missing references, or more than one selected reference raise a descriptive runtime error during marker setup instead of silently showing the wrong geometry.

Object selection is fixed when the scene is spawned, so the selected references and indices are computed once, not every simulation step. This keeps the hot visualization callback limited to pose updates.

## Verification

- Unit-test selected-reference extraction from fake per-environment prims.
- Unit-test prototype deduplication and per-environment indices.
- Unit-test that marker updates pass the retained indices.
- Run the committed instant-dexterity test suite and Python compilation.
- Validate against a real four-environment MultiUsd scene.
