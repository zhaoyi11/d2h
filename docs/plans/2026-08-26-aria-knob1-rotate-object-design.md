# ARIA Knob1 Rotate-Object Design

The rotate-object task will use ARIA's `knob1` geometry instead of a selected VisDex object. The
source `knob1.usd` and its complete relative-reference folder will live under
`src/assets/aria/knob1/` so the task has no dependency on the external ARIA checkout.

The source asset is a two-body articulation whose fixed `base` has no rendered bounds and whose
`handle` rotates about an authored Z-axis joint. The current task intentionally represents the
manipulated object as one `RigidObject`; its command, observations, rewards, resets, and goal marker
all track that rigid root. A small `knob1_handle.usda` composition layer will therefore reference
the copied source asset, remove its articulation APIs, and deactivate the source `base` and
`Joints` prims. This preserves the exact handle geometry, materials, and collision hierarchy while
leaving one rigid body for the task's existing world-anchored Z-axis joint.

ARIA authors knob1 at scale `1.0` with visible bounds from approximately `z=0.0` to `z=0.03`.
The D2H table top is `z=0.255`, so the object root will be `(0.55, 0.20, 0.255)`. Fingertip contact
filters will target `/Object/handle`. Tests will verify the copied dependency closure, the derived
stage topology and bounds, the scene configuration, and a real cloned IsaacLab scene plus goal
marker.
