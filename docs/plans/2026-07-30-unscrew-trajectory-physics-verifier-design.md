# Unscrew Trajectory Physics Verifier Design

## Goal

Verify that the scripted unscrew trajectory is physically feasible in the authored object/socket
collision model. The verifier must advance Isaac Sim normally and must not write the object's pose or
velocity after initialization. It intentionally isolates trajectory and contact feasibility from the
Franka controller, LEAP-hand grasp acquisition, and frozen low-level policy.

## Approach

Add `scripts/verify_unscrew_trajectory_physics.py`. The script launches Isaac Sim through
`AppLauncher`, creates a robotless `InteractiveScene` from the unscrew scene configuration, settles
the installed assemblies, and captures their live poses. It then generates dense targets from
`build_unscrew_object_pose_sequence` and drives each dynamic leg with a bounded world-frame SE(3)
PD wrench using `RigidObject.set_external_force_and_torque`. Each control iteration writes only the
external wrench, steps simulation, and updates scene buffers.

The test uses two environments under Earth gravity:

1. The helical trial follows three positive-yaw turns with the authored 15 mm pitch, then performs
   the 30 mm extraction and hold.
2. The counterfactual trial follows the same vertical profile while holding its initial orientation.

The virtual grasp is limited to 10 N net force and 0.2 N m net torque. These bounds are conservative
relative to the project's sub-20 N fingertip contact range and 0.5 N m finger actuator limits. The
script reports how often each bound saturates so an apparent failure caused by insufficient control
authority is distinguishable from collision-induced jamming.

## Measurements and Result Classification

The verifier records actual accumulated yaw by integrating world-frame angular velocity about the
installed `+Z` thread axis. This avoids quaternion wrapping after complete turns. It also records
vertical rise, lateral drift, position and orientation tracking errors, peak wrench, saturation
fractions, and final bolt clearance using the same bolt bounds and socket-top convention as
`StableUnscrewSuccess`.

Results are classified as:

- **PASS:** the helical trial completes approximately three physical turns, remains within tracking
  tolerances, clears the socket, and holds clear while the straight-pull trial remains blocked.
- **INVALID PHYSICS MODEL:** the straight-pull trial also clears, so the collision model does not
  demonstrate screw-like behavior.
- **TRAJECTORY INFEASIBLE:** the helical trial stalls or cannot clear while remaining within the
  agreed wrench and tracking limits.
- **CONTROLLER INCONCLUSIVE:** instability or persistent wrench saturation prevents a trustworthy
  feasibility conclusion.

The script exits with status zero only for `PASS` and prints per-trial diagnostics for every other
classification.

## Verification

Pure-Torch unit tests cover target sampling, wrench clamping, accumulated-yaw calculation, clearance,
and result classification without launching Isaac Sim. A headless integration run in
`env_isaaclab` then verifies that the script loads the real USD assets, never teleports the leg during
the trial, advances physics, and emits one of the defined classifications with complete diagnostics.
