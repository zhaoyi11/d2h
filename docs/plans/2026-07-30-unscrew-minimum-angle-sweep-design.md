# Unscrew Minimum-Angle Sweep Design

## Goal

Find the smallest helical pre-unscrew rotation that allows the installed square leg to be pulled
clear under the current authored PhysX model and the existing realistic virtual-grasp limits. Report
both the observed threshold and a slightly larger recommended angle.

## Controlled Experiment

Each candidate follows the same two-phase motion:

1. Rotate about world/base `+Z` while rising at the authored 15 mm thread pitch.
2. Stop rotating and pull vertically until the object reaches the same final target height used by
   the verified three-turn trajectory.

The total requested rise remains fixed at 75 mm for every candidate. A candidate that uses fewer
turns therefore receives a longer vertical extraction. This ensures rotation angle is the independent
variable rather than allowing candidates with more rotation to benefit from a higher final target.
Rotation speed remains 90 degrees per second and vertical extraction speed remains 15 mm per second.

The experiment retains the validated conditions: Earth gravity, authored USD mass, 10 N maximum net
force, 0.2 N m maximum net torque, the existing PD gains, and the existing pose, lateral, clearance,
and saturation tolerances. Every candidate also runs a counterfactual with the identical XYZ targets
but a fixed initial orientation.

## Search and Decision Rule

First sweep 0.25 through 3.0 turns in 0.25-turn increments. Then sweep the interval between the
largest failure and the first success in 0.05-turn (18-degree) increments. Repeat candidates around
the boundary three times.

A candidate passes only when:

- measured quaternion-delta winding reaches the candidate angle within tolerance;
- helical rise and dense twist tracking remain within the validated tolerances;
- the object remains at least 30 mm clear for the full one-second hold;
- the identical fixed-orientation counterfactual never clears; and
- the run remains finite and within the fixed wrench limits.

The output reports the largest failing angle, smallest repeatedly passing angle, and a recommended
angle one 0.05-turn increment above the observed threshold. If no candidate passes by three turns,
the result is reported as bounded by the existing three-turn trajectory rather than extrapolated.

## Implementation and Verification

Extend the physical verifier with one candidate-turns argument while preserving its current defaults.
For each candidate, derive helical rise from thread pitch, adjust vertical extraction so total rise is
75 mm, and scale phase durations to preserve the validated angular and vertical speeds. Add focused
tests for these calculations and unchanged default behavior. A small sweep driver then executes the
candidate grid, records process status and metrics, and performs the three boundary repetitions.

Verification consists of focused unit/contract tests, one unchanged default three-turn regression,
the full coarse/fine physical sweep, and repeated boundary trials. No force, torque, gain, collision,
mass, gravity, or classification threshold may be tuned in response to the result.
