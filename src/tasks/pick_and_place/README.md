# Pig pick-and-place

```bash
conda activate env_isaaclab
python scripts/instant_dexterity.py --task PickAndPlace_HRL-v0 --num_envs 1 --steps 3600 \
  --low_level_checkpoint /home/yizhao/yi/dex_reorient/model_14999.pt
```

Add `--headless --seed 0` for a repeatable run. The task uses cuRobo for the
Franka arm and the existing frozen LEAP policy (155 observations, 16 actions).
It reads `pick_and_place_pig_mano_cropped_isaac.npz` directly. MANO joints are
not used to control LEAP, and the original recording is unchanged.

The pig uses its textured mesh at native scale, a convex collision hull and a
fixed 0.2 kg mass. It starts 2 mm above the table, settles under gravity, and
the command captures its resting pose. Gravity is `(0, 0, -1.81)` throughout,
matching the existing clean-table controller setup. Object, table and box
resets are deterministic; the robot joints and MPC state are reset together.

The active stages are approach, confirm grasp, recorded carry, release, and
retreat. The shared deposit stage has zero waypoints and is skipped. Grasping requires 15 consecutive contact steps (0.5 seconds),
with a 90-step retry timeout, to seat the pig before lifting. Carry frames
320–511 run from the recording's pre-lift minimum to
its height peak; every 28th frame and both endpoints are included (8 recorded poses). The
path is rotated about world Z and scaled in XY to reach the box. Height is
scaled separately to reach the drop target at box-local `(0, 0, 0.10)`. Relative object
rotations start from the pig's live settled orientation. All recorded poses
use XYZ/WXYZ; world targets are converted into each robot's root frame.
The wrist uses clean-table's grasp anchor while the frozen policy receives the
recorded object orientation goals.

The carry ends directly at the drop target, with no separate descent. Release
and retreat repeat the final object pose. Clean-table's hand gate opens the fingers for release
and the arm retreats home. Success requires the object center inside the box,
the arm at its retreat goal, the hand clear, and object speed below 0.05 m/s
for five consecutive control steps. Pre-release drops use clean-table's grasp
recovery. A fall below the tabletop ends the episode and resets the scene.
This reuses clean-table's center-based containment check.

Physics runs at 120 Hz and control at 30 Hz, with a 120-second timeout.
Waypoints advance when tracking/contact gates pass, rather than at the
recording's original 200 Hz. Object pose tolerances are 5 cm and 1.0 rad
(about 57 degrees) for all active stages. The hand-base tracking tolerances,
contact confirmation and box-placement success checks retain their existing
settings. `pick_and_place_frame` reports the source frame
during approach/grasp/carry and `-1` during synthetic release/retreat;
`pick_and_place_progress` covers all 11 targets. Existing clean-table log lines
report grasp and placement state.

Task-local calibration lives in `PickAndPlaceTrajectoryCommandCfg` and
`PickAndPlaceEnvCfg`: the carry interval/stride, grasp anchor, tracking/contact
tolerances and `box_target_offset` can be adjusted without changing other tasks.
The inherited `above_box_offset` field is unused by this task.

Validation:

```bash
python -m pytest -q tests/tasks/test_pick_and_place.py
python tests/tasks/check_pick_and_place_runtime.py --headless
```

The unit checks cover recorded sampling, stage layout, adapted path endpoints,
relative rotations and registration. Runtime checks use two real IsaacLab
environments for support, settling, world/root transforms, manager dimensions,
contact gating, release/retreat, success rejection and independent resets.
The reset check also warms the MPC under `torch.inference_mode()` and forces
an automatic timeout reset. The common `reset_arm_mpc` helper temporarily enables
gradients only for cuRobo and replaces its inference-created action buffer with
a normal tensor before reinitializing the optimizer.

With the relaxed 5 cm / 1.0 rad tolerances, the 900-step seed-0 rollout completed
without runtime errors, including an automatic reset after an off-table drop.
The pig establishes a grasp and carries, but still slips before placement.
The release and success checks pass with controlled asset states; completed
box placement has not yet been demonstrated with this frozen policy.

For the original MANO skeleton and object replay, without robot control:

```bash
python src/tasks/pick_and_place/vis_traj_isaaclab.py --loop
```

## Experimental calibrated-anchor EE targets

```bash
conda activate env_isaaclab
python scripts/instant_dexterity.py --task PickAndPlace_HRL-v0 --num_envs 1 --steps 3600 \
  --low_level_checkpoint /home/yizhao/yi/dex_reorient/model_14999.pt \
  --hand_anchor_calibration /home/yizhao/yi/D2H/datasets/anchor_calibration/full_calibration/anchors.json
```

This optional mode reads the recording's `hand_root_pos` and `hand_root_quat_wxyz` as the MANO XML `right_palm` **body** pose. For each sampled source frame, it computes `T_O_A = inverse(T_demo_O) @ T_demo_M @ T_M_A`, then commands `T_B_L = T_B_O_goal @ T_O_A @ inverse(T_L_A)`. B is the Franka robot root, L is the LEAP `base` palm, and A is the calibrated anchor. cuRobo already uses `base` as its tool frame, so this target is passed directly through `command[:, 7:14]`.

The task's adapted object path transports the recorded hand-to-object relationship. Hand/contact offsets remain rigid in meters; they are not stretched by workspace path scaling. Existing bounded corrections still apply. Without the flag, the existing command behavior is retained.

Synthetic release frames retain the last recorded carry relationship. Retreat still returns to the existing home hand pose.

The supplied anchor calibration is inconclusive. This mode exposes the estimated correspondence for testing; it does not establish grasp preservation or task success, and it retains the existing reset joint pose.

Command/transform check (headless):

```bash
python tests/tasks/check_mano_anchor_commands.py --headless \
  --task PickAndPlace_HRL-v0 \
  --anchors /home/yizhao/yi/D2H/datasets/anchor_calibration/full_calibration/anchors.json
```

The calibrated-anchor command checks pass, including placement retreat. A 30-step seed-0 frozen-policy smoke rollout ran with finite commands but remained in approach at source frame 320 without contact. This short run does not establish successful grasping or placement.
