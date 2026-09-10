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

The six stages are approach, confirm grasp, recorded carry, deposit, release,
and retreat. Grasping requires 15 consecutive contact steps (0.5 seconds),
with a 90-step retry timeout, to seat the pig before lifting. Carry frames
320–511 run from the recording's pre-lift minimum to
its height peak; every seventh frame and both endpoints are included. The
path is rotated about world Z and scaled in XY to reach the box. Height is
scaled separately to reach 25 cm above the box origin. Relative object
rotations start from the pig's live settled orientation. All recorded poses
use XYZ/WXYZ; world targets are converted into each robot's root frame.
The wrist uses clean-table's grasp anchor while the frozen policy receives the
recorded object orientation goals.

Ten deposit waypoints lower the pig to box-local `(0, 0, 0.10)`, retaining its
final carry orientation. Clean-table's hand gate opens the fingers for release
and the arm retreats home. Success requires the object center inside the box,
the arm at its retreat goal, the hand clear, and object speed below 0.05 m/s
for five consecutive control steps. Pre-release drops use clean-table's grasp
recovery. A fall below the tabletop ends the episode and resets the scene.
This reuses clean-table's center-based containment check.

Physics runs at 120 Hz and control at 30 Hz, with a 120-second timeout.
Waypoints advance when tracking/contact gates pass, rather than at the
recording's original 200 Hz. Object pose tolerances are 5 cm and 1.0 rad
(about 57 degrees) for all six stages. The hand-base tracking tolerances,
contact confirmation and box-placement success checks retain their existing
settings. `pick_and_place_frame` reports the source frame
during approach/grasp/carry and `-1` during synthetic deposit/release/retreat;
`pick_and_place_progress` covers all 42 targets. Existing clean-table log lines
report grasp and placement state.

Task-local calibration lives in `PickAndPlaceTrajectoryCommandCfg` and
`PickAndPlaceEnvCfg`: the carry interval/stride, grasp anchor, tracking/contact
tolerances and box offsets can be adjusted without changing other tasks.

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
an automatic timeout reset. The shared pouring reset helper temporarily enables
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
