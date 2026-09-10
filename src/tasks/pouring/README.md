# Pouring trajectory

```bash
conda activate env_isaaclab
python scripts/instant_dexterity.py --task Pouring_HRL-v0 --num_envs 1 --steps 3600 \
  --low_level_checkpoint /home/yizhao/yi/dex_reorient/model_14999.pt
```

Add `--headless --seed 0` for repeatable validation. The task uses
`../pick_and_place/pouring_mano_isaac_trajectory.npz` (1,300 frames, 200 Hz)
and `../pick_and_place/bottle/bottle.obj` at scale 0.2. The NPZ and mesh are unchanged.
Object positions and skeleton points receive the same translation, placing the
initial object center at `(0.55, 0.10, 0.40)` while preserving recorded rotations.

The robot starts close to the bottle in a policy-settled grasp. The recording's
first human pose is still approaching, so initialization roughly matches a later
grasp shape instead of reproducing that distant approach pose. The bottle itself
starts at the recording's first orientation and follows the entire object path.

The reset values in `env_cfg.py` were selected by offline policy settling and then
tested from clean resets with the object free. No settling or object support runs
inside task resets. To refine the configured grasp offline:

```bash
python src/tasks/pouring/calibrate_initial_pose.py --headless
```

This tool temporarily supports the object at frame 0 for 90 control steps and
prints candidate joint angles and the hand-to-anchor transform. It uses the same
frozen checkpoint as the launcher. Validate any new candidate with the runtime
check below before copying the printed values into `env_cfg.py`; calibration
contact history alone does not prove the pose works from a clean reset.

The anchor stays at the recorded object center with a local +90-degree X rotation.
Its full transform relative to the initialized hand is calibrated at frame 0, so the
initial MPC goal matches the starting hand pose. Object/goal quaternion errors
use a unique sign to represent matching orientations as positive identity.

Frame 0 is duplicated for three consecutive thumb-plus-finger contact checks
before manipulation. There is no scripted open-hand phase: the existing distance
gate and frozen RL policy take over from the initialized grasp immediately. MPC controls
the arm. Waypoints sample every seven frames and include frame 1299; advancement
requires object error below **5 cm / 0.60 rad** and arm error below **5 cm / 0.40 rad**.
The final goal requires five consecutive achieved steps; episodes last 120 seconds.

One MPC target is applied each 30 Hz control step, with a 0.1333 s optimizer
interval and four interpolation steps. Reset restores joints, hand smoothing,
and MPC state, including resets inside the runner's inference-mode loop.

The object initially floats. Hand contact above **0.05 N** enables gravity
`(0, 0, -1.81)` until reset. Contact history covers four physics substeps, and
stale contacts cannot reactivate gravity immediately after reset.

## Validation

```bash
python -m pytest -q tests/tasks/test_pouring.py
python tests/tasks/check_pouring_runtime.py --headless
```

The runtime check covers rough skeleton alignment, initial contacts, a 60-step
rollout of the actual frozen policy, frame transforms, joint limits, full/subset
resets after motion, inference-mode reset, controller interfaces, grasp/success
gates, and contact-latched gravity. The startup regression requires object speed
below 0.35 m/s during the first six control steps, progress beyond frame 0, and
object-in-hand target error below 6 cm after 60 steps.

A four-environment, 600-step launcher run with seed 1 keeps all four hand
policies engaged. The logged first environment advances to recording frame 868
with bottle contact maintained. This validates the new recording and initial
grasp, but does not establish completion of all 1,300 frames.

The policy consumes 155 observations and produces 16 hand actions. No liquid
simulation or policy training is added. `clean_table`, shared controllers, and
`instant_dexterity.py` are unchanged by this initialization update.

Replay the same recording without robot control:

```bash
python src/tasks/pouring/vis_traj_isaaclab.py --loop
```
