# Knob policy distillation

`model_14999.pt` cannot be deployed from its 155D observation directly. It uses simulator-only fingertip poses, object pose/velocity, gravity in the hand frame, goal-pose deltas, and separated object/external contact masks, magnitudes, and contact locations. Real hardware does not provide those quantities without an additional tracker and tactile/contact-estimation stack.

`Rotate_Knob_Distill-v0` keeps that exact 155D `low_level` group for the teacher and trains a student from hardware-observable state only. One student frame is:

```text
[normalized LEAP q (16), applied absolute target in radians (16),
 knob angle (1), knob velocity (1), signed target angle (1)] = 35
```

IsaacLab stores three frames oldest-to-newest, producing 105 inputs. The action contract is identical to `Rotate_Knob_HRL-v0`: 16 normalized absolute joint targets are clamped to `[-1, 1]`, unscaled to the LEAP limits, and filtered as `applied = 0.5 * target + 0.5 * previous_applied`. This is not a relative or cumulative-delta action.

The task uses the ARIA knob-handle geometry and hand base pose with D2H's checkpoint-compatible zero joint reset, a 15 second episode, 240 Hz physics, and 60 Hz control. Hand-to-knob pose is randomized by ±8 mm and ±0.08 rad. Each target is a signed 60–90° world-Z delta from the current knob angle; reaching it with wrapped angle error ≤0.02 rad and |velocity| ≤0.2 rad/s immediately samples the next target without resetting the episode.

## Train and export

The known teacher checksum is:

```text
e47003a9f3ccba98c0b7d36c7944900b4f4515b5f51f39feb42b0f1600e7f0d1
```

```bash
conda activate env_isaaclab
python scripts/rsl_rl/train.py \
  --task Rotate_Knob_Distill-v0 \
  --checkpoint /home/yizhao/yi/dex_reorient/model_14999.pt \
  --checkpoint_sha256 e47003a9f3ccba98c0b7d36c7944900b4f4515b5f51f39feb42b0f1600e7f0d1 \
  --headless
```

The native RSL-RL `DistillationRunner` trains a `[512, 256, 128]` ELU student with observation normalization. Use the existing play entry point on a distilled checkpoint to export `exported/policy.pt` and `policy.onnx`:

```bash
python scripts/rsl_rl/play.py --task Rotate_Knob_Distill_Play-v0 \
  --checkpoint logs/rsl_rl/rotate_knob_distill/<run>/model_300.pt --headless
```

For like-for-like inspection, add `--teacher-policy` to run the teacher from the same environment. Before hardware use, compare teacher and student over at least 1,000 fixed-seed nominal episodes and 1,000 randomized episodes; proceed only if student success is within five percentage points of teacher success.

## Hardware

Deployment is dry-run unless `--execute` is supplied. `--aria-source` must point to the existing `ARIA/source/ARIA` directory so the tested ARIA Dynamixel transport is reused. The script validates finite 16D policy output, enforces the physical joint limits, limits target slew per cycle, starts from the checkpoint-compatible zero joint pose, runs at 60 Hz, unwraps the knob encoder across ±pi, and disables hand torque on exit.

```bash
python scripts/deployment/deploy_knob_student.py \
  logs/rsl_rl/rotate_knob_distill/<run>/exported/policy.pt \
  --target 1.57

python scripts/deployment/deploy_knob_student.py \
  logs/rsl_rl/rotate_knob_distill/<run>/exported/policy.pt \
  --target 1.57 --execute \
  --aria-source /home/yizhao/yi/aria_isaaclab/IsaacLab/source/ARIA \
  --hand-port /dev/ttyUSB0 --knob-port /dev/ttyUSB1
```

The original reorientation teacher ran at 30 Hz in its saved play configuration, while this deployment contract is 60 Hz to match the current knob target. Treat a large 30/60 Hz behavior gap as a deployment blocker and retrain at the chosen hardware frequency rather than compensating silently in the hardware loop.
