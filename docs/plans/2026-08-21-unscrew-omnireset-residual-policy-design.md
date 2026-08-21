# Unscrew OmniReset Residual Policy Design

**Date:** 2026-08-21

**Status:** Validated

## Goal

Train `Unscrew_OmniReset-v0` with an RSL-RL PPO policy that reuses the existing frozen 16-DOF hand policy while learning:

- seven normalized relative Franka joint-position actions,
- a sixteen-dimensional correction to the frozen hand action, and
- a new value function over the expanded observation.

The frozen hand policy must remain unchanged throughout training. A saved residual-policy checkpoint must contain everything needed for resume and playback, including the frozen actor weights and its observation-normalization state.

## Existing Environment Contract

`src/tasks/unscrew_omnireset/env_cfg.py` already provides the required environment interface:

- `low_level`: the current-step 155-D observation expected by the frozen reorientation policy;
- `residual`: a 55-D observation containing arm state and task context;
- `arm_action`: seven relative Franka joint-position commands with scale `0.1`; and
- `hand_action`: sixteen normalized LEAP-hand commands processed by the existing EMA action term.

The RSL-RL config currently maps both `low_level` and `residual` into the actor and critic observation sets. Therefore, the expanded observation is 210-D and the environment action is ordered as seven arm values followed by sixteen hand values.

No new Gym task or environment action configuration is required. This design modifies the policy selected by the existing `Unscrew_OmniReset-v0` trainer entry point.

## Chosen Architecture

Add a project-local `ResidualActorCritic` implementing the interface expected by RSL-RL PPO.

```text
obs["low_level"] (155)
        |
        v
frozen checkpoint normalizer -> frozen actor -> frozen hand action (16)

concat(obs["low_level"], obs["residual"]) (210)
        |
        +-> fresh residual normalizer -> fresh residual MLP -> raw delta (23)
        |
        +-> fresh critic normalizer   -> fresh critic MLP   -> value (1)
```

The actor mean is composed in normalized action space:

```python
base_action = torch.cat((torch.zeros_like(raw_delta[..., :7]), frozen_hand_action), dim=-1)
arm_delta = arm_residual_scale * torch.tanh(raw_delta[..., :7])
hand_delta = hand_residual_scale * torch.tanh(raw_delta[..., 7:])
mean_action = base_action + torch.cat((arm_delta, hand_delta), dim=-1)
```

Start with `arm_residual_scale=1.0` and `hand_residual_scale=0.1`. The residual MLP's final layer is zero-initialized. Its initial mean therefore commands zero arm motion and reproduces the frozen hand action exactly.

The frozen actor is deterministic. PPO owns a single Gaussian distribution centered on the composed 23-D mean so sampled actions and their log probabilities refer to the same action seen by the environment. Arm and hand exploration standard deviations are initialized separately, with smaller hand exploration to avoid immediately destroying the pretrained behavior. The trainer clips final normalized actions to `[-1, 1]` before environment action processing.

The critic is entirely new. It initially consumes the same 210-D expanded observation as the residual actor because `Unscrew_OmniReset-v0` does not currently expose a separate privileged observation group. A privileged group may be added later only when concrete privileged signals are specified.

## Parameter and Mode Ownership

The frozen actor and its saved mean and standard deviation are registered as child-module parameters or buffers so they are included in the composite policy state dictionary. All frozen parameters use `requires_grad=False`.

Calling `train()` on the composite policy must not put the frozen branch into a trainable or state-updating mode. The residual actor and critic follow the runner's train/eval mode normally. Updating the new residual observation normalizer must never modify the frozen checkpoint normalizer.

The PPO optimizer may receive `ResidualActorCritic.parameters()` normally: tensors with `requires_grad=False` will not be optimized. Tests must nevertheless verify both absence of gradients and bitwise stability of every frozen tensor after an optimizer update.

## Project Structure and Code Changes

### New policy package

Create:

```text
src/policy/rsl_rl/__init__.py
src/policy/rsl_rl/residual_actor_critic.py
src/policy/rsl_rl/registry.py
```

`residual_actor_critic.py` contains `ResidualActorCritic` and its IsaacLab `@configclass` policy configuration. The configuration contains only required architecture and checkpoint fields:

- `class_name = "ResidualActorCritic"`;
- `frozen_checkpoint`;
- `frozen_obs_group = "low_level"`;
- `frozen_obs_dim = 155`;
- `frozen_action_dim = 16`;
- residual and critic hidden dimensions;
- arm and hand residual scales; and
- separate initial arm and hand exploration standard deviations.

The arm dimension is derived as `num_actions - frozen_action_dim` and validated as seven rather than configured independently.

`registry.py` provides one explicit registration function. The installed RSL-RL runner resolves `policy.class_name` with `eval()` against names in `rsl_rl.runners.on_policy_runner`, so the function exposes `ResidualActorCritic` there. This keeps the compatibility workaround in one place and avoids editing installed RSL-RL code.

### Task trainer configuration

Change `src/tasks/unscrew_omnireset/rsl_rl_ppo_cfg.py` to:

- replace the inherited built-in `RslRlPpoActorCriticCfg` with the residual policy config;
- retain `obs_groups = {"policy": ["low_level", "residual"], "critic": ["low_level", "residual"]}`; and
- set `clip_actions = 1.0`.

Other tasks continue using the common built-in actor-critic configuration.

### Training and playback

Change both `scripts/rsl_rl/train.py` and `scripts/rsl_rl/play.py` to register project-local RSL-RL classes before constructing `OnPolicyRunner`. Do not change their existing `AppLauncher` startup ordering.

No change is required in `src/tasks/__init__.py`: the existing Gym registration continues pointing to `UnscrewOmniResetRslRlPpoCfg`.

## Checkpoint Lifecycle

For a fresh training run, the policy must load the configured frozen checkpoint before learning begins. A missing or incompatible source checkpoint is a fatal configuration error.

The composite RSL-RL checkpoint includes:

- frozen actor weights;
- frozen normalization buffers;
- residual actor weights and normalization state;
- critic weights and normalization state;
- action-distribution parameters; and
- the normal PPO optimizer and runner state.

For resume and playback, the model structure can be constructed from configured dimensions without access to the source checkpoint. `runner.load()` then restores the frozen branch from the composite checkpoint before any environment action is produced. The policy tracks whether the frozen branch was initialized from the source checkpoint or restored from a composite state dictionary.

A fresh run must reject an uninitialized frozen branch. Resume and playback must reject a composite checkpoint that does not restore every required frozen key. Partial frozen-policy loads are not allowed.

## Validation and Errors

Construction fails with a clear message when:

- the `low_level` group is absent or not 155-D;
- the concatenated actor observation is not 210-D;
- the environment action count is not 23;
- the frozen actor output is not 16-D;
- the checkpoint lacks compatible actor weights or normalization statistics; or
- the implied action split is not `[7 arm, 16 hand]`.

Inference and training must use the same observation routing, normalization, residual scaling, action ordering, and clipping behavior.

## Verification Plan

Add CPU unit tests that use a small synthetic frozen RSL-RL checkpoint and verify:

1. A zero residual mean produces `[zeros(7), frozen_hand_action]` exactly.
2. The frozen actor receives only `low_level`; the residual actor and critic receive the full 210-D input.
3. Actor actions are 23-D, critic values have shape `(num_envs, 1)`, and log probability and entropy have one value per environment.
4. Only the residual actor, critic, new normalizers, and distribution parameters receive gradients.
5. Every frozen parameter and buffer remains bitwise unchanged after an optimizer step.
6. Missing observation groups and all dimension/order mismatches fail explicitly.
7. Project-local registration lets the installed `OnPolicyRunner` resolve `ResidualActorCritic`.
8. A composite checkpoint restores identical inference output when the original frozen checkpoint is unavailable.

Retain the existing AST/config tests for the 155-D `low_level`, 55-D `residual`, and 7+16 action contracts. Add or extend a task config test to assert the residual policy class and `clip_actions=1.0`.

Finally, run an IsaacLab smoke test in `env_isaaclab` with a small number of `Unscrew_OmniReset-v0` environments. Verify the 155+55 observation groups, 23-D action space, one environment step, and one short PPO update without modifying the frozen tensors.

## Non-goals

- Do not change the frozen hand policy architecture or train its parameters.
- Do not replace PPO or modify installed RSL-RL/IsaacLab packages.
- Do not create a new Gym task ID.
- Do not add speculative privileged observations.
- Do not change other tasks to use the residual policy.
