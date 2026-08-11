# Clean Table OmniReset Frozen Viewer Design

## Goal

Add a standalone viewer for inspecting one randomly sampled reset state from
`Clean_Table_OmniReset-v0` without sending actions or advancing physics.

## Design

Create `scripts/visualize_clean_table_omnireset.py` following the repository's
IsaacLab launch order: parse arguments, construct `AppLauncher`, then import
runtime-dependent IsaacLab and task modules.

The script will:

1. Parse `--num_envs` (default `1`), optional `--reset_dataset_dir`, and standard
   `AppLauncher` arguments.
2. Load the registered `Clean_Table_OmniReset-v0` configuration.
3. Apply the optional dataset-directory override before environment creation.
4. Create the environment and call `env.reset()` once, allowing the existing
   reset event to sample complete and partial frames uniformly.
5. Pause the Omniverse timeline immediately after reset.
6. Keep the application and viewport responsive with application updates only.
   The script must never call `env.step()` or send an action.
7. Close the environment and simulator cleanly when the application exits.

The RL task and its EMA hand controller remain unchanged.

## Verification

- Add a source-level test confirming the script resets and pauses without an
  environment step.
- Run the focused test.
- Launch the viewer headlessly as a startup smoke check and confirm it reaches
  the frozen state before terminating the check.
