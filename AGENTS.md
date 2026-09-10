IsaacLab rule: When a task touches IsaacLab code or concepts, invoke the `isaaclab-api-context` skill before answering or editing. Trigger on `isaaclab*` imports, IsaacLab env/task configs, or concepts such as `AppLauncher`, `SimulationContext`, `InteractiveScene`, `ManagerBasedEnv`, `ManagerBasedRLEnv`, `DirectRLEnv`, managers, assets, sensors, markers, actuators, or gym task registration using IsaacLab entry points.

D2H IsaacLab notes:
- Use `conda activate env_isaaclab` for the running environment.
- Prefer local task logic under `src/env/tasks/...` and keep IsaacLab-specific behavior near env configs and manager terms.
- Preserve the existing gym registration pattern in `_src/env/__init__.py`, including string entry points such as `isaaclab.envs:ManagerBasedRLEnv` and `env_cfg_entry_point="module.path:ConfigClass"`, unless the task explicitly requires a different shape.
- In standalone tooling or evaluation scripts, preserve the `AppLauncher` startup flow before runtime-dependent IsaacLab usage.
- In command generators or related manager terms, be explicit about base-frame versus world-frame transforms and verify tensor shapes against local code before changing behavior.
