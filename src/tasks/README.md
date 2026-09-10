# Task structure

Each task assembles its scene, observations, actions, commands, rewards, events,
and terminations in `env_cfg.py`. Task trajectories, assets, reset rules and
success conditions stay beside that task.

Shared components live in `common`:

- `env_cfg.py`: scene pieces, actions, tabletop events and placement rewards,
  fingertip sensor setup, contact physics and success visualization.
- `observations_cfg.py`: proprioception, point-cloud and frozen-hand observation groups.
- `mdps/`: reusable manager terms; `placement.py` owns shared grasp/release mechanics.
- `trajectory.py`: recording loading and validation.
- `rsl_rl_ppo_cfg_base.py`: shared PPO defaults.

Import only the pieces a task needs. For example, a task using the existing
Franka tabletop setup can declare its own object and command:

```python
from isaaclab.utils import configclass
from src.tasks.common.env_cfg import TabletopSceneCfg, HrlActionsCfg
from src.tasks.common.observations_cfg import LowLevelObsCfg

@configclass
class SceneCfg(TabletopSceneCfg):
    object = TASK_OBJECT_CFG  # the task's RigidObjectCfg

# Inside the task's ManagerBasedRLEnvCfg:
scene: SceneCfg = SceneCfg(num_envs=2, env_spacing=3)
actions: HrlActionsCfg = HrlActionsCfg()
# Add LowLevelObsCfg() to the task's observation groups when a frozen hand policy is used.
```

Use explicit overrides for differing observation terms or action settings.
Keep observation/action order, contact-filter order, control timing and reset
order intact: trained policies depend on these contracts. Common modules must
not import individual tasks. Named variants can reuse their parent task.

## Checks

Activate `env_isaaclab`. Before a structural change, capture all registered task
configs (including trainer settings and term ordering):

```sh
python tests/tasks/check_task_configs.py --headless --output /tmp/tasks-before.json
```

Afterwards, compare them and check a representative environment:

```sh
python tests/tasks/check_task_configs.py --headless --output /tmp/tasks-after.json --compare /tmp/tasks-before.json
python tests/tasks/check_task_runtime.py --headless --task Clean_Table_HRL-v0
python tests/tasks/check_pouring_runtime.py --headless
python tests/tasks/check_pick_and_place_runtime.py --headless
```

The snapshot checker accounts for the explicitly listed callable moves in this
refactor. It reports import/configuration errors separately; matching errors do
not establish runtime coverage for that task. Runtime scripts start AppLauncher
before importing simulation-dependent task code.
