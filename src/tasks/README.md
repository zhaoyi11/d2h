# Task structure

Each task assembles its scene, observations, actions, commands, rewards, events,
and terminations in `env_cfg.py`. Task trajectories, assets, reset rules and
success conditions stay beside that task.

`clean_table`, `pick_and_place`, `pick_anyrotate`, `pick_insert`, `pouring`,
`rotate_knob`, `unscrew`, and `rotate_knob_student` expose only HRL execution.
They use cuRobo arm control and a frozen hand policy, with `rewards = None`
and `curriculum = None`. Steps return zero rewards; command progress and
success/termination checks remain active. Configured reset randomization remains,
but gravity and observation noise no longer adapt across episodes.

The two OmniReset tasks, `reorient`, and `rotate_knob_distill` retain their
training configurations, rewards and curricula. Their Play registrations remain.
The removed flat tasks and HRL PPO trainer entry points have no compatibility aliases.

Shared components live in `common`:

- `env_cfg.py`: scene pieces, actions, tabletop events,
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
rewards = None
curriculum = None
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

Use `--allow-hrl-cleanup` with `--compare` only when comparing against a snapshot
from before the HRL-only cleanup. It permits exactly the eight removed flat
registrations and the removed HRL reward/curriculum/trainer settings. Ordinary
comparisons remain strict. The checker accounts for explicitly listed callable moves.
Configuration errors fail the check. Runtime scripts start AppLauncher before
importing simulation-dependent task code.

Pass `--checkpoint /path/to/model.pt` to `check_task_runtime.py` to drive a
compatible frozen hand policy instead of zero actions. HRL checks also require
zero rewards and empty reward/curriculum managers.
