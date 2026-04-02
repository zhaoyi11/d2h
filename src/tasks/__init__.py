import gymnasium as gym
from pathlib import Path

# Register Gym environments.

# Reorient Environment
gym.register(
    id="Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reorient.env_cfg:LeapObjectEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.reorient.rsl_rl_ppo_cfg:LeapObjectRslRlPpoCfg",
    },
)

# Grasp Environments
gym.register(
    id="Grasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp.env_cfg:LeapObjectEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.grasp.rsl_rl_ppo_cfg:LeapObjectRslRlPpoCfg",
    },
)
gym.register(
    id="GraspCollect-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp.env_cfg:LeapObjectCollectGraspEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.grasp.rsl_rl_ppo_cfg:LeapObjectRslRlPpoCfg",
    },
)

gym.register(
    id="GraspReplay-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grasp.env_cfg:LeapObjectReplayEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.grasp.rsl_rl_ppo_cfg:LeapObjectRslRlPpoCfg",
    },
)

# Pick Lift / AnyRotate Environment
gym.register(
    id="Pick_AnyRotate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_anyrotate.env_cfg:DexsuiteFrankaLeapReorientEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_anyrotate.rsl_rl_ppo_cfg:PickAnyRotateRslRlPpoCfg",
    },
)

gym.register(
    id="Pick_Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_anyrotate.env_cfg:DexsuiteFrankaLeapLiftEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_anyrotate.rsl_rl_ppo_cfg:PickAnyRotateRslRlPpoCfg",
    },
)

gym.register(
    id="Pick_AnyRotate_Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_anyrotate.env_cfg:DexsuiteFrankaLeapReorientEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_anyrotate.rsl_rl_ppo_cfg:PickAnyRotateRslRlPpoCfg",
    },
)

gym.register(
    id="Pick_Lift_Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_anyrotate.env_cfg:DexsuiteFrankaLeapLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_anyrotate.rsl_rl_ppo_cfg:PickAnyRotateRslRlPpoCfg",
    },
)

## Pick Insert Environment
gym.register(
    id="Pick_Insert-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_insert.env_cfg:DexsuiteFrankaLeapInsertEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_insert.rsl_rl_ppo_cfg:PickInsertRslRlPpoCfg",
    },
)