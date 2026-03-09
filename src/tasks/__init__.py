import gymnasium as gym
from pathlib import Path

# Register Gym environments.

# Pick_AnyRotate Environment
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

