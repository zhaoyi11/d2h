import gymnasium as gym

# from . import agents

##
# Register Gym environments.
##


gym.register(
    id="AnyGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.anygrasp.allegro_env_cfg:AllegroCubeEnvCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": "/home/yizhao/yi/D2H/src/config/rl_games_ppo_cfg.yaml",
    },
)