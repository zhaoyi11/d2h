import gymnasium as gym
from pathlib import Path

##
# Register Gym environments.
##

# Path to rl-games config, relative to this file.
_RL_GAMES_CFG_PATH = str(
    Path(__file__).resolve().parent.parent / "config" / "rl_games_ppo_cfg.yaml"
)

gym.register(
    id="AnyGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.anygrasp.anygrasp_env_cfg:AllegroCubeEnvCfg",
        "rl_games_cfg_entry_point": _RL_GAMES_CFG_PATH,
    },
)

# Dexsuite Reorient Environment
gym.register(
    id="DexsuiteReorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.dexsuite.dexsuite_env_cfg:DexsuiteKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": _RL_GAMES_CFG_PATH,
    },
)
# Dexsuite Lift Environment
gym.register(
    id="DexsuiteLift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.dexsuite.dexsuite_env_cfg:DexsuiteKukaAllegroLiftEnvCfg",
        "rl_games_cfg_entry_point": _RL_GAMES_CFG_PATH,
    },
)