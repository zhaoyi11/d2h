import gymnasium as gym
from pathlib import Path

##
# Register Gym environments.
##

# Path to rl-games config, relative to this file.
_RL_GAMES_CFG_PATH = str(
    Path(__file__).resolve().parent.parent / "config" / "rl_games_ppo_cfg.yaml"
)
# _RSL_RL_CFG_PATH = str(
#     Path(__file__).resolve().parent.parent / "config" / "rsl_rl_ppo_cfg.py"
# )

# Twist Environment
gym.register(
    id="Twist-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tasks.twist.twist_env_cfg:TwistEnvCfg", 
        "rsl_rl_cfg_entry_point": "src.config.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
    },
)

gym.register(
    id="AnyGrasp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # "env_cfg_entry_point": f"{__name__}.tasks.anygrasp.anygrasp_env_cfg:LeapObjectEnvCfg",
        "env_cfg_entry_point": f"{__name__}.tasks.anygrasp.reorient_env_cfg:LeapObjectEnvCfg",
        "rl_games_cfg_entry_point": _RL_GAMES_CFG_PATH,
        # Use explicit module path for RSL-RL config to avoid incorrect
        # nesting under `src.env` (the config module lives in `src.config`).
        "rsl_rl_cfg_entry_point": "src.config.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
    },
)
# # Backwards-compatible alias without the version suffix, since some training
# # scripts / configs may refer to the task as just "AnyGrasp".
# gym.register(
#     id="AnyGrasp",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.tasks.anygrasp.reorient_env_cfg:LeapObjectEnvCfg",
#         "rl_games_cfg_entry_point": _RL_GAMES_CFG_PATH,
#         "rsl_rl_cfg_entry_point": "src.config.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
#     },
# )

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
