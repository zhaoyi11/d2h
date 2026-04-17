import gymnasium as gym
from pathlib import Path

# Patch IsaacLab-registered environments with flash_sac_cfg_entry_point.
_allegro_spec = gym.spec("Isaac-Repose-Cube-Allegro-Direct-v0")
_allegro_spec.kwargs["flash_sac_cfg_entry_point"] = (
    "src.tasks.allegro_hand.flash_sac_cfg:AllegroHandFlashSacCfg"
)

# Register Gym environments.

# Reorient Environment
gym.register(
    id="Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reorient.env_cfg:LeapObjectEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.reorient.rsl_rl_ppo_cfg:LeapObjectRslRlPpoCfg",
        "flash_sac_cfg_entry_point": "src.tasks.reorient.flash_sac_cfg:LeapObjectFlashSacCfg",
    },
)

# Pick AnyRotate Environment
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

# Pick Insert Environment
gym.register(
    id="Pick_Insert-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_insert.env_cfg:DexsuiteFrankaLeapInsertEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_insert.rsl_rl_ppo_cfg:PickInsertRslRlPpoCfg",
    },
)

# Pick Screw Environment
gym.register(
    id="Pick_Screw-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_screw.env_cfg:DexsuiteFrankaLeapScrewEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_screw.rsl_rl_ppo_cfg:PickScrewRslRlPpoCfg",
    },
)

# Clean Table Environment
gym.register(
    id="Clean_Table-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clean_table.env_cfg:DexsuiteFrankaLeapCleanTableEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.clean_table.rsl_rl_ppo_cfg:CleanTableRslRlPpoCfg",
    },
)

# Cupcake on Plate Environment
gym.register(
    id="Cupcake_on_Plate-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cupcake_on_plate.env_cfg:DexsuiteFrankaLeapCupcakeOnPlateEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.cupcake_on_plate.rsl_rl_ppo_cfg:CupcakeOnPlateRslRlPpoCfg",
    },
)

# Threading Environment
gym.register(
    id="Threading-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.threading.env_cfg:DexsuiteFrankaLeapThreadingEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.threading.rsl_rl_ppo_cfg:ThreadingRslRlPpoCfg",
    },
)
