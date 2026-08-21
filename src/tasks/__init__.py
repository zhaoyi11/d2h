import gymnasium as gym
from pathlib import Path

# Patch IsaacLab-registered environments with flash_sac_cfg_entry_point.
try:
    _allegro_spec = gym.spec("Isaac-Repose-Cube-Allegro-Direct-v0")
except gym.error.Error:
    _allegro_spec = None
if _allegro_spec is not None:
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

gym.register(
    id="Reorient_Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reorient.env_cfg:LeapObjectEnvCfg_PLAY",
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
    id="Pick_AnyRotate_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_anyrotate.env_cfg:DexsuiteFrankaLeapAnyRotateHrlEnvCfg",
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

gym.register(
    id="Pick_Insert_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_insert.env_cfg:DexsuiteFrankaLeapInsertHrlEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.pick_insert.rsl_rl_ppo_cfg:PickInsertRslRlPpoCfg",
    },
)

gym.register(
    id="Pick_Insert_OmniReset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.pick_insert_omnireset.env_cfg:"
            "DexsuiteFrankaLeapPickInsertOmniResetEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "src.tasks.pick_insert_omnireset.rsl_rl_ppo_cfg:"
            "PickInsertOmniResetRslRlPpoCfg"
        ),
    },
)

# Unscrew Environment
gym.register(
    id="Unscrew-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unscrew.env_cfg:DexsuiteFrankaLeapUnscrewEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.unscrew.rsl_rl_ppo_cfg:UnscrewRslRlPpoCfg",
    },
)

gym.register(
    id="Unscrew_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unscrew.env_cfg:DexsuiteFrankaLeapUnscrewHrlEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.unscrew.rsl_rl_ppo_cfg:UnscrewRslRlPpoCfg",
    },
)

gym.register(
    id="Unscrew_OmniReset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.unscrew_omnireset.env_cfg:"
            "DexsuiteFrankaLeapUnscrewOmniResetEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "src.tasks.unscrew_omnireset.rsl_rl_ppo_cfg:"
            "UnscrewOmniResetRslRlPpoCfg"
        ),
    },
)

gym.register(
    id="Unscrew_OmniReset_Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.unscrew_omnireset.env_cfg:"
            "DexsuiteFrankaLeapUnscrewOmniResetEnvCfg_PLAY"
        ),
        "rsl_rl_cfg_entry_point": (
            "src.tasks.unscrew_omnireset.rsl_rl_ppo_cfg:"
            "UnscrewOmniResetRslRlPpoCfg"
        ),
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

gym.register(
    id="Clean_Table_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clean_table.env_cfg:DexsuiteFrankaLeapCleanTableHrlEnvCfg",
        "rsl_rl_cfg_entry_point": "src.tasks.clean_table.rsl_rl_ppo_cfg:CleanTableRslRlPpoCfg",
    },
)

gym.register(
    id="Clean_Table_OmniReset-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.clean_table_omnireset.env_cfg:"
            "DexsuiteFrankaLeapCleanTableOmniResetEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "src.tasks.clean_table_omnireset.rsl_rl_ppo_cfg:"
            "CleanTableOmniResetRslRlPpoCfg"
        ),
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

gym.register(
    id="Cupcake_on_Plate_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cupcake_on_plate.env_cfg:DexsuiteFrankaLeapCupcakeOnPlateHrlEnvCfg",
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
