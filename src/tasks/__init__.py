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
    id="Pick_AnyRotate_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_anyrotate.env_cfg:DexsuiteFrankaLeapAnyRotateHrlEnvCfg",
    },
)


# Pick Insert Environment

gym.register(
    id="Pick_Insert_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_insert.env_cfg:DexsuiteFrankaLeapInsertHrlEnvCfg",
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
    id="Unscrew_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.unscrew.env_cfg:DexsuiteFrankaLeapUnscrewHrlEnvCfg",
    },
)

# Clean Table Environment

gym.register(
    id="Clean_Table_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clean_table.env_cfg:DexsuiteFrankaLeapCleanTableHrlEnvCfg",
    },
)

gym.register(
    id="Pouring_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pouring.env_cfg:PouringEnvCfg",
    },
)

gym.register(
    id="PickAndPlace_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pick_and_place.env_cfg:PickAndPlaceEnvCfg",
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

# Rotate Knob Environment

gym.register(
    id="Rotate_Knob_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotate_knob.env_cfg:DexsuiteFrankaLeapRotateObjectHrlEnvCfg",
    },
)

gym.register(
    id="Rotate_Knob_Student_HRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotate_knob_student.env_cfg:DexsuiteFrankaLeapRotateObjectStudentHrlEnvCfg",
    },
)

# Knob policy distillation environment
gym.register(
    id="Rotate_Knob_Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotate_knob_distill.env_cfg:RotateKnobDistillEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "src.tasks.rotate_knob_distill.rsl_rl_distillation_cfg:"
            "RotateKnobDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Rotate_Knob_Distill_Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rotate_knob_distill.env_cfg:RotateKnobDistillEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "src.tasks.rotate_knob_distill.rsl_rl_distillation_cfg:"
            "RotateKnobDistillationRunnerCfg"
        ),
    },
)

