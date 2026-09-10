import gymnasium as gym


def test_hrl_only_task_registrations_preserve_protected_trainers() -> None:
    import src.tasks  # noqa: F401

    hrl_entries = {
        "Clean_Table_HRL-v0": "clean_table.env_cfg:DexsuiteFrankaLeapCleanTableHrlEnvCfg",
        "PickAndPlace_HRL-v0": "pick_and_place.env_cfg:PickAndPlaceEnvCfg",
        "Pick_AnyRotate_HRL-v0": "pick_anyrotate.env_cfg:DexsuiteFrankaLeapAnyRotateHrlEnvCfg",
        "Pick_Insert_HRL-v0": "pick_insert.env_cfg:DexsuiteFrankaLeapInsertHrlEnvCfg",
        "Pouring_HRL-v0": "pouring.env_cfg:PouringEnvCfg",
        "Rotate_Knob_HRL-v0": "rotate_knob.env_cfg:DexsuiteFrankaLeapRotateObjectHrlEnvCfg",
        "Unscrew_HRL-v0": "unscrew.env_cfg:DexsuiteFrankaLeapUnscrewHrlEnvCfg",
        "Rotate_Knob_Student_HRL-v0": "rotate_knob_student.env_cfg:DexsuiteFrankaLeapRotateObjectStudentHrlEnvCfg",
    }
    for task_id, entry in hrl_entries.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv", task_id
        assert spec.kwargs == {"env_cfg_entry_point": f"src.tasks.{entry}"}, task_id

    removed_ids = {
        "Pick_AnyRotate-v0", "Pick_AnyRotate_Play-v0", "Pick_Lift-v0", "Pick_Lift_Play-v0",
        "Pick_Insert-v0", "Unscrew-v0", "Clean_Table-v0", "Rotate_Knob-v0",
    }
    assert not removed_ids.intersection(gym.registry)

    protected_trainers = {
        "Clean_Table_OmniReset-v0": "clean_table_omnireset.rsl_rl_ppo_cfg:CleanTableOmniResetRslRlPpoCfg",
        "Pick_Insert_OmniReset-v0": "pick_insert_omnireset.rsl_rl_ppo_cfg:PickInsertOmniResetRslRlPpoCfg",
        "Reorient-v0": "reorient.rsl_rl_ppo_cfg:LeapObjectRslRlPpoCfg",
        "Reorient_Play-v0": "reorient.rsl_rl_ppo_cfg:LeapObjectRslRlPpoCfg",
        "Rotate_Knob_Distill-v0": "rotate_knob_distill.rsl_rl_distillation_cfg:RotateKnobDistillationRunnerCfg",
        "Rotate_Knob_Distill_Play-v0": "rotate_knob_distill.rsl_rl_distillation_cfg:RotateKnobDistillationRunnerCfg",
    }
    for task_id, entry in protected_trainers.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == "isaaclab.envs:ManagerBasedRLEnv", task_id
        assert spec.kwargs["rsl_rl_cfg_entry_point"] == f"src.tasks.{entry}", task_id
    for task_id in ("Reorient-v0", "Reorient_Play-v0"):
        assert gym.spec(task_id).kwargs["flash_sac_cfg_entry_point"] == (
            "src.tasks.reorient.flash_sac_cfg:LeapObjectFlashSacCfg"
        )
