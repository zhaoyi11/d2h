from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_rotate_object_student_task_contract() -> None:
    env = (ROOT / "src/tasks/rotate_knob_student/env_cfg.py").read_text()
    registry = (ROOT / "src/tasks/__init__.py").read_text()
    student_obs = (ROOT / "src/tasks/rotate_knob_distill/mdps/task_mdps.py").read_text()

    assert "from src.tasks.rotate_knob.env_cfg import" in env
    assert "DexsuiteFrankaLeapRotateObjectStudentHrlEnvCfg" in env
    assert "DexsuiteFrankaLeapRotateObjectHrlEnvCfg" in env
    assert "RotateKnobDistillObservationsCfg.StudentCfg()" in env
    assert "self.commands.object_pose.yaw_delta_range = (math.pi / 3.0, math.pi / 3.0)" in env
    assert 'id="Rotate_Knob_Student_HRL-v0"' in registry
    assert "rotate_knob_student.env_cfg:DexsuiteFrankaLeapRotateObjectStudentHrlEnvCfg" in registry
    assert "rotate_knob_student.rsl_rl_ppo_cfg" not in registry
    assert not (ROOT / "src/tasks/rotate_knob_student/rsl_rl_ppo_cfg.py").exists()
    assert "quat_mul(robot.data.root_quat_w, command.pose_command_b[:, 3:])" in student_obs
