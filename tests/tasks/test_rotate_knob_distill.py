from pathlib import Path

from src.policy.knob_interface import FRAME_DIM, HISTORY_DIM


ROOT = Path(__file__).resolve().parents[2]


def test_knob_distillation_contract_is_locked() -> None:
    env = (ROOT / "src/tasks/rotate_knob_distill/env_cfg.py").read_text()
    runner = (ROOT / "src/tasks/rotate_knob_distill/rsl_rl_distillation_cfg.py").read_text()
    registry = (ROOT / "src/tasks/__init__.py").read_text()

    assert FRAME_DIM == 35
    assert HISTORY_DIM == 105
    assert "history_length=3" in env
    assert "EMAJointPositionToLimitsActionCfg" in env
    assert "alpha=0.5" in env
    assert "rescale_to_limits=True" in env
    assert "dt=1.0 / 240.0" in env
    assert "self.decimation = 4" in env
    assert 'obs_groups = {"policy": ["policy"], "teacher": ["low_level"]}' in runner
    assert "student_hidden_dims=[512, 256, 128]" in runner
    assert "teacher_hidden_dims=[512, 256, 128]" in runner
    assert 'id="Rotate_Knob_Distill-v0"' in registry
