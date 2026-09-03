import math

import torch

from src.policy.knob_interface import (
    ACTION_ALPHA,
    FRAME_DIM,
    HISTORY_DIM,
    REAL_TO_SIM,
    SIM_TO_REAL,
    append_history,
    build_aria_frame,
    ema_absolute_targets,
    initialize_history,
    scale_absolute_actions,
    unwrap_angle,
)


def test_absolute_action_scaling_and_ema() -> None:
    lower = torch.tensor([[-2.0, 0.0]])
    upper = torch.tensor([[2.0, 4.0]])
    scaled = scale_absolute_actions(torch.tensor([[-1.0, 1.0]]), lower, upper)
    torch.testing.assert_close(scaled, torch.tensor([[-2.0, 4.0]]))
    torch.testing.assert_close(
        ema_absolute_targets(scaled, torch.zeros_like(scaled)),
        ACTION_ALPHA * scaled,
    )


def test_frame_and_history_are_frame_major_oldest_to_newest() -> None:
    joint_pos = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    target = joint_pos + 20.0
    lower = torch.zeros_like(joint_pos)
    upper = torch.full_like(joint_pos, 15.0)
    frame = build_aria_frame(
        joint_pos,
        target,
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([3.0]),
        lower,
        upper,
    )
    assert frame.shape == (1, FRAME_DIM)
    torch.testing.assert_close(frame[:, 16:32], target)
    torch.testing.assert_close(frame[:, 32:], torch.tensor([[1.0, 2.0, 3.0]]))

    history = initialize_history(frame)
    assert history.shape == (1, HISTORY_DIM)
    torch.testing.assert_close(history[:, :FRAME_DIM], frame)
    next_frame = frame + 100.0
    shifted = append_history(history, next_frame)
    torch.testing.assert_close(shifted[:, : 2 * FRAME_DIM], history[:, FRAME_DIM:])
    torch.testing.assert_close(shifted[:, -FRAME_DIM:], next_frame)


def test_joint_permutations_and_angle_unwrap() -> None:
    values = torch.arange(16)
    torch.testing.assert_close(values[SIM_TO_REAL][REAL_TO_SIM], values)
    unwrapped = unwrap_angle(torch.tensor([3.13]), torch.tensor([-3.13]))
    assert 0.0 < float(unwrapped - 3.13) < 0.03
    reverse = unwrap_angle(torch.tensor([-3.13]), torch.tensor([3.13]))
    assert -0.03 < float(reverse + 3.13) < 0.0
    assert math.isfinite(float(unwrapped))
