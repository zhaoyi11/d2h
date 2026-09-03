"""Shared simulation-to-real contract for the distilled knob policy."""

import math

import torch


CONTROL_HZ = 60
ACTION_ALPHA = 0.5
JOINT_COUNT = 16
FRAME_DIM = 35
HISTORY_FRAMES = 3
HISTORY_DIM = FRAME_DIM * HISTORY_FRAMES

SIM_TO_REAL = [4, 0, 8, 12, 6, 2, 10, 14, 7, 3, 11, 15, 1, 5, 9, 13]
REAL_TO_SIM = [1, 12, 5, 9, 0, 13, 4, 8, 2, 14, 6, 10, 3, 15, 7, 11]

JOINT_LOWER = (
    -0.314, -0.349, -0.314, -0.314, -1.047, -0.470, -1.047, -1.047,
    -0.506, -1.200, -0.506, -0.506, -0.366, -1.340, -0.366, -0.366,
)
JOINT_UPPER = (
    2.230, 2.094, 2.230, 2.230, 1.047, 2.443, 1.047, 1.047,
    1.885, 1.900, 1.885, 1.885, 2.042, 1.880, 2.042, 2.042,
)


def normalize_joint_positions(joint_pos: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    return 2.0 * (joint_pos - lower) / (upper - lower) - 1.0


def scale_absolute_actions(actions: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map normalized policy actions to absolute joint targets in radians."""
    actions = actions.clamp(-1.0, 1.0)
    return lower + 0.5 * (actions + 1.0) * (upper - lower)


def ema_absolute_targets(
    scaled_targets: torch.Tensor,
    previous_targets: torch.Tensor,
    alpha: float = ACTION_ALPHA,
) -> torch.Tensor:
    return alpha * scaled_targets + (1.0 - alpha) * previous_targets


def build_aria_frame(
    joint_pos: torch.Tensor,
    applied_targets: torch.Tensor,
    knob_angle: torch.Tensor,
    knob_velocity: torch.Tensor,
    target_angle: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    scalars = [value.unsqueeze(-1) if value.ndim == joint_pos.ndim - 1 else value for value in (knob_angle, knob_velocity, target_angle)]
    return torch.cat((normalize_joint_positions(joint_pos, lower, upper), applied_targets, *scalars), dim=-1)


def initialize_history(frame: torch.Tensor) -> torch.Tensor:
    return torch.cat((frame,) * HISTORY_FRAMES, dim=-1)


def append_history(history: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    return torch.cat((history[..., FRAME_DIM:], frame), dim=-1)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def unwrap_angle(previous: torch.Tensor, wrapped: torch.Tensor) -> torch.Tensor:
    return previous + wrap_to_pi(wrapped - wrap_to_pi(previous))
