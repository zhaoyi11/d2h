from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reorientation_sequence_complete(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
) -> torch.Tensor:
    """Terminate environments that completed all six orientation targets."""
    command = env.command_manager.get_term(command_name)
    return command.metrics["sequence_complete"].bool()
