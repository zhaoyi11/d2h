# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment drop-detection / recovery state machine for the scripted trajectory command.

Pure ``torch`` (no isaaclab), so it can be loaded by file path and unit-tested in isolation --
mirroring ``trajectory_stepper.py`` / ``anchor_correction.py``. The command term feeds it three
per-env boolean masks each step (whether the object is currently *secured*, whether it looks
*dropped*, and whether it is *at rest*) and gets back the envs whose trajectory should be
regenerated. It owns only the bookkeeping; the drop/secured/at-rest signals and the actual
regeneration (``_resample_command``) stay in the command term.

Per env the machine is::

    DISARMED --(secured: grasped & lifted)--> ARMED
    ARMED    --(dropped: lift height collapses)--> RECOVERING
    RECOVERING --(at rest for ``settle_steps`` consecutive steps)--> emit as "settled"

An env is *armed* only after it has genuinely secured the object, so the pre-grasp reach (object at
its spawn pose, not yet grasped) never looks like a drop. The caller regenerates the emitted envs and
then calls :meth:`reset` on them (disarming), so the fresh pre-grasp reach after a regeneration is
likewise a non-event until the object is secured again.
"""

from __future__ import annotations

import torch


class DropRecoveryTracker:
    """Holds the per-env arm/recover/settle state for object-drop recovery.

    See the module docstring for the state machine. All buffers are ``(num_envs,)``.
    """

    def __init__(self, num_envs: int, device: torch.device | str, settle_steps: int) -> None:
        if settle_steps < 1:
            raise ValueError(f"settle_steps must be >= 1; got {settle_steps}.")
        self.num_envs = num_envs
        self.device = device
        self._settle_steps = int(settle_steps)
        self._armed = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._recovering = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._settle_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def recovering(self) -> torch.Tensor:
        """Per-env mask: is the env waiting for the dropped object to settle? Shape ``(num_envs,)``."""
        return self._recovering

    @property
    def armed(self) -> torch.Tensor:
        """Per-env mask: has the object been secured (so a drop can be detected)? Shape ``(num_envs,)``."""
        return self._armed

    def reset(self, env_ids: torch.Tensor) -> None:
        """Disarm ``env_ids`` and clear their recovery/settle state (call on every (re)sample)."""
        self._armed[env_ids] = False
        self._recovering[env_ids] = False
        self._settle_counter[env_ids] = 0

    def update(
        self,
        secured: torch.Tensor,
        dropped: torch.Tensor,
        at_rest: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the state machine one step and return the envs ready to be regenerated.

        ``secured`` / ``dropped`` / ``at_rest`` are ``(num_envs,)`` bool masks. Returns a 1-D tensor of
        env indices whose dropped object has come to rest and should be resampled by the caller (which
        must then call :meth:`reset` on them, e.g. via ``_resample_command``).
        """
        # Arm latches on once the object has been genuinely secured.
        self._armed |= secured
        # A drop is only meaningful for an armed env that is not already recovering.
        newly = self._armed & dropped & ~self._recovering
        self._recovering |= newly
        self._settle_counter[newly] = 0
        # Settle counting: count consecutive at-rest steps while recovering; reset the count on motion.
        self._settle_counter[self._recovering & ~at_rest] = 0
        self._settle_counter[self._recovering & at_rest] += 1
        settled = self._recovering & (self._settle_counter >= self._settle_steps)
        # Transition out of RECOVERING; the caller regenerates + disarms via reset().
        self._recovering[settled] = False
        self._settle_counter[settled] = 0
        return settled.nonzero().flatten()


__all__ = ["DropRecoveryTracker"]
