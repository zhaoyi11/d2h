# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded PI(D) controller that nudges the anchor pose so the object reaches its goal.

Pure ``torch`` (no isaaclab), so it can be loaded by file path and unit-tested in isolation --
mirroring ``trajectory_stepper.py``. The command term owns everything frame-coupled (robot/object
poses, ``compute_pose_error``, the anchor-achieved gate) and feeds this controller plain per-env
error tensors; the controller owns the stateful PI(D) buffers and the stall gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def clamp_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    """Scale each row of ``vec`` so its norm does not exceed ``max_norm``."""
    norm = torch.norm(vec, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / torch.clamp(norm, min=1e-8), max=1.0)
    return vec * scale


def slew_limit(prev: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    """Move ``prev`` toward ``target`` by at most ``max_step`` (per-vector norm) this step."""
    delta = target - prev
    norm = torch.norm(delta, dim=-1, keepdim=True)
    scale = torch.clamp(max_step / torch.clamp(norm, min=1e-8), max=1.0)
    return prev + delta * scale


class ObjectAnchorPIDController:
    """Stateful bounded PI(D) controller producing an anchor-frame correction (pos[3], axis-angle[3]).

    The correction is gated by two conditions supplied/derived per step:

    1. **Anchor achieved** (passed in): the hand-base (arm) has settled within tolerance of its
       commanded pose. Computed by the command term (it owns the robot pose).
    2. **Object stalled** (computed here): the object error has not improved by at least
       ``stall_delta_{pos,rot}`` over the last ``stall_window`` steps. ``stall_window == 0``
       disables this gate.

    When both hold, a bounded PI(D) controller runs on the measured object->goal error expressed in
    the root-aligned anchor frame: deadband -> anti-windup integral -> P+I+(optional filtered D) ->
    output clamp -> output slew. The deadband lets the in-hand policy own small errors; the
    anti-windup + output clamp bound the deviation to ``max_pos`` / ``max_rot``; the slew limit
    makes the applied correction change gradually.

    ``cfg`` is duck-typed (any object exposing the ``deadband_pos``, ``kp_pos``, ``ki_pos``,
    ``kd_pos``, ``d_lowpass``, ``max_pos``, ``slew_pos`` -- and ``*_rot`` -- plus ``stall_window``,
    ``stall_delta_pos``, ``stall_delta_rot`` attributes), so this module stays isaaclab-free.
    """

    def __init__(self, num_envs: int, device: torch.device | str, cfg) -> None:
        self._cfg = cfg
        self.num_envs = num_envs
        self.device = device
        # Integral accumulator (anti-windup clamped).
        self._integral = torch.zeros(num_envs, 6, device=device)
        # Slew-limited applied output (the correction added to the nominal anchor).
        self._correction = torch.zeros(num_envs, 6, device=device)
        # Previous (deadbanded) error and filtered derivative, for the optional D term.
        self._prev_err = torch.zeros(num_envs, 6, device=device)
        self._deriv = torch.zeros(num_envs, 6, device=device)
        # Stall detection: last object error magnitudes and consecutive no-improvement counter.
        self._err_history_pos = torch.zeros(num_envs, device=device)
        self._err_history_rot = torch.zeros(num_envs, device=device)
        self._stall_counter = torch.zeros(num_envs, dtype=torch.long, device=device)

    @property
    def correction(self) -> torch.Tensor:
        """The applied anchor-frame correction buffer. Shape ``(num_envs, 6)``."""
        return self._correction

    def reset(self, env_ids: torch.Tensor | Sequence[int] | slice) -> None:
        """Zero all controller state for ``env_ids`` (sit at the nominal anchor)."""
        self._integral[env_ids] = 0.0
        self._correction[env_ids] = 0.0
        self._prev_err[env_ids] = 0.0
        self._deriv[env_ids] = 0.0
        self._err_history_pos[env_ids] = 0.0
        self._err_history_rot[env_ids] = 0.0
        self._stall_counter[env_ids] = 0

    def clear_stall(self, env_ids: torch.Tensor | Sequence[int] | slice) -> None:
        """Reset only the stall counter for ``env_ids`` (e.g. when the trajectory advances)."""
        self._stall_counter[env_ids] = 0

    def update(
        self,
        env_ids: torch.Tensor,
        err_pos: torch.Tensor,
        err_rot: torch.Tensor,
        anchor_achieved: torch.Tensor,
    ) -> None:
        """Advance the controller for ``env_ids`` and write the correction for the gated subset.

        Args:
            env_ids: Long tensor of environment indices this call operates on.
            err_pos: Object->goal position error (root frame) for ``env_ids``. Shape ``(N, 3)``.
            err_rot: Object->goal orientation error (axis-angle, root frame). Shape ``(N, 3)``.
            anchor_achieved: Bool mask (over ``env_ids``) that the arm has settled. Shape ``(N,)``.
        """
        if env_ids.numel() == 0:
            return
        cfg = self._cfg

        cur_err_pos = torch.norm(err_pos, dim=-1)
        cur_err_rot = torch.norm(err_rot, dim=-1)

        # --- Object stall gate (updates history/counter for every env in env_ids) ---
        if cfg.stall_window > 0:
            improved = (
                (self._err_history_pos[env_ids] - cur_err_pos > cfg.stall_delta_pos)
                | (self._err_history_rot[env_ids] - cur_err_rot > cfg.stall_delta_rot)
            )
            self._stall_counter[env_ids] = torch.where(
                improved,
                torch.zeros_like(self._stall_counter[env_ids]),
                self._stall_counter[env_ids] + 1,
            )
            self._err_history_pos[env_ids] = cur_err_pos
            self._err_history_rot[env_ids] = cur_err_rot
            object_stalled = self._stall_counter[env_ids] >= cfg.stall_window
        else:
            object_stalled = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)

        # Only run the PI(D) where both gates hold; operate on that subset consistently.
        gate_mask = anchor_achieved & object_stalled
        active_ids = env_ids[gate_mask]
        if active_ids.numel() == 0:
            return
        err_pos = err_pos[gate_mask]
        err_rot = err_rot[gate_mask]

        # Per-axis deadband: within tolerance the in-hand policy owns the error; the arm waits.
        err_pos = torch.sign(err_pos) * torch.clamp(err_pos.abs() - cfg.deadband_pos, min=0.0)
        err_rot = torch.sign(err_rot) * torch.clamp(err_rot.abs() - cfg.deadband_rot, min=0.0)
        err = torch.cat((err_pos, err_rot), dim=-1)

        # Integral term (anti-windup clamped) -- removes steady-state error.
        integ = self._integral[active_ids].clone()
        integ[:, :3] = clamp_norm(integ[:, :3] + cfg.ki_pos * err_pos, cfg.max_pos)
        integ[:, 3:] = clamp_norm(integ[:, 3:] + cfg.ki_rot * err_rot, cfg.max_rot)
        self._integral[active_ids] = integ

        # Proportional + Integral term.
        raw = torch.zeros_like(integ)
        raw[:, :3] = cfg.kp_pos * err_pos + integ[:, :3]
        raw[:, 3:] = cfg.kp_rot * err_rot + integ[:, 3:]

        # Optional filtered-derivative term (off by default; pose-error derivatives are noisy).
        if cfg.kd_pos != 0.0 or cfg.kd_rot != 0.0:
            d_err = err - self._prev_err[active_ids]
            deriv = (1.0 - cfg.d_lowpass) * self._deriv[active_ids] + cfg.d_lowpass * d_err
            self._deriv[active_ids] = deriv
            raw[:, :3] = raw[:, :3] + cfg.kd_pos * deriv[:, :3]
            raw[:, 3:] = raw[:, 3:] + cfg.kd_rot * deriv[:, 3:]
        self._prev_err[active_ids] = err

        # Output clamp to the +-max_pos / +-max_rot budget.
        raw[:, :3] = clamp_norm(raw[:, :3], cfg.max_pos)
        raw[:, 3:] = clamp_norm(raw[:, 3:], cfg.max_rot)

        # Slew-limit the applied correction toward the target so it changes gradually.
        prev = self._correction[active_ids]
        corr = torch.cat(
            (
                slew_limit(prev[:, :3], raw[:, :3], cfg.slew_pos),
                slew_limit(prev[:, 3:], raw[:, 3:], cfg.slew_rot),
            ),
            dim=-1,
        )
        self._correction[active_ids] = corr


@dataclass
class AnchorCorrectionCfg:
    """Bounded PI(D) object->goal correction (nudges the anchor so the object reaches its goal).

    Consumed by :class:`ObjectAnchorPIDController`; ``enable`` / ``anchor_achieved_*`` are also read
    by the command term (it owns the anchor-achieved gate). A plain ``@dataclass`` (not isaaclab's
    ``@configclass``) so this module stays pure-``torch`` and loadable in isolation -- it is nested
    into the ``@configclass`` command cfg, which handles the mutable-default wrapping of the
    ``correction`` field. Gains/limits default to a starting point -- see the TODO on tuning below.
    """

    enable: bool = False
    """Whether to run the correction (PI(D) on the measured object->goal error). When False the
    command exposes the bare nominal anchor offset."""

    deadband_pos: float = 0.00
    """Per-axis object position error (m) below which the controller idles -- the in-hand policy
    owns sub-deadband error, so the arm only acts when the hand can't (residual tolerance)."""

    deadband_rot: float = 0.0
    """Per-axis object orientation error (rad) below which the controller idles (residual tol)."""

    # TODO: !!!! Tune the PI(D) gains and correction limits. The current values are a starting point
    # based on intuition and preliminary experiments; systematic tuning is needed for best results.
    kp_pos: float = 0.1
    """Proportional gain on the (deadbanded) position error -- responsiveness."""

    kp_rot: float = 0.05
    """Proportional gain on the rotation error (smaller so the hand does most reorientation)."""

    ki_pos: float = 0.1
    """Integral gain on the position error -- removes steady-state error (object reaches goal)."""

    ki_rot: float = 0.05
    """Integral gain on the rotation error (smaller -- rotation commits slowly/reluctantly)."""

    kd_pos: float = 0.1
    """Derivative gain on the position error (off by default; pose-error derivatives are noisy)."""

    kd_rot: float = 0.05
    """Derivative gain on the rotation error (off by default)."""

    d_lowpass: float = 0.2
    """EMA factor for the derivative term (smaller = more smoothing). Used only when Kd != 0."""

    max_pos: float = 0.05
    """Output + integral bound: max position deviation (m) of the offset from its initial value."""

    max_rot: float = 0.1745
    """Output + integral bound: max rotation deviation (rad, ~10 deg) of the offset from initial."""

    slew_pos: float = 0.002
    """Max change (m) of the applied position correction per step -- keeps adjustment gradual."""

    slew_rot: float = 0.005
    """Max change (rad) of the applied rotation correction per step -- keeps adjustment gradual."""

    anchor_achieved_pos: float = 0.01
    """Hand-base position tolerance (m) to consider the anchor achieved (arm settled)."""

    anchor_achieved_rot: float = 0.05
    """Hand-base orientation tolerance (rad) to consider the anchor achieved."""

    stall_window: int = 30
    """Number of steps over which the object error must not improve by stall_delta to be considered
    stalled. Zero disables the stall gate (correction fires whenever the anchor is achieved)."""

    stall_delta_pos: float = 0.003
    """Object position improvement (m) below which the object is considered stalled."""

    stall_delta_rot: float = 0.01
    """Object orientation improvement (rad) below which the object is considered stalled."""


__all__ = ["AnchorCorrectionCfg", "ObjectAnchorPIDController", "clamp_norm", "slew_limit"]
