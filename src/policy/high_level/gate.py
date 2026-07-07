"""Gate the frozen low-level hand policy on how close the hand is to the object.

While the arm (cuRobo MPC) is still carrying the hand toward the object, running the frozen
dex_reorient grasp/reorient policy makes the fingers close on empty air and can fight the approach.
This gate holds the hand open ("stretch") during the pre-grasp reach and switches to the low-level
policy once the hand has arrived at the object.

The decision is a single distance test, per env, each step::

    use_low_level = hand_base_object_error < hand_at_object_dist    # else -> stretch (open) hand

``hand_base_object_error`` is published by the trajectory command term: the distance from the current
hand-base to the grasp anchor of the *live* object (its current pose, not the commanded goal). It is
large during the approach (-> stretch), small once the arm arrives and stays small through transport
(the grasped object moves with the hand, so the policy stays engaged with no latch), and large again
after a drop (-> stretch, so the next reach re-opens the hand and re-grasps). Because it keys off the
current object, no goal-frame terms, contact predicate, or reset/replan bookkeeping are needed.

The stretch pose is the open/flat hand (all finger joints at 0 rad). Because the hand action term
uses ``rescale_to_limits=True`` (``unscale_transform`` maps ``[-1, 1]`` onto the joint limits), the
policy's outputs are *normalized*; the open pose is therefore fed as ``scale_transform(open, lo, hi)``
(the inverse of that mapping), not a zero vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

import isaaclab.utils.math as math_utils


@dataclass
class LowLevelGateCfg:
    """Configuration for :class:`LowLevelHandGate`."""

    # -- scene / manager handles --
    command_name: str = "object_pose"
    """Command term publishing ``metrics["hand_base_object_error"]`` (the current hand<->object distance)."""
    robot_asset: str = "robot"
    hand_action_term: str = "hand_action"
    """Action term driving the finger joints; used to resolve the joint order for the stretch pose."""

    # -- threshold --
    hand_at_object_dist: float = 0.08  # TODO: tune from the logged ``hand_base_object_error``
    """Max hand-base<->live-object distance (m) to run the low-level policy; above this => stretch."""

    # -- stretch (open) pose --
    open_joint_pos: Optional[Sequence[float]] = None
    """Open-hand joint targets (rad). ``None`` => the robot's ``default_joint_pos`` for the hand joints."""


class LowLevelHandGate:
    """Decide, per env, whether to run the frozen hand policy or hold the hand open."""

    def __init__(self, cfg: LowLevelGateCfg | None = None) -> None:
        self.cfg = cfg or LowLevelGateCfg()
        self._stretch: Tensor | None = None  # cached (N, num_hand_joints); limits/defaults are static
        self.last_mask: Tensor | None = None  # last use-low-level mask, for logging/inspection

    def _hand_joint_ids(self, unwrapped):
        term = unwrapped.action_manager.get_term(self.cfg.hand_action_term)
        # Resolved from ["a_.*"]; preserves the joint order the frozen policy's actions use.
        return term._joint_ids

    def _stretch_action(self, unwrapped) -> Tensor:
        if self._stretch is not None:
            return self._stretch
        robot = unwrapped.scene[self.cfg.robot_asset]
        joint_ids = self._hand_joint_ids(unwrapped)
        limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
        lo = limits[..., 0]
        hi = limits[..., 1]
        if self.cfg.open_joint_pos is None:
            open_pos = robot.data.default_joint_pos[:, joint_ids]
        else:
            open_pos = torch.as_tensor(self.cfg.open_joint_pos, dtype=lo.dtype, device=lo.device)
            open_pos = open_pos.reshape(1, -1).expand(lo.shape[0], -1)
        # Inverse of the action term's unscale_transform: the normalized action mapping to the open pose.
        self._stretch = math_utils.scale_transform(open_pos, lo, hi)
        return self._stretch

    def use_low_level_mask(self, unwrapped) -> Tensor:
        """Return a ``(N,)`` bool mask: True => run the low-level policy, False => stretch."""
        cmd = unwrapped.command_manager.get_term(self.cfg.command_name)
        # The command term logs the current hand-base distance to the live object's grasp anchor:
        # large during the reach, small at/through the grasp, large after a drop. One threshold does it.
        metrics = getattr(cmd, "metrics", {})
        err = metrics.get("hand_base_object_error")
        if err is None:
            # Non-trajectory command (e.g. a fixed-step demo): nothing to gate on, run the policy.
            return torch.ones(unwrapped.num_envs, dtype=torch.bool, device=unwrapped.device)
        return err < self.cfg.hand_at_object_dist

    def apply(self, hand_action: Tensor, unwrapped) -> Tensor:
        """Return ``hand_action`` where the gate is open and the stretch pose elsewhere."""
        mask = self.use_low_level_mask(unwrapped)
        self.last_mask = mask
        stretch = self._stretch_action(unwrapped).to(dtype=hand_action.dtype, device=hand_action.device)
        return torch.where(mask.unsqueeze(-1), hand_action, stretch)
