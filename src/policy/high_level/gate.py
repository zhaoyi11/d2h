"""Gate the frozen low-level hand policy on approach distance, contact, and anchor-reached.

While the arm (cuRobo MPC) is still carrying the hand toward the object, running the frozen
dex_reorient grasp/reorient policy makes the fingers close on empty air and can fight the
approach. This gate lets the low-level policy act only when it can do useful in-hand work, and
otherwise holds the hand open ("stretch"). Per env, each step:

    within5 = || anchor_goal - object_goal || < anchor_object_dist  # commanded anchor vs commanded object goal
    contact = good grasp (thumb + one other finger) touches the object
    reached = the arm has settled at the commanded hand-base pose (anchor reached)

    use_low_level = within5 AND (contact OR reached)               # else -> stretch (open) hand

The stretch pose is the open/flat hand (all finger joints at 0 rad). Because the hand action term
uses ``rescale_to_limits=True`` (``unscale_transform`` maps ``[-1, 1]`` onto the joint limits), the
policy's outputs are *normalized*; the open pose is therefore fed as ``scale_transform(open, lo, hi)``
(the inverse of that mapping), not a zero vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor

import isaaclab.utils.math as math_utils


@dataclass
class LowLevelGateCfg:
    """Configuration for :class:`LowLevelHandGate`."""

    # -- scene / manager handles --
    command_name: str = "object_pose"
    """Command term carrying the anchor target and hand-base tracking metrics."""
    robot_asset: str = "robot"
    object_asset: str = "object"
    hand_action_term: str = "hand_action"
    """Action term driving the finger joints; used to resolve the joint order for the stretch pose."""

    # -- thresholds --
    anchor_object_dist: float = 0.05  # TODO: tune this
    """Max distance (m) between commanded anchor and current object to allow the low-level policy."""
    contact_threshold: float = 1.0
    """Object contact force (N) threshold; matches the ``good_finger_contact`` reward."""
    anchor_achieved_pos: float = 0.01
    """Hand-base position error (m) below which the anchor is considered reached (``correction.anchor_achieved_pos``)."""
    anchor_achieved_rot: Optional[float] = 0.05
    """Hand-base orientation error (rad) for anchor-reached; ``None`` => position only."""

    # -- stretch (open) pose --
    open_joint_pos: Optional[Sequence[float]] = None
    """Open-hand joint targets (rad). ``None`` => the robot's ``default_joint_pos`` for the hand joints."""

    # -- contact predicate --
    contact_fn: Optional[Callable[[object, float], Tensor]] = None
    """Predicate ``(env, threshold) -> bool tensor (N,)``. ``None`` => the repo's good-grasp
    ``contacts`` helper (imported lazily so this module stays free of the heavy task imports)."""


class LowLevelHandGate:
    """Decide, per env, whether to run the frozen hand policy or hold the hand open."""

    def __init__(self, cfg: LowLevelGateCfg | None = None) -> None:
        self.cfg = cfg or LowLevelGateCfg()
        self._contact_fn = self.cfg.contact_fn
        self._stretch: Tensor | None = None  # cached (N, num_hand_joints); limits/defaults are static
        self.last_mask: Tensor | None = None  # last use-low-level mask, for logging/inspection
        self._engaged: Tensor | None = None  # (N,) sticky "grasp established" latch; see _sync_latch
        self._prev_ep_len: Tensor | None = None  # previous episode_length_buf, for reset detection

    def _contacts(self, unwrapped, threshold: float) -> Tensor:
        if self._contact_fn is None:
            # Lazy import: avoids pulling the task/isaaclab-sensor stack in at module load.
            from src.tasks.common.mdps.rewards import contacts

            self._contact_fn = contacts
        return self._contact_fn(unwrapped, threshold)

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
        cfg = self.cfg
        cmd = unwrapped.command_manager.get_term(cfg.command_name)

        # within5: commanded anchor vs commanded object goal, both already in the robot root frame.
        # Comparing the two *commanded* frames (not the anchor against the object's measured pose)
        # keeps the gate stable when the trajectory advances to the next waypoint: the anchor and
        # object goal jump together, so their offset stays ~constant and the hand does not snap back
        # to stretch while the grasped object is still catching up to the new goal.
        anchor_pos_b = cmd.anchor_pose_command_b[:, :3]
        object_goal_pos_b = cmd.pose_command_b[:, :3]
        within5 = torch.norm(anchor_pos_b - object_goal_pos_b, dim=-1) < cfg.anchor_object_dist

        # reached: the arm has settled at the commanded hand-base pose.
        reached = cmd.metrics["hand_base_position_error"] < cfg.anchor_achieved_pos
        if cfg.anchor_achieved_rot is not None:
            reached = reached & (cmd.metrics["hand_base_orientation_error"] < cfg.anchor_achieved_rot)

        # contact: good grasp (thumb + one other finger).
        contact = self._contacts(unwrapped, cfg.contact_threshold).bool()

        # engage_now: the original per-step condition (also drives the first grasp during reach).
        engage_now = within5 & (contact | reached)

        # Grasp latch: once a real grasp (within5 & contact) is seen, stay engaged for the rest of
        # the manipulation. Without it the hand snaps back to stretch at every trajectory advance --
        # the hand-base command jumps ~0.1 m ahead of the arm so `reached` drops, and the fast slew
        # momentarily breaks contact, so `contact | reached` collapses and the open hand drops the
        # object. The latch is cleared on episode reset (in _sync_latch), so the next reach phase
        # still starts with an open hand.
        engaged = self._sync_latch(unwrapped, engage_now)
        engaged |= within5 & contact
        self._engaged = engaged
        return engaged | engage_now

    def _sync_latch(self, unwrapped, template: Tensor) -> Tensor:
        """Return the grasp latch, allocating it lazily and clearing envs that just reset.

        A reset sets ``episode_length_buf`` back to 0, so a strictly-decreasing counter flags it.
        Envs without that buffer (e.g. the fixed-step smoke demo) simply never clear the latch.
        Clearing runs before ``engaged |= within5 & contact`` in :meth:`use_low_level_mask`, so a
        reset step (object back on the table, not grasped) leaves the latch cleared.
        """
        engaged = self._engaged
        if engaged is None or engaged.shape != template.shape:
            engaged = torch.zeros_like(template, dtype=torch.bool)
        ep_len = getattr(unwrapped, "episode_length_buf", None)
        if ep_len is not None:
            if self._prev_ep_len is None or self._prev_ep_len.shape != ep_len.shape:
                self._prev_ep_len = ep_len.clone()
            reset_mask = ep_len < self._prev_ep_len  # counter went backwards => env reset
            if reset_mask.any():
                engaged[reset_mask] = False
            self._prev_ep_len = ep_len.clone()
        return engaged

    def apply(self, hand_action: Tensor, unwrapped) -> Tensor:
        """Return ``hand_action`` where the gate is open and the stretch pose elsewhere."""
        mask = self.use_low_level_mask(unwrapped)
        self.last_mask = mask
        stretch = self._stretch_action(unwrapped).to(dtype=hand_action.dtype, device=hand_action.device)
        return torch.where(mask.unsqueeze(-1), hand_action, stretch)
