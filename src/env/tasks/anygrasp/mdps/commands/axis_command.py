# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command generator for axis-alignment goals for in-hand manipulation.

This command generates a target unit axis in world frame. The task is to rotate the object such that a
specified object-frame axis aligns with the target axis (rotation about that axis is unconstrained).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import InHandAxisAlignCommandCfg


class InHandAxisAlignCommand(CommandTerm):
    """Command term that generates target axes for axis-alignment in-hand manipulation."""

    cfg: InHandAxisAlignCommandCfg

    def __init__(self, cfg: InHandAxisAlignCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.object: RigidObject = env.scene[cfg.asset_name]

        # buffers
        self.axis_command_w = torch.zeros(self.num_envs, 3, device=self.device)

        axis_b = torch.tensor(cfg.object_axis, dtype=torch.float32, device=self.device)
        self._object_axis_b = axis_b / torch.linalg.norm(axis_b).clamp_min(1e-12)
        self._object_axis_b = self._object_axis_b.repeat(self.num_envs, 1)

        # unit vectors
        self._Z_UNIT_VEC = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
            (self.num_envs, 1)
        )
        self._X_UNIT_VEC = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(
            (self.num_envs, 1)
        )

        # metrics
        self.metrics["axis_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["axis_alignment"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(
            self.num_envs, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        """Target axis in world frame. Shape: (num_envs, 3)."""
        return self.axis_command_w

    def _update_metrics(self):
        # object axis in world
        obj_axis_w = math_utils.quat_apply(
            self.object.data.root_quat_w, self._object_axis_b
        )
        obj_axis_w = obj_axis_w / torch.linalg.norm(
            obj_axis_w, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        tgt_axis_w = self.axis_command_w

        # alignment
        dot = (obj_axis_w * tgt_axis_w).sum(dim=-1).clamp(-1.0, 1.0)
        if self.cfg.use_abs_dot:
            dot = dot.abs()
        self.metrics["axis_alignment"] = dot
        self.metrics["axis_error"] = torch.arccos(dot)

        successes = self.metrics["axis_error"] < self.cfg.axis_success_threshold
        self.metrics["consecutive_success"] += successes.float()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if self.cfg.target_axis_mode == "world_z":
            axis = self._Z_UNIT_VEC[env_ids]
        elif self.cfg.target_axis_mode == "random":
            axis = torch.randn((len(env_ids), 3), device=self.device)
            axis = axis / torch.linalg.norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            raise ValueError(
                f"Unsupported target_axis_mode: {self.cfg.target_axis_mode}"
            )

        self.axis_command_w[env_ids] = axis

    def _update_command(self):
        if not self.cfg.update_goal_on_success:
            return
        goal_resets = self.metrics["axis_error"] < self.cfg.axis_success_threshold
        goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
        self._resample(goal_reset_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "axis_visualizer"):
                self.axis_visualizer = VisualizationMarkers(
                    self.cfg.axis_visualizer_cfg
                )
            self.axis_visualizer.set_visibility(True)
        else:
            if hasattr(self, "axis_visualizer"):
                self.axis_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # visualize the target axis as a frame at the object position.
        pos = self.object.data.root_pos_w + torch.tensor(
            self.cfg.marker_pos_offset, device=self.device
        )

        # build quaternion that rotates +Z to target axis
        z = self._Z_UNIT_VEC
        t = self.axis_command_w
        dot = (z * t).sum(dim=-1).clamp(-1.0, 1.0)
        axis = torch.cross(z, t, dim=-1)
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)

        # handle near-parallel vectors
        near = axis_norm.squeeze(-1) < 1e-6
        axis = axis / axis_norm.clamp_min(1e-12)
        angle = torch.arccos(dot)

        # if z ~ -t, choose a stable axis (x)
        axis[near] = self._X_UNIT_VEC[near]
        angle[near] = torch.where(
            dot[near] > 0.0,
            torch.zeros_like(angle[near]),
            torch.full_like(angle[near], torch.pi),
        )

        quat = math_utils.quat_from_angle_axis(angle, axis)
        self.axis_visualizer.visualize(translations=pos, orientations=quat)
