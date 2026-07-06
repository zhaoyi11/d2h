# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level (reference / subgoal generation) building blocks for the HRL stack.

Groups the pure-``torch`` "brains" that decide *what to do* -- the scripted object-pose trajectory
primitives, the per-env trajectory stepper, and the bounded PI(D) anchor correction. The isaaclab
``CommandTerm`` in ``src.tasks.common.mdps.commands`` is a thin adapter that imports and drives
these; execution (cuRobo MPC arm + frozen low-level hand policy) lives in ``src.policy.low_level``.

The engagement gate (:mod:`.gate`) is intentionally *not* re-exported here: it depends on isaaclab,
so importing it eagerly would pull isaaclab into this package. Import it as
``from src.policy.high_level.gate import LowLevelHandGate`` when needed, keeping the pure-torch
modules above loadable in isolation.
"""

from .anchor_correction import ObjectAnchorPIDController, clamp_norm, slew_limit
from .object_trajectory import (
    DEFAULT_SEGMENT_STEPS,
    build_object_pose_sequence_from_keyframes,
    interpolate_pose_segment,
    slerp,
)
from .trajectory_stepper import TrajectoryStepper

__all__ = [
    "DEFAULT_SEGMENT_STEPS",
    "ObjectAnchorPIDController",
    "TrajectoryStepper",
    "build_object_pose_sequence_from_keyframes",
    "clamp_norm",
    "interpolate_pose_segment",
    "slerp",
    "slew_limit",
]
