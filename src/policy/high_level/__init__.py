# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""High-level (reference / subgoal generation) building blocks for the HRL stack.

Groups the pure-``torch`` "brains" that decide *what to do* -- the per-env trajectory stepper and
the bounded PI(D) anchor correction. The isaaclab ``CommandTerm`` in
``src.tasks.common.mdps.commands`` is a thin adapter that imports and drives these; execution
(cuRobo MPC arm + frozen low-level hand policy) lives in ``src.policy.low_level``.

The isaaclab-dependent modules (:mod:`.gate`, :mod:`.utils` and :mod:`.trajectory_command`) are
intentionally *not* re-exported here: importing them eagerly would pull isaaclab into this package.
Import them directly (e.g. ``from src.policy.high_level.gate import LowLevelHandGate``,
``from src.policy.high_level.utils import hand_base_pose_from_object_command_b`` -- which also holds
the scripted object-pose trajectory primitives -- or
``from src.policy.high_level.trajectory_command import TrajectoryObjectAndHandBasePoseCommand``) when
needed, keeping the pure-torch modules above loadable in isolation. ``AnchorCorrectionCfg`` is a
plain ``@dataclass`` (no isaaclab), so it *is* re-exported alongside its controller.
"""

from .anchor_correction import AnchorCorrectionCfg, ObjectAnchorPIDController, clamp_norm, slew_limit
from .trajectory_stepper import StageObjTol, TrajectoryStepper, stage_tolerance_tensors

__all__ = [
    "AnchorCorrectionCfg",
    "ObjectAnchorPIDController",
    "StageObjTol",
    "TrajectoryStepper",
    "clamp_norm",
    "slew_limit",
    "stage_tolerance_tensors",
]
