from __future__ import annotations

import math
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = lhs.unbind(-1)
    w2, x2, y2, z2 = rhs.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vec[..., :1]), vec), dim=-1)
    conjugate = quat * quat.new_tensor([1.0, -1.0, -1.0, -1.0])
    return _quat_mul(_quat_mul(quat, pure), conjugate)[..., 1:]


def _load_trajectory_module():
    module_names = ("isaaclab", "isaaclab.utils", "isaaclab.utils.math")
    previous = {name: sys.modules.get(name) for name in module_names}
    isaaclab = types.ModuleType("isaaclab")
    utils = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")
    math_module.quat_apply = _quat_apply
    math_module.apply_delta_pose = lambda *args, **kwargs: None
    math_module.combine_frame_transforms = lambda *args, **kwargs: None
    math_module.subtract_frame_transforms = lambda *args, **kwargs: None
    sys.modules["isaaclab"] = isaaclab
    sys.modules["isaaclab.utils"] = utils
    sys.modules["isaaclab.utils.math"] = math_module
    spec = importlib.util.spec_from_file_location(
        "clean_table_trajectory_under_test",
        REPO_ROOT / "src/tasks/clean_table/mdps/trajectory.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


trajectory_module = _load_trajectory_module()
DEFAULT_CLEAN_TABLE_SEGMENT_STEPS = trajectory_module.DEFAULT_CLEAN_TABLE_SEGMENT_STEPS
build_clean_table_object_pose_sequence = trajectory_module.build_clean_table_object_pose_sequence


def test_clean_table_trajectory_releases_in_box_then_retreats() -> None:
    current = torch.tensor([0.55, 0.25, 0.34, 0.5, 0.5, 0.5, 0.5])
    box = torch.tensor([0.50, -0.20, 0.271, 1.0, 0.0, 0.0, 0.0])

    trajectory = build_clean_table_object_pose_sequence(current, box)

    assert trajectory.shape == (1 + sum(DEFAULT_CLEAN_TABLE_SEGMENT_STEPS), 7)
    torch.testing.assert_close(trajectory[0], current)
    torch.testing.assert_close(trajectory[:, 3:7], current[3:7].expand_as(trajectory[:, 3:7]))

    deposit_step = 1 + sum(DEFAULT_CLEAN_TABLE_SEGMENT_STEPS[:5])
    release_end_step = 1 + sum(DEFAULT_CLEAN_TABLE_SEGMENT_STEPS[:6])
    torch.testing.assert_close(trajectory[deposit_step, :3], torch.tensor([0.50, -0.20, 0.336]))
    torch.testing.assert_close(
        trajectory[deposit_step:release_end_step, :3],
        trajectory[deposit_step, :3].expand(release_end_step - deposit_step, 3),
    )
    torch.testing.assert_close(trajectory[-1, :3], torch.tensor([0.50, -0.20, 0.521]))


def test_clean_table_trajectory_applies_box_rotation_to_local_offsets() -> None:
    half_sqrt = math.sqrt(0.5)
    current = torch.tensor([0.55, 0.25, 0.34, 1.0, 0.0, 0.0, 0.0])
    box = torch.tensor([0.40, -0.10, 0.30, half_sqrt, half_sqrt, 0.0, 0.0])

    trajectory = build_clean_table_object_pose_sequence(
        current,
        box,
        box_target_offset=(0.02, 0.0, 0.065),
        retreat_offset=(0.02, 0.0, 0.25),
    )

    deposit_step = 1 + sum(DEFAULT_CLEAN_TABLE_SEGMENT_STEPS[:5])
    torch.testing.assert_close(
        trajectory[deposit_step, :3],
        torch.tensor([0.42, -0.165, 0.30]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        trajectory[-1, :3],
        torch.tensor([0.42, -0.35, 0.30]),
        atol=1.0e-6,
        rtol=1.0e-6,
    )


@pytest.mark.parametrize("steps", [(0, 1), (0, 1, 1, 2, 2, 12, -1)])
def test_clean_table_trajectory_rejects_invalid_segment_steps(steps: tuple[int, ...]) -> None:
    pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        build_clean_table_object_pose_sequence(pose, pose, segment_steps=steps)
