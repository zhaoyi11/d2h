from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pick_insert_low_level_observation_layout_matches_155_dim_checkpoint() -> None:
    tree = ast.parse((REPO_ROOT / "src" / "tasks" / "pick_insert" / "env_cfg.py").read_text())
    assert "low_level: LowLevelObsCfg = LowLevelObsCfg()" in ast.unparse(tree)
    common_tree = ast.parse((REPO_ROOT / "src/tasks/common/observations_cfg.py").read_text())
    low_level_cfg = next(
        node for node in common_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LowLevelObsCfg"
    )
    term_names = [
        node.targets[0].id
        for node in low_level_cfg.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "ObsTerm"
    ]
    term_widths = {
        "joint_pos": 16,
        "joint_vel": 16,
        "fingertip_pose": 52,
        "contact_mask": 4,
        "contact_force_mag": 4,
        "contact_pose": 8,
        "external_contact_mask": 4,
        "external_contact_force_mag": 4,
        "external_contact_pose": 8,
        "object_pos": 3,
        "object_quat": 4,
        "object_lin_vel": 3,
        "object_ang_vel": 3,
        "gravity_dir": 3,
        "goal_pos_diff": 3,
        "goal_quat_diff": 4,
        "last_action": 16,
    }

    assert term_names == list(term_widths)
    assert sum(term_widths.values()) == 155
