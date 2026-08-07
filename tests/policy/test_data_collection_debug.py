from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_data_collection_debug_defaults_to_reorient_debug_play_task() -> None:
    tree = ast.parse((REPO_ROOT / "src" / "policy" / "data_collection_debug.py").read_text())
    task_default = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant) or node.args[0].value != "--task":
            continue
        task_default = next(kw.value.value for kw in node.keywords if kw.arg == "default")
        break

    assert task_default == "Reorient_Debug_Play-v0"
