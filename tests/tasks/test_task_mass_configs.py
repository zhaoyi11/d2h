from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)


def _assigned_call(class_node: ast.ClassDef, name: str) -> ast.Call:
    for node in class_node.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Call)
        ):
            return node.value
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
            and isinstance(node.value, ast.Call)
        ):
            return node.value
    raise AssertionError(f"Could not find call assigned to {class_node.name}.{name}.")


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _dict_value(node: ast.expr, key: str) -> ast.expr:
    assert isinstance(node, ast.Dict)
    for dict_key, value in zip(node.keys, node.values, strict=True):
        if isinstance(dict_key, ast.Constant) and dict_key.value == key:
            return value
    raise AssertionError(f"Could not find {key!r} in dictionary.")


def _assert_task_mass_config(relative_path: str, expected_range: tuple[float, float]) -> None:
    tree = ast.parse((REPO_ROOT / relative_path).read_text())

    event_cfg = _class(tree, "EventCfg")
    mass_event = _assigned_call(event_cfg, "object_scale_mass")
    assert ast.unparse(_keyword(mass_event, "func")) == "mdp.randomize_rigid_body_mass"
    assert ast.literal_eval(_keyword(mass_event, "mode")) == "startup"
    params = _keyword(mass_event, "params")
    assert tuple(ast.literal_eval(_dict_value(params, "mass_distribution_params"))) == expected_range
    assert ast.literal_eval(_dict_value(params, "operation")) == "abs"

    scene_cfg = _class(tree, "SceneCfg")
    for asset_name in ("object", "receptive_object"):
        asset_cfg = _assigned_call(scene_cfg, asset_name)
        spawn_cfg = _keyword(asset_cfg, "spawn")
        assert isinstance(spawn_cfg, ast.Call)
        assert all(keyword.arg != "mass_props" for keyword in spawn_cfg.keywords)


def test_unscrew_uses_intended_absolute_mass_range() -> None:
    _assert_task_mass_config("src/tasks/unscrew/env_cfg.py", (0.004, 0.040))


def test_pick_insert_uses_intended_absolute_mass_range() -> None:
    _assert_task_mass_config("src/tasks/pick_insert/env_cfg.py", (0.010, 0.100))


def test_clean_table_uses_pick_anyrotate_mass_baseline() -> None:
    tree = ast.parse((REPO_ROOT / "src/tasks/clean_table/env_cfg.py").read_text())
    event_cfg = _class(tree, "EventCfg")
    mass_event = _assigned_call(event_cfg, "object_scale_mass")
    params = _keyword(mass_event, "params")
    assert tuple(ast.literal_eval(_dict_value(params, "mass_distribution_params"))) == (0.2, 2.0)
    assert ast.literal_eval(_dict_value(params, "operation")) == "scale"

    scene_cfg = _class(tree, "SceneCfg")
    object_cfg = _assigned_call(scene_cfg, "object")
    spawn_cfg = _keyword(object_cfg, "spawn")
    mass_props = _keyword(spawn_cfg, "mass_props")
    assert ast.literal_eval(_keyword(mass_props, "mass")) == 0.2
