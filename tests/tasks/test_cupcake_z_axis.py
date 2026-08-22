from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_CFG_PATH = REPO_ROOT / "src/tasks/cupcake_on_plate/env_cfg.py"
TRAJECTORY_PATH = REPO_ROOT / "src/tasks/cupcake_on_plate/mdps/trajectory.py"
COMMANDS_PATH = REPO_ROOT / "src/tasks/cupcake_on_plate/mdps/commands.py"
TASK_MDPS_PATH = REPO_ROOT / "src/tasks/cupcake_on_plate/mdps/task_mdps.py"
ASSET_PATH = REPO_ROOT / "src/assets/uwlab/cake_z_axix.usd"


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name)


def _assignments(class_node: ast.ClassDef) -> dict[str, ast.expr]:
    result = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node.value
    return result


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _dict_value(node: ast.Dict, key: str) -> ast.expr:
    for dict_key, value in zip(node.keys, node.values, strict=True):
        if isinstance(dict_key, ast.Constant) and dict_key.value == key:
            return value
    raise AssertionError(f"Missing dictionary key {key!r}.")


def _load_trajectory_module():
    module_names = ("isaaclab", "isaaclab.utils", "isaaclab.utils.math")
    previous = {name: sys.modules.get(name) for name in module_names}
    previous_utils = sys.modules.pop("src.policy.high_level.utils", None)
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")
    for name in ("apply_delta_pose", "combine_frame_transforms", "subtract_frame_transforms"):
        setattr(isaaclab_math, name, lambda *args, **kwargs: None)
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.utils": isaaclab_utils,
            "isaaclab.utils.math": isaaclab_math,
        }
    )
    spec = importlib.util.spec_from_file_location("cupcake_z_axis_trajectory_under_test", TRAJECTORY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
        sys.modules.pop("src.policy.high_level.utils", None)
        if previous_utils is not None:
            sys.modules["src.policy.high_level.utils"] = previous_utils
    return module


def _load_commands_module():
    trajectory = _load_trajectory_module()
    module_names = (
        "isaaclab",
        "isaaclab.utils",
        "src.policy.high_level.trajectory_command",
        "src.tasks.cupcake_on_plate.mdps.trajectory",
    )
    previous = {name: sys.modules.get(name) for name in module_names}
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_utils.configclass = lambda cls: cls

    trajectory_command = types.ModuleType("src.policy.high_level.trajectory_command")

    class FakeTrajectoryCommand:
        def _resample_command(self, env_ids):
            self.resampled.append(env_ids.clone())
            self._stepper.step[env_ids] = 0

        def _apply_objanchor_correction(self, env_ids):
            self.corrected.append(env_ids.clone())

        def _update_hand_base_pose_command(self, env_ids):
            self.hand_base_updated.append(env_ids.clone())

    class FakeTrajectoryCommandCfg:
        pass

    trajectory_command.TrajectoryObjectAndHandBasePoseCommand = FakeTrajectoryCommand
    trajectory_command.TrajectoryObjectAndHandBasePoseCommandCfg = FakeTrajectoryCommandCfg
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.utils": isaaclab_utils,
            "src.policy.high_level.trajectory_command": trajectory_command,
            "src.tasks.cupcake_on_plate.mdps.trajectory": trajectory,
        }
    )
    spec = importlib.util.spec_from_file_location("cupcake_z_axis_commands_under_test", COMMANDS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
    return module


@pytest.mark.parametrize("yaw_delta", [math.pi / 3.0, math.pi / 2.0, -math.pi / 3.0, -math.pi / 2.0])
def test_yaw_trajectory_holds_position_and_rotates_only_about_z(yaw_delta: float) -> None:
    module = _load_trajectory_module()
    current = torch.tensor([0.55, 0.20, 0.335, 1.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    trajectory = module.build_cupcake_z_axis_object_pose_sequence(current, yaw_delta=yaw_delta)

    assert module.DEFAULT_CUPCAKE_Z_AXIS_SEGMENT_STEPS == (0, 1)
    assert trajectory.shape == (2, 7)
    torch.testing.assert_close(trajectory[:, :3], current[:3].expand(2, -1))
    torch.testing.assert_close(trajectory[0], current)
    expected = torch.tensor(
        [math.cos(yaw_delta / 2.0), 0.0, 0.0, math.sin(yaw_delta / 2.0)],
        dtype=current.dtype,
    )
    torch.testing.assert_close(trajectory[1, 3:7], expected, atol=1.0e-6, rtol=1.0e-6)


def test_signed_yaw_sampler_stays_between_sixty_and_ninety_degrees() -> None:
    module = _load_trajectory_module()
    generator = torch.Generator().manual_seed(7)

    deltas = module.sample_signed_yaw_deltas(
        1024,
        module.DEFAULT_CUPCAKE_Z_AXIS_YAW_DELTA_RANGE,
        dtype=torch.float64,
        device="cpu",
        generator=generator,
    )

    assert deltas.dtype == torch.float64
    assert torch.all(deltas.abs() >= math.pi / 3.0)
    assert torch.all(deltas.abs() <= math.pi / 2.0)
    assert torch.any(deltas < 0.0)
    assert torch.any(deltas > 0.0)


def test_yaw_trajectory_rejects_invalid_parameters() -> None:
    module = _load_trajectory_module()
    pose = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    with pytest.raises(ValueError, match="segment_steps must contain 2 values"):
        module.build_cupcake_z_axis_object_pose_sequence(pose, yaw_delta=math.pi / 3.0, segment_steps=(1,))
    with pytest.raises(ValueError, match="yaw_delta_range"):
        module.sample_signed_yaw_deltas(1, (math.pi / 2.0, math.pi / 3.0), dtype=torch.float32, device="cpu")


def test_cake_z_axix_asset_has_one_unlimited_z_revolute_joint() -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(ASSET_PATH))
    assert stage is not None
    root = stage.GetDefaultPrim()
    assert root.GetName() == "cake_z_axix"
    rigid_bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert [prim.GetPath() for prim in rigid_bodies] == [root.GetPath()]

    joint = UsdPhysics.RevoluteJoint.Get(stage, root.GetPath().AppendChild("z_axis_joint"))
    assert joint
    assert joint.GetAxisAttr().Get() == "Z"
    assert joint.GetBody0Rel().GetTargets() == []
    assert joint.GetBody1Rel().GetTargets() == [root.GetPath()]
    assert math.isinf(joint.GetLowerLimitAttr().Get())
    assert math.isinf(joint.GetUpperLimitAttr().Get())


def test_cupcake_scene_uses_fixed_upright_z_axis_asset() -> None:
    tree = ast.parse(ENV_CFG_PATH.read_text())
    scene = _assignments(_class(tree, "SceneCfg"))
    events = _assignments(_class(tree, "EventCfg"))

    object_cfg = scene["object"]
    object_spawn = _keyword(object_cfg, "spawn")
    assert "cake_z_axix.usd" in ast.unparse(_keyword(object_spawn, "usd_path"))
    object_init = _keyword(object_cfg, "init_state")
    assert ast.literal_eval(_keyword(object_init, "pos")) == (0.55, 0.20, 0.335)
    assert ast.literal_eval(_keyword(object_init, "rot")) == (1.0, 0.0, 0.0, 0.0)

    reset_params = _keyword(events["reset_object"], "params")
    pose_range = ast.literal_eval(_dict_value(reset_params, "pose_range"))
    assert pose_range == {"x": [0.0, 0.0], "y": [0.0, 0.0], "yaw": [0.0, 0.0]}
    assert ast.literal_eval(_keyword(events["anchor_cake_z_axis_joint"], "mode")) == "prestartup"


def test_achieved_yaw_goal_is_immediately_resampled() -> None:
    module = _load_commands_module()

    class Stepper:
        length = 2

        def __init__(self):
            self.step = torch.tensor([0, 1, 1])

        def advance(self, env_ids):
            self.step[env_ids] += 1

        def current_object_pose(self, env_ids):
            result = torch.zeros(env_ids.numel(), 7)
            result[:, 3] = 1.0
            return result

    class Correction:
        def __init__(self):
            self.cleared = []

        def clear_stall(self, env_ids):
            self.cleared.append(env_ids.clone())

    command = object.__new__(module.CupcakeOnPlateTrajectoryObjectAndHandBasePoseCommand)
    command._stepper = Stepper()
    command._corr = Correction()
    command.pose_command_b = torch.zeros(3, 7)
    command._trajectory_command_achieved = torch.tensor([True, True, False])
    command.metrics = {"trajectory_command_achieved": torch.ones(3)}
    command.resampled = []
    command.corrected = []
    command.hand_base_updated = []

    command._update_command()

    assert torch.equal(command._stepper.step, torch.ones(3, dtype=torch.long))
    assert len(command.resampled) == 1
    assert torch.equal(command.resampled[0], torch.tensor([1]))
    assert not command._trajectory_command_achieved[:2].any()
    assert torch.equal(command.metrics["trajectory_command_achieved"], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(command.corrected[0], torch.tensor([0, 1, 2]))
    assert torch.equal(command.hand_base_updated[0], torch.tensor([0, 1, 2]))


def test_prestartup_joint_anchor_uses_each_cloned_world_pose() -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root_paths = []
    expected_positions = ((0.55, 0.20, 0.335), (3.55, 0.20, 0.335))
    for env_index, position in enumerate(expected_positions):
        root_path = f"/World/envs/env_{env_index}/Object"
        root = stage.DefinePrim(root_path, "Xform")
        root.GetReferences().AddReference(str(ASSET_PATH))
        UsdGeom.XformCommonAPI(root).SetTranslate(position)
        root_paths.append(root_path)

    module_names = ("isaaclab", "isaaclab.sim", "isaaclab.sim.utils", "isaaclab.sim.utils.stage")
    previous = {name: sys.modules.get(name) for name in module_names}
    isaaclab = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    sim.find_matching_prim_paths = lambda _: root_paths
    sim_utils = types.ModuleType("isaaclab.sim.utils")
    stage_utils = types.ModuleType("isaaclab.sim.utils.stage")
    stage_utils.get_current_stage = lambda: stage
    sys.modules.update(
        {
            "isaaclab": isaaclab,
            "isaaclab.sim": sim,
            "isaaclab.sim.utils": sim_utils,
            "isaaclab.sim.utils.stage": stage_utils,
        }
    )
    spec = importlib.util.spec_from_file_location("cupcake_z_axis_task_mdps_under_test", TASK_MDPS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)

        asset = types.SimpleNamespace(cfg=types.SimpleNamespace(prim_path="/World/envs/env_.*/Object"))
        scene = {"object": asset}
        env = types.SimpleNamespace(scene=scene, num_envs=2)
        module.anchor_cake_z_axis_joint(env, None)
    finally:
        for name, prior in previous.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior

    for root_path, expected in zip(root_paths, expected_positions, strict=True):
        joint = UsdPhysics.RevoluteJoint.Get(stage, Sdf.Path(root_path).AppendChild("z_axis_joint"))
        assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx(expected)
        assert tuple(joint.GetLocalPos1Attr().Get()) == pytest.approx((0.0, 0.0, 0.0))
