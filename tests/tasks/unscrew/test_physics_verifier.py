import ast
import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[3] / "src/tasks/unscrew/mdps/physics_verifier.py"
SCRIPT_PATH = Path(__file__).parents[3] / "scripts/verify_unscrew_trajectory_physics.py"
SPEC = importlib.util.spec_from_file_location("unscrew_physics_verifier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
physics_verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = physics_verifier
SPEC.loader.exec_module(physics_verifier)


FORBIDDEN_STATE_MUTATION_CALLS = {
    # RigidObject root/link/COM writers in the installed IsaacLab version.
    "write_root_state_to_sim",
    "write_root_com_state_to_sim",
    "write_root_link_state_to_sim",
    "write_root_pose_to_sim",
    "write_root_link_pose_to_sim",
    "write_root_com_pose_to_sim",
    "write_root_velocity_to_sim",
    "write_root_com_velocity_to_sim",
    "write_root_link_velocity_to_sim",
    # Direct PhysX-view bypasses for rigid objects and articulation roots.
    "set_transforms",
    "set_velocities",
    "set_kinematic_targets",
    "set_root_transforms",
    "set_root_velocities",
    "set_root_kinematic_targets",
}


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _runner_source_and_tree() -> tuple[str, ast.Module]:
    source = SCRIPT_PATH.read_text()
    return source, ast.parse(source)


def test_physics_verifier_runner_source_contract():
    source, tree = _runner_source_and_tree()

    app_launcher_assignment = source.index("app_launcher = AppLauncher(args_cli)")
    repo_path_precedence = source.index("sys.path.insert(0, str(REPO_ROOT))")
    runtime_imports = (
        "import isaaclab.sim",
        "from isaaclab.scene import InteractiveScene",
        "from isaaclab.sim import SimulationContext",
    )
    assert all(app_launcher_assignment < source.index(statement) for statement in runtime_imports)
    assert all(repo_path_precedence < source.index(statement) for statement in runtime_imports)

    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attribute_calls = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
    assert {
        "build_unscrew_object_pose_sequence",
        "straight_pull_targets",
    } <= attribute_calls

    assert "SceneCfg(num_envs=2" in source
    assert "gravity=(0.0, 0.0, -9.81)" in source
    assert "targets_w[1] = physics_verifier.straight_pull_targets" in source


def test_physics_verifier_runner_forbids_state_mutation_bypasses():
    _, tree = _runner_source_and_tree()
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert called_attributes.isdisjoint(FORBIDDEN_STATE_MUTATION_CALLS)


def test_physics_verifier_runner_control_loop_applies_wrench_then_steps_physics():
    _, tree = _runner_source_and_tree()
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    control_loops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "target_index"
    ]
    assert len(control_loops) == 1

    call_names = [
        _qualified_name(node.func)
        for node in sorted(
            (node for node in ast.walk(control_loops[0]) if isinstance(node, ast.Call)),
            key=lambda node: (node.lineno, node.col_offset),
        )
    ]
    expected_order = (
        "obj.set_external_force_and_torque",
        "scene.write_data_to_sim",
        "sim.step",
        "scene.update",
    )
    assert all(call_names.count(name) == 1 for name in expected_order)
    assert [call_names.index(name) for name in expected_order] == sorted(
        call_names.index(name) for name in expected_order
    )


def test_physics_verifier_runner_closes_app_on_delayed_import_failure_and_module_import():
    _, tree = _runner_source_and_tree()
    delayed_import_try = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.Try)
            and any(
                isinstance(child, ast.Import) and any(alias.name == "isaaclab.sim" for alias in child.names)
                for child in node.body
            )
        ),
        None,
    )
    assert delayed_import_try is not None
    assert len(delayed_import_try.handlers) == 1
    handler = delayed_import_try.handlers[0]
    assert isinstance(handler.type, ast.Name) and handler.type.id == "BaseException"
    handler_calls = [
        _qualified_name(node.func)
        for statement in handler.body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
    ]
    assert handler_calls.index("traceback.print_exc") < handler_calls.index("simulation_app.close")
    assert isinstance(handler.body[-1], ast.Raise)

    main_guard = next(
        node
        for node in tree.body
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"
    )
    assert any(
        isinstance(node, ast.Call) and _qualified_name(node.func) == "simulation_app.close"
        for statement in main_guard.orelse
        for node in ast.walk(statement)
    )


def test_physics_verifier_runner_logs_and_uses_reproducible_configuration():
    _, tree = _runner_source_and_tree()
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    printed_config = "\n".join(
        ast.unparse(node)
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print"
    )
    required_logged_values = {
        "MAX_FORCE",
        "MAX_TORQUE",
        "POSITION_STIFFNESS",
        "POSITION_DAMPING",
        "ROTATION_STIFFNESS",
        "ROTATION_DAMPING",
        "CLEARANCE_MARGIN",
        "trajectory.DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE",
        "trajectory.DEFAULT_UNSCREW_THREAD_PITCH",
        "YAW_TOLERANCE",
        "POSITION_TOLERANCE",
        "ORIENTATION_TOLERANCE",
        "LATERAL_TOLERANCE",
        "SATURATION_INCONCLUSIVE_FRACTION",
    }
    assert all(name in printed_config for name in required_logged_values)
    assert "OBJECT_MASS_KG" in printed_config
    assert "masses.tolist()" in printed_config

    classify_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and _qualified_name(node.func) == "physics_verifier.classify_verification"
    )
    actual_keywords = {keyword.arg: ast.unparse(keyword.value) for keyword in classify_call.keywords}
    assert actual_keywords == {
        "expected_yaw": "trajectory.DEFAULT_UNSCREW_TWIST_TOTAL_ANGLE",
        "yaw_tolerance": "YAW_TOLERANCE",
        "position_tolerance": "POSITION_TOLERANCE",
        "orientation_tolerance": "ORIENTATION_TOLERANCE",
        "lateral_tolerance": "LATERAL_TOLERANCE",
        "saturation_inconclusive_fraction": "SATURATION_INCONCLUSIVE_FRACTION",
    }


def test_bounded_pd_wrench_tracks_pose_and_clamps_vector_norms():
    current_pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    target_pose = torch.tensor(
        [[3.0, 4.0, 0.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]
    )

    command = physics_verifier.bounded_pd_wrench(
        current_pose,
        target_pose,
        linear_velocity_w=torch.tensor([[1.0, 0.0, 0.0]]),
        angular_velocity_w=torch.tensor([[0.0, 0.0, 0.5]]),
        mass=torch.tensor([2.0]),
        gravity_w=torch.tensor([0.0, 0.0, -10.0]),
        position_stiffness=2.0,
        position_damping=1.0,
        rotation_stiffness=2.0,
        rotation_damping=1.0,
        max_force=5.0,
        max_torque=1.0,
    )

    raw_force = torch.tensor([[5.0, 8.0, 20.0]])
    expected_force = raw_force * (5.0 / torch.linalg.vector_norm(raw_force, dim=-1, keepdim=True))
    torch.testing.assert_close(command.force_w, expected_force)
    torch.testing.assert_close(command.torque_w, torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(torch.linalg.vector_norm(command.force_w, dim=-1), torch.tensor([5.0]))
    torch.testing.assert_close(torch.linalg.vector_norm(command.torque_w, dim=-1), torch.tensor([1.0]))
    assert torch.equal(command.force_saturated, torch.tensor([True]))
    assert torch.equal(command.torque_saturated, torch.tensor([True]))


def test_bounded_pd_wrench_composes_world_rotation_error_with_nonidentity_current():
    half_sqrt = math.sqrt(0.5)
    current_pose = torch.tensor([[0.0, 0.0, 0.0, half_sqrt, half_sqrt, 0.0, 0.0]])
    # target = q_world_z_90 * q_current_x_90 in wxyz convention.
    target_pose = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]])

    command = physics_verifier.bounded_pd_wrench(
        current_pose,
        target_pose,
        linear_velocity_w=torch.zeros(1, 3),
        angular_velocity_w=torch.zeros(1, 3),
        mass=1.0,
        gravity_w=torch.zeros(3),
        position_stiffness=1.0,
        position_damping=0.0,
        rotation_stiffness=1.0,
        rotation_damping=0.0,
        max_force=1.0,
        max_torque=10.0,
    )

    torch.testing.assert_close(command.torque_w, torch.tensor([[0.0, 0.0, math.pi / 2.0]]))


def test_bounded_pd_wrench_treats_antipodal_quaternions_as_same_orientation():
    half_sqrt = math.sqrt(0.5)
    current_pose = torch.tensor([[0.0, 0.0, 0.0, half_sqrt, 0.0, 0.0, half_sqrt]])
    target_pose = current_pose.clone()
    target_pose[:, 3:7] *= -1.0

    command = physics_verifier.bounded_pd_wrench(
        current_pose,
        target_pose,
        linear_velocity_w=torch.zeros(1, 3),
        angular_velocity_w=torch.zeros(1, 3),
        mass=1.0,
        gravity_w=torch.zeros(3),
        position_stiffness=1.0,
        position_damping=0.0,
        rotation_stiffness=1.0,
        rotation_damping=0.0,
        max_force=1.0,
        max_torque=1.0,
    )

    torch.testing.assert_close(command.torque_w, torch.zeros(1, 3), atol=1.0e-7, rtol=0.0)
    assert not command.torque_saturated.item()


def test_straight_pull_targets_retain_xyz_and_initial_orientation():
    half_sqrt = math.sqrt(0.5)
    helical_targets = torch.tensor(
        [
            [1.0, 2.0, 3.0, half_sqrt, 0.0, 0.0, half_sqrt],
            [1.0, 2.0, 4.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 5.0, -half_sqrt, 0.0, 0.0, half_sqrt],
        ]
    )

    straight_targets = physics_verifier.straight_pull_targets(helical_targets)

    torch.testing.assert_close(straight_targets[:, :3], helical_targets[:, :3])
    torch.testing.assert_close(
        straight_targets[:, 3:7], helical_targets[0, 3:7].expand(helical_targets.shape[0], -1)
    )
    assert straight_targets.data_ptr() != helical_targets.data_ptr()


def test_straight_pull_targets_keep_each_batched_trajectory_initial_orientation():
    half_sqrt = math.sqrt(0.5)
    helical_targets = torch.tensor(
        [
            [
                [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 4.0, half_sqrt, 0.0, 0.0, half_sqrt],
            ],
            [
                [4.0, 5.0, 6.0, half_sqrt, 0.0, 0.0, half_sqrt],
                [4.0, 5.0, 7.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    straight_targets = physics_verifier.straight_pull_targets(helical_targets)

    torch.testing.assert_close(straight_targets[..., :3], helical_targets[..., :3])
    torch.testing.assert_close(
        straight_targets[..., 3:7], helical_targets[..., :1, 3:7].expand(-1, 2, -1)
    )


def test_accumulated_world_yaw_integrates_world_angular_velocity_z():
    accumulated_yaw = torch.tensor([1.0, -2.0])
    angular_velocity_w = torch.tensor([[9.0, 8.0, 0.5], [7.0, 6.0, -1.0]])

    result = physics_verifier.accumulate_world_yaw(accumulated_yaw, angular_velocity_w, dt=0.2)

    torch.testing.assert_close(result, torch.tensor([1.1, -2.2]))


def test_bolt_bottom_clearance_uses_rotated_aabb_corners():
    corners = torch.tensor(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-2.0, 2.0) for z in (-0.25, 0.25)]
    )
    half_sqrt = math.sqrt(0.5)

    clearance = physics_verifier.bolt_bottom_clearance(
        object_pos_r=torch.tensor([[0.0, 0.0, 3.0]]),
        object_quat_r=torch.tensor([[half_sqrt, 0.0, half_sqrt, 0.0]]),
        bolt_corners=corners,
        socket_top_z=torch.tensor([1.5]),
    )

    # A +90 degree Y rotation maps local x to -Z, so the lowest corner is at z=3-1.
    torch.testing.assert_close(clearance, torch.tensor([0.5]))


def test_classification_pass_requires_helical_clear_and_pull_blocked():
    helix = physics_verifier.TrialSummary(
        finite=True,
        held_clear=True,
        accumulated_yaw=6.0 * math.pi,
        final_position_error=0.002,
        final_orientation_error=0.10,
        max_lateral_drift=0.003,
        force_saturation_fraction=0.10,
        torque_saturation_fraction=0.20,
        peak_force=5.0,
        peak_torque=0.1,
        final_clearance=0.01,
    )
    straight_pull = physics_verifier.TrialSummary(
        finite=True,
        held_clear=False,
        accumulated_yaw=0.0,
        final_position_error=0.02,
        final_orientation_error=0.0,
        max_lateral_drift=0.001,
        force_saturation_fraction=0.95,
        torque_saturation_fraction=0.0,
        peak_force=10.0,
        peak_torque=0.0,
        final_clearance=-0.01,
    )

    result = physics_verifier.classify_verification(
        helix, straight_pull, expected_yaw=6.0 * math.pi
    )

    assert result is physics_verifier.VerificationResult.PASS


def test_classification_reports_invalid_collision_model_if_pull_clears():
    helix = physics_verifier.TrialSummary(
        finite=True,
        held_clear=True,
        accumulated_yaw=6.0 * math.pi,
        final_position_error=0.0,
        final_orientation_error=0.0,
        max_lateral_drift=0.0,
        force_saturation_fraction=0.0,
        torque_saturation_fraction=0.0,
        peak_force=1.0,
        peak_torque=0.1,
        final_clearance=0.01,
    )
    straight_pull = physics_verifier.TrialSummary(
        finite=True,
        held_clear=True,
        accumulated_yaw=0.0,
        final_position_error=0.0,
        final_orientation_error=0.0,
        max_lateral_drift=0.0,
        force_saturation_fraction=0.0,
        torque_saturation_fraction=0.0,
        peak_force=1.0,
        peak_torque=0.0,
        final_clearance=0.01,
    )

    result = physics_verifier.classify_verification(
        helix, straight_pull, expected_yaw=6.0 * math.pi
    )

    assert result is physics_verifier.VerificationResult.INVALID_PHYSICS_MODEL


def test_classification_nonfinite_precedes_clear_pull_invalid_model():
    helix = physics_verifier.TrialSummary(
        finite=False,
        held_clear=True,
        accumulated_yaw=float("nan"),
        final_position_error=float("nan"),
        final_orientation_error=float("nan"),
        max_lateral_drift=float("nan"),
        force_saturation_fraction=0.0,
        torque_saturation_fraction=0.0,
        peak_force=float("nan"),
        peak_torque=float("nan"),
        final_clearance=float("nan"),
    )
    straight_pull = physics_verifier.TrialSummary(
        finite=True,
        held_clear=True,
        accumulated_yaw=0.0,
        final_position_error=0.0,
        final_orientation_error=0.0,
        max_lateral_drift=0.0,
        force_saturation_fraction=0.0,
        torque_saturation_fraction=0.0,
        peak_force=1.0,
        peak_torque=0.0,
        final_clearance=0.01,
    )

    result = physics_verifier.classify_verification(helix, straight_pull, expected_yaw=0.0)

    assert result is physics_verifier.VerificationResult.CONTROLLER_INCONCLUSIVE


def test_classification_reports_saturated_failed_trial_as_inconclusive():
    helix = physics_verifier.TrialSummary(
        finite=True,
        held_clear=False,
        accumulated_yaw=1.0,
        final_position_error=0.02,
        final_orientation_error=0.5,
        max_lateral_drift=0.01,
        force_saturation_fraction=0.20,
        torque_saturation_fraction=0.95,
        peak_force=10.0,
        peak_torque=0.2,
        final_clearance=-0.01,
    )
    straight_pull = physics_verifier.TrialSummary(
        finite=True,
        held_clear=False,
        accumulated_yaw=0.0,
        final_position_error=0.02,
        final_orientation_error=0.0,
        max_lateral_drift=0.0,
        force_saturation_fraction=0.95,
        torque_saturation_fraction=0.0,
        peak_force=10.0,
        peak_torque=0.0,
        final_clearance=-0.01,
    )

    result = physics_verifier.classify_verification(
        helix, straight_pull, expected_yaw=6.0 * math.pi
    )

    assert result is physics_verifier.VerificationResult.CONTROLLER_INCONCLUSIVE


def test_classification_reports_unsaturated_failed_trial_as_infeasible():
    helix = physics_verifier.TrialSummary(
        finite=True,
        held_clear=False,
        accumulated_yaw=1.0,
        final_position_error=0.02,
        final_orientation_error=0.5,
        max_lateral_drift=0.01,
        force_saturation_fraction=0.20,
        torque_saturation_fraction=0.30,
        peak_force=5.0,
        peak_torque=0.1,
        final_clearance=-0.01,
    )
    straight_pull = physics_verifier.TrialSummary(
        finite=True,
        held_clear=False,
        accumulated_yaw=0.0,
        final_position_error=0.02,
        final_orientation_error=0.0,
        max_lateral_drift=0.0,
        force_saturation_fraction=0.95,
        torque_saturation_fraction=0.0,
        peak_force=10.0,
        peak_torque=0.0,
        final_clearance=-0.01,
    )

    result = physics_verifier.classify_verification(
        helix, straight_pull, expected_yaw=6.0 * math.pi
    )

    assert result is physics_verifier.VerificationResult.TRAJECTORY_INFEASIBLE
