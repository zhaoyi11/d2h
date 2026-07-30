import ast
import importlib.util
import math
import sys
import traceback
from dataclasses import replace
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


def test_physics_verifier_runner_prioritizes_worktree_root_already_later_on_sys_path():
    _, tree = _runner_source_and_tree()
    repo_root_assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "REPO_ROOT" for target in node.targets)
    )
    path_bootstrap = next(
        statement
        for statement in tree.body
        if any(
            isinstance(node, ast.Call) and _qualified_name(node.func) == "sys.path.insert"
            for node in ast.walk(statement)
        )
    )
    bootstrap = ast.Module(body=[repo_root_assignment, path_bootstrap], type_ignores=[])
    fake_sys = type("FakeSys", (), {})()
    worktree_root = str(SCRIPT_PATH.resolve().parents[1])
    fake_sys.path = ["/some/other/checkout", worktree_root, "/some/dependency"]

    exec(
        compile(bootstrap, filename=str(SCRIPT_PATH), mode="exec"),
        {"Path": Path, "sys": fake_sys, "__file__": str(SCRIPT_PATH)},
    )

    assert fake_sys.path[0] == worktree_root


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
    loop_source = ast.unparse(control_loops[0])
    assert "target_index <= twist_end_index" in loop_source
    assert "physics_verifier.accumulate_geometric_world_yaw" in loop_source
    assert "physics_verifier.accumulate_velocity_integrated_world_yaw" in loop_source
    assert "target_index == twist_end_index" in loop_source
    assert "twist_end_rise = current_pose_w[:, 2] - settled_pose_w[:, 2]" in loop_source
    assert "target_index >= extraction_start_index" in loop_source
    assert "ever_cleared |= clear" in loop_source
    assert "max_clearance = torch.maximum(max_clearance, final_clearance)" in loop_source


def test_physics_verifier_runner_footer_preserves_exit_status_after_close():
    _, tree = _runner_source_and_tree()
    footer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_main_and_close"
    )
    namespace = {"traceback": traceback}
    exec(compile(ast.Module(body=[footer], type_ignores=[]), str(SCRIPT_PATH), "exec"), namespace)
    run_and_close = namespace["_run_main_and_close"]

    for should_fail, expected_status in ((False, 0), (True, 1)):
        kit_app = type(
            "FakeKitApp",
            (),
            {
                "post_uncancellable_quit": lambda self, code: setattr(
                    self, "return_code", code
                )
            },
        )()
        kit_app.return_code = None
        app = type(
            "FakeApp",
            (),
            {
                "app": kit_app,
                "close": lambda self: setattr(self, "closed", True),
            },
        )()
        app.closed = False

        def fake_main():
            if should_fail:
                raise RuntimeError("forced failure")

        try:
            run_and_close(fake_main, app)
        except SystemExit as exc:
            status = exc.code
        else:
            raise AssertionError("Runner footer must terminate with an explicit status.")

        assert app.closed
        assert kit_app.return_code == expected_status
        assert status == expected_status


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
    assert handler_calls.index("simulation_app.app.post_uncancellable_quit") < handler_calls.index(
        "simulation_app.close"
    )
    delayed_exit = handler.body[-1]
    assert isinstance(delayed_exit, ast.Raise)
    assert isinstance(delayed_exit.exc, ast.Call)
    assert _qualified_name(delayed_exit.exc.func) == "SystemExit"
    assert ast.literal_eval(delayed_exit.exc.args[0]) == 1

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
        "expected_twist_rise": "expected_twist_rise",
        "yaw_tolerance": "YAW_TOLERANCE",
        "position_tolerance": "POSITION_TOLERANCE",
        "orientation_tolerance": "ORIENTATION_TOLERANCE",
        "lateral_tolerance": "LATERAL_TOLERANCE",
        "saturation_inconclusive_fraction": "SATURATION_INCONCLUSIVE_FRACTION",
    }

    summary_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call) and _qualified_name(node.func) == "physics_verifier.TrialSummary"
    )
    summary_fields = {keyword.arg for keyword in summary_call.keywords}
    assert {
        "geometric_yaw",
        "velocity_integrated_yaw",
        "twist_end_rise",
        "max_twist_position_error",
        "max_twist_orientation_error",
        "ever_cleared",
        "max_clearance",
    } <= summary_fields


def test_physics_verifier_runner_releases_normal_stage_without_stopping_sim():
    _, tree = _runner_source_and_tree()
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    simulation_try = next(node for node in main.body if isinstance(node, ast.Try) and node.finalbody)
    cleanup_source = "\n".join(ast.unparse(statement) for statement in simulation_try.finalbody)
    main_calls = {
        _qualified_name(node.func) for node in ast.walk(main) if isinstance(node, ast.Call)
    }

    assert "sim.stop" not in main_calls
    expected_order = ("scene = None", "sim.clear_all_callbacks()", "sim.clear_instance()")
    assert all(statement in cleanup_source for statement in expected_order)
    assert [cleanup_source.index(statement) for statement in expected_order] == sorted(
        cleanup_source.index(statement) for statement in expected_order
    )

    footer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_main_and_close"
    )
    footer_calls = {
        _qualified_name(node.func) for node in ast.walk(footer) if isinstance(node, ast.Call)
    }
    assert "simulation_app.close" in footer_calls


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


def test_velocity_integrated_world_yaw_integrates_world_angular_velocity_z():
    accumulated_yaw = torch.tensor([1.0, -2.0])
    angular_velocity_w = torch.tensor([[9.0, 8.0, 0.5], [7.0, 6.0, -1.0]])

    result = physics_verifier.accumulate_velocity_integrated_world_yaw(
        accumulated_yaw, angular_velocity_w, dt=0.2
    )

    torch.testing.assert_close(result, torch.tensor([1.1, -2.2]))


def test_geometric_world_yaw_accumulates_four_quarter_turns():
    angles = torch.arange(5, dtype=torch.float64) * (math.pi / 2.0)
    quaternions = torch.stack(
        (
            torch.cos(angles / 2.0),
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            torch.sin(angles / 2.0),
        ),
        dim=-1,
    )
    accumulated = torch.zeros((), dtype=torch.float64)
    for previous, current in zip(quaternions[:-1], quaternions[1:]):
        accumulated = physics_verifier.accumulate_geometric_world_yaw(
            accumulated, previous, current
        )

    torch.testing.assert_close(accumulated, torch.tensor(2.0 * math.pi, dtype=torch.float64))


def test_geometric_world_yaw_is_invariant_to_antipodal_quaternions():
    half_sqrt = math.sqrt(0.5)
    quaternions = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-half_sqrt, 0.0, 0.0, -half_sqrt],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    accumulated = torch.zeros(())
    for previous, current in zip(quaternions[:-1], quaternions[1:]):
        accumulated = physics_verifier.accumulate_geometric_world_yaw(
            accumulated, previous, current
        )

    torch.testing.assert_close(accumulated, torch.tensor(math.pi))


def test_geometric_world_yaw_does_not_count_non_z_rotation():
    half_sqrt = math.sqrt(0.5)
    accumulated = physics_verifier.accumulate_geometric_world_yaw(
        torch.zeros(()),
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor([half_sqrt, half_sqrt, 0.0, 0.0]),
    )

    torch.testing.assert_close(accumulated, torch.zeros(()), atol=1.0e-7, rtol=0.0)


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


EXPECTED_YAW = 6.0 * math.pi
EXPECTED_TWIST_RISE = 0.045


def _valid_helix_summary(**changes) -> physics_verifier.TrialSummary:
    summary = physics_verifier.TrialSummary(
        finite=True,
        held_clear=True,
        geometric_yaw=EXPECTED_YAW,
        velocity_integrated_yaw=123.0,
        twist_end_rise=EXPECTED_TWIST_RISE,
        max_twist_position_error=0.002,
        max_twist_orientation_error=0.10,
        ever_cleared=True,
        max_clearance=0.04,
        final_position_error=0.002,
        final_orientation_error=0.10,
        max_lateral_drift=0.003,
        force_saturation_fraction=0.10,
        torque_saturation_fraction=0.20,
        peak_force=5.0,
        peak_torque=0.1,
        final_clearance=0.04,
    )
    return replace(summary, **changes)


def _blocked_pull_summary(**changes) -> physics_verifier.TrialSummary:
    summary = _valid_helix_summary(
        held_clear=False,
        geometric_yaw=0.0,
        velocity_integrated_yaw=0.0,
        twist_end_rise=0.0,
        max_twist_position_error=0.02,
        max_twist_orientation_error=0.0,
        ever_cleared=False,
        max_clearance=-0.01,
        final_position_error=0.02,
        final_orientation_error=0.0,
        max_lateral_drift=0.001,
        force_saturation_fraction=0.95,
        torque_saturation_fraction=0.0,
        peak_force=10.0,
        peak_torque=0.0,
        final_clearance=-0.01,
    )
    return replace(summary, **changes)


def _classify(
    helix: physics_verifier.TrialSummary, straight_pull: physics_verifier.TrialSummary
) -> physics_verifier.VerificationResult:
    return physics_verifier.classify_verification(
        helix,
        straight_pull,
        expected_yaw=EXPECTED_YAW,
        expected_twist_rise=EXPECTED_TWIST_RISE,
    )


def test_classification_pass_uses_geometric_twist_and_blocked_pull():
    result = _classify(_valid_helix_summary(), _blocked_pull_summary())

    assert result is physics_verifier.VerificationResult.PASS


def test_classification_requires_each_geometric_twist_measurement():
    for field_name, failed_value in (
        ("geometric_yaw", EXPECTED_YAW - 0.36),
        ("twist_end_rise", EXPECTED_TWIST_RISE - 0.006),
        ("max_twist_position_error", 0.006),
        ("max_twist_orientation_error", 0.36),
    ):
        result = _classify(
            _valid_helix_summary(**{field_name: failed_value}), _blocked_pull_summary()
        )

        assert result is physics_verifier.VerificationResult.TRAJECTORY_INFEASIBLE


def test_classification_reports_transiently_cleared_pull_as_invalid_model():
    straight_pull = _blocked_pull_summary(
        ever_cleared=True,
        max_clearance=0.031,
        final_clearance=-0.01,
    )

    result = _classify(_valid_helix_summary(), straight_pull)

    assert result is physics_verifier.VerificationResult.INVALID_PHYSICS_MODEL


def test_classification_reports_partially_held_clear_pull_as_invalid_model():
    straight_pull = _blocked_pull_summary(
        held_clear=False,
        ever_cleared=True,
        max_clearance=0.04,
        final_clearance=0.04,
    )

    result = _classify(_valid_helix_summary(), straight_pull)

    assert result is physics_verifier.VerificationResult.INVALID_PHYSICS_MODEL


def test_classification_nonfinite_precedes_clear_pull_invalid_model():
    helix = _valid_helix_summary(
        finite=False,
        geometric_yaw=float("nan"),
        velocity_integrated_yaw=float("nan"),
        twist_end_rise=float("nan"),
        max_twist_position_error=float("nan"),
        max_twist_orientation_error=float("nan"),
    )
    straight_pull = _blocked_pull_summary(ever_cleared=True, max_clearance=0.04)

    result = _classify(helix, straight_pull)

    assert result is physics_verifier.VerificationResult.CONTROLLER_INCONCLUSIVE


def test_classification_reports_saturated_failed_trial_as_inconclusive():
    helix = _valid_helix_summary(
        held_clear=False,
        geometric_yaw=1.0,
        torque_saturation_fraction=0.95,
    )

    result = _classify(helix, _blocked_pull_summary())

    assert result is physics_verifier.VerificationResult.CONTROLLER_INCONCLUSIVE


def test_classification_reports_unsaturated_failed_trial_as_infeasible():
    helix = _valid_helix_summary(
        held_clear=False,
        geometric_yaw=1.0,
        force_saturation_fraction=0.20,
        torque_saturation_fraction=0.30,
    )

    result = _classify(helix, _blocked_pull_summary())

    assert result is physics_verifier.VerificationResult.TRAJECTORY_INFEASIBLE
