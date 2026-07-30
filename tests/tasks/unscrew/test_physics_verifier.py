import importlib.util
import math
import sys
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[3] / "src/tasks/unscrew/mdps/physics_verifier.py"
SPEC = importlib.util.spec_from_file_location("unscrew_physics_verifier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
physics_verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = physics_verifier
SPEC.loader.exec_module(physics_verifier)


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
