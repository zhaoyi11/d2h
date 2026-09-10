"""Run with env_isaaclab: python tests/deployment/check_simulation.py --headless."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from isaaclab.app import AppLauncher

launcher = AppLauncher(headless="--headless" in sys.argv, device="cpu")

try:
    import torch
    from scripts.deployment.simulation import SimMirror
    from scripts.deployment.deploy_knob_student import observation_frame
    from src.tasks.rotate_knob_distill.mdps.task_mdps import student_frame

    sim = SimMirror()
    try:
        assert abs(sim.env.step_dt - 1 / 60) < 1e-8
        for angle, target in ((0.4, -0.8), (-1.0, 0.7)):
            sim.reset(torch.zeros(16), angle, target)
            q, actual_angle, velocity = sim.read()
            assert abs(actual_angle - angle) < 1e-5
            torch.testing.assert_close(q, torch.zeros(16), atol=1e-5, rtol=0)
            frame = observation_frame(q, q, actual_angle, velocity, target)
            torch.testing.assert_close(frame, student_frame(sim.env).cpu(), atol=1e-5, rtol=1e-5)
            applied = torch.full((16,), 0.01)
            for _ in range(4):
                q, actual_angle, velocity = sim.step(applied)
            torch.testing.assert_close(sim.action._prev_applied_actions.cpu(), applied[None])
            torch.testing.assert_close(sim.robot.data.joint_pos_target[:, sim.joint_ids].cpu(), applied[None])
            assert torch.isfinite(q).all()
            frame = observation_frame(q, applied, actual_angle, velocity, target)
            torch.testing.assert_close(frame, student_frame(sim.env).cpu(), atol=1e-5, rtol=1e-5)
        print("PASS: simulation reset, 60 Hz timing, target equality, and student observation parity", flush=True)
    finally:
        sim.disconnect()
except BaseException:
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    launcher.app.close()
