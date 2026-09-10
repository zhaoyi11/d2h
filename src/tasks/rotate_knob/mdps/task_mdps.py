"""USD-stage setup terms for the Z-axis-constrained object."""

from __future__ import annotations

import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import math as math_utils


def trajectory_yaw_tracking(
    env,
    command_name: str = "object_pose",
    std: float = 0.5,
) -> torch.Tensor:
    """Reward the active yaw target without rewarding the initial reach stage."""
    command = env.command_manager.get_term(command_name)
    goal_quat_w = math_utils.quat_mul(command.robot.data.root_quat_w, command.pose_command_b[:, 3:7])
    yaw_error = math_utils.quat_error_magnitude(command.object.data.root_quat_w, goal_quat_w)
    tracking = 1.0 - torch.tanh(yaw_error / std)
    return tracking * command.metrics["yaw_target_active"]


def anchor_object_z_axis_joint(
    env,
    env_ids,
    asset_name: str = "object",
    joint_name: str = "z_axis_joint",
) -> None:
    """Create and anchor every cloned object's Z joint before PhysX starts."""
    if env_ids is not None:
        raise ValueError("anchor_object_z_axis_joint is a global pre-startup event.")

    asset = env.scene[asset_name]
    stage = get_current_stage()
    root_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    if len(root_paths) != env.num_envs:
        raise RuntimeError(
            f"Expected {env.num_envs} object prims for {asset.cfg.prim_path!r}; found {len(root_paths)}."
        )

    xform_cache = UsdGeom.XformCache()
    for root_path in root_paths:
        root_prim = stage.GetPrimAtPath(root_path)
        rigid_body_prims = [
            prim for prim in Usd.PrimRange(root_prim) if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(rigid_body_prims) != 1:
            raise RuntimeError(
                f"Expected one rigid body below sampled object {root_path}; found "
                f"{[str(prim.GetPath()) for prim in rigid_body_prims]}."
            )
        body_prim = rigid_body_prims[0]
        body_path = body_prim.GetPath()

        joint_path = Sdf.Path(root_path).AppendChild(joint_name)
        joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)
        if not joint:
            joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
            joint.CreateAxisAttr(UsdGeom.Tokens.z)
            joint.CreateBody0Rel()
            joint.CreateBody1Rel().SetTargets([body_path])
            joint.CreateLowerLimitAttr(float("-inf"))
            joint.CreateUpperLimitAttr(float("inf"))
        if joint.GetAxisAttr().Get() != UsdGeom.Tokens.z:
            raise RuntimeError(f"Object joint {joint_path} must rotate around Z.")
        if joint.GetBody0Rel().GetTargets():
            raise RuntimeError(f"Object joint {joint_path} must connect body0 to the world.")
        if joint.GetBody1Rel().GetTargets() != [body_path]:
            raise RuntimeError(f"Object joint {joint_path} must connect body1 to {body_path}.")

        world_transform = xform_cache.GetLocalToWorldTransform(body_prim)
        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*(float(value) for value in translation)))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(rotation.GetReal()),
                Gf.Vec3f(*(float(value) for value in imaginary)),
            )
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


__all__ = ["anchor_object_z_axis_joint", "trajectory_yaw_tracking"]
