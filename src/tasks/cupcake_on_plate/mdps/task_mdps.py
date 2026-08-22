"""USD-stage setup terms for the Z-axis-constrained cupcake."""

from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import get_current_stage


def anchor_cake_z_axis_joint(
    env,
    env_ids,
    asset_name: str = "object",
    joint_name: str = "z_axis_joint",
) -> None:
    """Anchor every cloned cupcake joint at its spawned world pose before PhysX starts."""
    if env_ids is not None:
        raise ValueError("anchor_cake_z_axis_joint is a global pre-startup event.")

    asset = env.scene[asset_name]
    stage = get_current_stage()
    root_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    if len(root_paths) != env.num_envs:
        raise RuntimeError(
            f"Expected {env.num_envs} cupcake prims for {asset.cfg.prim_path!r}; found {len(root_paths)}."
        )

    xform_cache = UsdGeom.XformCache()
    for root_path in root_paths:
        root_prim = stage.GetPrimAtPath(root_path)
        joint_path = Sdf.Path(root_path).AppendChild(joint_name)
        joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)
        if not joint:
            raise RuntimeError(f"Missing cupcake revolute joint at {joint_path}.")
        if joint.GetAxisAttr().Get() != UsdGeom.Tokens.z:
            raise RuntimeError(f"Cupcake joint {joint_path} must rotate around Z.")
        if joint.GetBody0Rel().GetTargets():
            raise RuntimeError(f"Cupcake joint {joint_path} must connect body0 to the world.")
        if joint.GetBody1Rel().GetTargets() != [Sdf.Path(root_path)]:
            raise RuntimeError(f"Cupcake joint {joint_path} must connect body1 to {root_path}.")

        world_transform = xform_cache.GetLocalToWorldTransform(root_prim)
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


__all__ = ["anchor_cake_z_axis_joint"]
