#!/usr/bin/env python

"""Convert a URDF (with mesh parts) into a single USD file for direct use in scenes.

This is a thin wrapper around Isaac Lab's ``UrdfConverter`` that:

- Loads a URDF describing an object made of multiple mesh parts (OBJ/STL/FBX, etc.).
- Runs the URDF → USD conversion once.
- Prints the absolute path to the generated USD file.

Typical usage (from the D2H workspace root):

.. code-block:: bash

    python scripts/merge_obj_parts_to_usd.py \\
        --urdf_path /absolute/path/to/object.urdf \\
        --collider_type convex_decomposition
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a URDF (with mesh parts) into a single USD file using Isaac Lab's UrdfConverter."
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        required=True,
        help="Path to the URDF file describing the object. Can be absolute or relative.",
    )
    parser.add_argument(
        "--usd_dir",
        type=str,
        default=None,
        help="Optional output directory for the generated USD. "
        "If omitted, a temporary directory under /tmp/IsaacLab will be used.",
    )
    parser.add_argument(
        "--usd_file_name",
        type=str,
        default=None,
        help="Optional name for the generated USD file (e.g. 'my_object.usd'). "
        "If omitted, it is inferred from the URDF filename.",
    )
    parser.add_argument(
        "--collider_type",
        type=str,
        choices=["convex_hull", "convex_decomposition"],
        default="convex_decomposition",
        help="Collision simplification type for generated colliders.",
    )
    parser.add_argument(
        "--collision_from_visuals",
        action="store_true",
        help="If set, generate collision meshes from the visual geometry in the URDF.",
    )

    # Append AppLauncher CLI args (e.g. --device, --headless, etc.)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run URDF → USD conversion."""
    # Resolve absolute paths
    urdf_path = os.path.abspath(args.urdf_path)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    usd_dir = os.path.abspath(args.usd_dir) if args.usd_dir is not None else None

    # Import after the app is launched
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    # Build converter configuration
    urdf_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=usd_dir,
        usd_file_name=args.usd_file_name,
        # Geometry / collision options
        collision_from_visuals=args.collision_from_visuals,
        collider_type=args.collider_type,
        # Reasonable defaults for a single rigid object
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        # Set joint_drive to None for rigid objects (no joints)
        joint_drive=None,
    )

    # Run conversion (work is done lazily in the converter)
    converter = UrdfConverter(urdf_cfg)
    # Print the resulting USD path so it can be copy-pasted into RigidObjectCfg/UsdFileCfg
    print(converter.usd_path)


if __name__ == "__main__":
    args_cli = parse_args()

    # Launch Omniverse / Isaac Sim app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        main(args_cli)
    finally:
        # Cleanly close the app
        simulation_app.close()

