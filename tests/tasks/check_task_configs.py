"""Snapshot resolved task configs under IsaacLab; compare snapshots after a refactor.

Run with --headless --output /tmp/before.json, then --compare /tmp/before.json.
Use --repo to inspect a separate checkout with the same checker.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    import warp  # noqa: F401 -- load installed Warp before Isaac Sim extensions
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--allow-hrl-cleanup", action="store_true", help="Compare against configs before removal of flat tasks, HRL rewards, curricula and trainers.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    app = AppLauncher(args).app
    try:
        import gymnasium as gym
        import src.tasks  # noqa: F401
        from isaaclab.managers import ObservationTermCfg

        result = {}
        for name, spec in gym.registry.items():
            entry = spec.kwargs.get("env_cfg_entry_point", "")
            if not isinstance(entry, str) or not entry.startswith("src.tasks."):
                continue
            # These two stale registrations are intentionally removed by the refactor.
            if name in ("Unscrew_OmniReset-v0", "Unscrew_OmniReset_Play-v0"):
                continue
            try:
                module, cls = entry.split(":")
                config = getattr(importlib.import_module(module), cls)()
                data = config.to_dict()
                data["scene_field_order"] = list(vars(config.scene))
                data["observation_term_order"] = {
                    name: [key for key, term in vars(group).items() if isinstance(term, ObservationTermCfg)]
                    for name, group in vars(config.observations).items() if group is not None
                }
                trainer_entry = spec.kwargs.get("rsl_rl_cfg_entry_point")
                if trainer_entry:
                    trainer_module, trainer_cls = trainer_entry.split(":")
                    data["trainer"] = getattr(importlib.import_module(trainer_module), trainer_cls)().to_dict()
                # Check independent instances: mutating nested config cannot leak.
                other = getattr(importlib.import_module(module), cls)()
                config.scene.num_envs = 7
                assert other.scene.num_envs == data["scene"]["num_envs"]
                config.scene.robot.init_state.joint_pos["__isolation_check__"] = 1.0
                assert "__isolation_check__" not in other.scene.robot.init_state.joint_pos
                for group_name, group in vars(config.observations).items():
                    if group is None:
                        continue
                    for term_name, term in vars(group).items():
                        if isinstance(term, ObservationTermCfg):
                            term.params["__isolation_check__"] = True
                            assert "__isolation_check__" not in getattr(
                                getattr(other.observations, group_name), term_name
                            ).params
                for section in ("observations", "actions", "events", "rewards", "terminations"):
                    value = data.get(section)
                    if isinstance(value, dict):
                        data[section] = list(value.items())
                result[name] = {"config": data}
            except AssertionError:
                raise
            except Exception as error:
                result[name] = {"error": f"{type(error).__name__}: {error}"}
            print("CONFIG", name, "ERROR" if "error" in result[name] else "OK", flush=True)
        # Config serializers express callables as module paths. Keep those in snapshots
        # so an intentional move must be accounted for when comparing.
        payload = json.dumps(result, indent=2, default=str).replace(str(args.repo.resolve()), "{REPO}")
        if args.output:
            args.output.write_text(payload + "\n")
        if args.compare:
            baseline = args.compare.read_text()
            moves = {
                "src.tasks.common.mdps.rewards:contacts": "src.tasks.common.mdps.contacts:contacts",
                "src.tasks.rotate_knob.mdps.task_mdps:anchor_object_z_axis_joint": "src.tasks.common.mdps.events:anchor_object_z_axis_joint",
                "src.tasks.clean_table.mdps.task_mdps:": "src.tasks.common.mdps.placement:",
                "src.tasks.clean_table_omnireset.mdps.task_mdps:object_outside_table": "src.tasks.common.mdps.terminations:object_outside_table",
                "src.tasks.pick_insert_omnireset.mdps.task_mdps:object_outside_table": "src.tasks.common.mdps.terminations:object_outside_table",
            }
            # Only the three contact wrappers moved out of reorientation observations.
            for name in ("tip_contact_mask_obs", "tip_contact_force_mag_obs", "tip_contact_pose_flat"):
                baseline = baseline.replace(
                    "src.tasks.reorient.mdps.observations:" + name,
                    "src.tasks.common.mdps.observations:" + name,
                )
            for old, new in moves.items():
                baseline = baseline.replace(old, new)
            before = json.loads(baseline)
            after = json.loads(payload)
            assert all("error" not in value for value in before.values()), "Baseline contains configuration errors."
            assert all("error" not in value for value in after.values()), "Current snapshot contains configuration errors."
            if args.allow_hrl_cleanup:
                removed = {
                    "Pick_AnyRotate-v0", "Pick_AnyRotate_Play-v0", "Pick_Lift-v0", "Pick_Lift_Play-v0",
                    "Pick_Insert-v0", "Unscrew-v0", "Clean_Table-v0", "Rotate_Knob-v0",
                }
                assert before.keys() - after.keys() == removed
                for name in removed:
                    del before[name]
                for name, value in after.items():
                    if name.endswith("_HRL-v0"):
                        assert value["config"]["rewards"] is None
                        assert value["config"]["curriculum"] is None
                        assert "trainer" not in value["config"]
                        for section in ("rewards", "curriculum", "trainer"):
                            before[name]["config"].pop(section, None)
                            value["config"].pop(section, None)
            assert before == after, "Task configs differ; inspect the two JSON snapshots."
        assert all("error" not in value for value in result.values()), "Task configuration failed."
        print("CONFIG SUMMARY", len(result), "tasks,", sum("error" in v for v in result.values()), "errors", flush=True)
    except BaseException:
        import os
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)  # Kit shutdown can otherwise hide exceptions with a zero exit status.
    finally:
        app.close()


if __name__ == "__main__":
    main()
