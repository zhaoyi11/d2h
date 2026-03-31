import os
import pickle
import inspect
from typing import Any
from dataclasses import dataclass
import numpy as np
import torch


def load_pickle(filename: str) -> Any:
    """Loads an input PKL file safely.

    Args:
        filename: The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(filename: str, data: Any):
    """Saves data into a pickle file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename: The path to save the file at.
        data: The data to save.
    """
    # check ending
    if not filename.endswith("pkl"):
        filename += ".pkl"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def apply_grasp_overrides_from_cli(env_cfg: Any, args_cli: Any) -> bool:
    """Apply CLI grasp overrides to configs that expose the current grasp-path schema.

    Returns ``True`` when the env config supports ``grasp_path`` and the caller
    should stop checking legacy schemas.
    """
    if not hasattr(env_cfg, "grasp_path"):
        return False

    should_reapply = False

    if getattr(args_cli, "grasp_path", None) is not None:
        env_cfg.grasp_path = args_cli.grasp_path
        if hasattr(env_cfg, "grasp_cache_path"):
            env_cfg.grasp_cache_path = args_cli.grasp_path
        should_reapply = True

    if getattr(args_cli, "obj_urdf_path", None) is not None:
        env_cfg.object_urdf_path = args_cli.obj_urdf_path
        should_reapply = True

    if getattr(args_cli, "obj_scale", None) is not None:
        env_cfg.object_scale_override = args_cli.obj_scale
        should_reapply = True

    post_init = getattr(env_cfg, "__post_init__", None)
    if should_reapply and callable(post_init):
        post_init()

    return True


def patch_startup_class_event_terms(env_cfg: Any) -> bool:
    """Wrap class-based startup events so they instantiate lazily on first use.

    This keeps manager configs compatible with IsaacLab runtimes where startup
    events can be applied before class-based event terms are instantiated.
    """
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is None:
        return False

    if isinstance(events_cfg, dict):
        cfg_items = events_cfg.items()
    else:
        cfg_items = events_cfg.__dict__.items()

    patched = False

    for _, term_cfg in cfg_items:
        if term_cfg is None or getattr(term_cfg, "mode", None) != "startup":
            continue

        term_cls = getattr(term_cfg, "func", None)
        if not inspect.isclass(term_cls):
            continue

        cached_instance = None

        def _wrapped(env, env_ids, *, _term_cls=term_cls, _fallback_cfg=term_cfg, **kwargs):
            nonlocal cached_instance
            if cached_instance is None:
                runtime_cfg = _fallback_cfg
                event_manager = getattr(env, "event_manager", None)
                if event_manager is not None:
                    for mode_cfgs in getattr(event_manager, "_mode_term_cfgs", {}).values():
                        for candidate_cfg in mode_cfgs:
                            if getattr(candidate_cfg, "func", None) is _wrapped:
                                runtime_cfg = candidate_cfg
                                break
                        else:
                            continue
                        break
                cached_instance = _term_cls(cfg=runtime_cfg, env=env)
            return cached_instance(env, env_ids, **kwargs)

        term_cfg.func = _wrapped
        patched = True

    return patched


def _safe_wandb_scalar(value):
    """Convert logger values to stable python scalars before passing to wandb."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().float().mean().cpu().item())
    try:
        return float(value)
    except Exception:
        return 0.0


def _patch_rsl_wandb_writer():
    """Patch rsl-rl wandb writer to avoid crashes from unsupported payloads."""
    os.environ.setdefault("WANDB_CONSOLE", "off")

    try:
        import wandb
        from rsl_rl.utils import wandb_utils
    except Exception as exc:
        print(f"[WARN] Failed to set up wandb compatibility patch: {exc}")
        return

    writer_cls = wandb_utils.WandbSummaryWriter

    if getattr(writer_cls, "_safe_patch", False):
        return

    original_add_scalar = writer_cls.add_scalar

    def _patched_add_scalar(
        self,
        tag,
        scalar_value,
        global_step=None,
        walltime=None,
        new_style=False,
    ):
        return original_add_scalar(
            self,
            tag,
            _safe_wandb_scalar(scalar_value),
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )

    writer_cls.add_scalar = _patched_add_scalar
    writer_cls._safe_patch = True


# Load grasp data
@dataclass
class GraspInitData:
    """Precomputed grasp initialization data."""
    robot_root_state: np.ndarray
    robot_joint_pos: np.ndarray
    robot_joint_vel: np.ndarray
    object_root_state: np.ndarray
    object_asset_path: str|None
    object_scale: float|None


def load_grasp_data(grasp_path: str) -> GraspInitData:
    """Load grasp data and sample a random grasp for each environment."""
    grasp_data = np.load(grasp_path, allow_pickle=True).item()

    robot_root_state = np.array(grasp_data["robot_root_state"])
    robot_joint_pos = np.array(grasp_data["robot_joint_pos"])
    robot_joint_vel = np.array(grasp_data["robot_joint_vel"])
    object_root_state = np.array(grasp_data["object_root_state"])

    return GraspInitData(
        robot_root_state=robot_root_state,
        robot_joint_pos=robot_joint_pos,
        robot_joint_vel=robot_joint_vel,
        object_root_state=object_root_state,
        object_asset_path=None,
        object_scale=None,
    )
