import os
import pickle
from typing import Any
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
