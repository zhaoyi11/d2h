"""Roll out a trained RSL-RL policy and save episode trajectories to disk.

Launcher + runner setup mirrors ``scripts/rsl_rl/play.py``. Per-episode disk
layout mirrors ``FlashSAC_dex/flash_rl/buffers.py::ReplayBufferStorage``:
one compressed ``.npz`` per episode, named ``{idx:010d}.npz``, written
asynchronously from a bounded executor.

Each saved episode contains:
    observation.<group>   (N+1, *obs_dim)   float32   — one row per saved obs group
    action                (N, act_dim)      float32
    reward                (N,)              float32
    terminated            (N,)              bool
    truncated             (N,)              bool
    success               ()                bool      — optional, when detectable

The trailing row on ``observation.<group>`` is the post-reset obs from the
*next* episode, not the true pre-reset terminal (IsaacLab auto-resets
inside ``step()``). Downstream loaders should use rows ``[:N]`` for
learning and ignore the trailing row. This matches the compromise made in
``src/flash_sac/env_adapter.py`` and ``flash_rl/buffers.py``.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import argparse
import io
import json
import pathlib
import subprocess
import sys
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any

from isaaclab.app import AppLauncher

# ``scripts/rsl_rl/cli_args.py`` is not on sys.path when this script is run as
# ``python -m src.policy.data_collection``; add it explicitly so we can reuse
# the same rsl_rl CLI flags used by play.py/train.py.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "rsl_rl"))
import cli_args  # noqa: E402  isort: skip

parser = argparse.ArgumentParser(description="Collect RL policy rollout trajectories.")
# --- mirrored play.py flags ---
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--grasp_path", type=str, default=None, help="Path to grasp data file (.npy).")
parser.add_argument("--obj_urdf_path", type=str, default=None, help="Path to object URDF.")
parser.add_argument("--obj_scale", type=float, default=None, help="Override object scale from grasp data.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_last_checkpoint", action="store_true", help="When no checkpoint provided, use the last saved model."
)
# --- data collection flags ---
parser.add_argument("--num_episodes", type=int, default=1000, help="Stop after this many completed episodes.")
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Directory to write {idx:010d}.npz files into. Defaults to datasets/<task>/<timestamp>/.",
)
parser.add_argument(
    "--obs_groups",
    type=str,
    default="policy,proprio,perception",
    help="Comma-separated observation group names to save (each as its own array in the npz).",
)
parser.add_argument(
    "--deterministic",
    dest="deterministic",
    action="store_true",
    default=True,
    help="Use policy_nn.act_inference(obs) (mean action). Default.",
)
parser.add_argument(
    "--stochastic",
    dest="deterministic",
    action="store_false",
    help="Sample actions from the stochastic policy instead.",
)
parser.add_argument(
    "--success_key",
    type=str,
    default=None,
    help="Dot-path on env.unwrapped producing a per-env bool/float success tensor.",
)
parser.add_argument(
    "--max_in_flight_writes",
    type=int,
    default=8,
    help="Maximum number of compressed-write jobs allowed to be queued at once.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym  # noqa: E402
import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import DistillationRunner, OnPolicyRunner  # noqa: E402

from isaaclab.envs import (  # noqa: E402
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import src.tasks  # noqa: E402, F401 — registers custom gym IDs
from src.utils import apply_grasp_overrides_from_cli  # noqa: E402


# --------------------------------------------------------------------------- #
# EpisodeWriter
# --------------------------------------------------------------------------- #


def _save_episode(episode: dict[str, np.ndarray], path: pathlib.Path) -> None:
    """Compress and write a single episode atomically."""
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with path.open("wb") as f:
            f.write(bs.read())


def _save_and_release(
    episode: dict[str, np.ndarray],
    path: pathlib.Path,
    sem: "threading.BoundedSemaphore",
) -> None:
    try:
        _save_episode(episode, path)
    finally:
        sem.release()


def _episode_path(replay_dir: pathlib.Path, eps_idx: int) -> pathlib.Path:
    return replay_dir / f"{eps_idx:010d}.npz"


class EpisodeWriter:
    """Per-env accumulator that writes one compressed ``.npz`` per episode.

    Modelled on :class:`flash_rl.buffers.ReplayBufferStorage` but stripped of
    n-step / ring-buffer / preload machinery — we are producing an offline
    dataset, not a training replay. Writes are dispatched to a small
    :class:`ThreadPoolExecutor` and gated by a
    :class:`threading.BoundedSemaphore` so a synchronized reset storm
    cannot materialise thousands of episode dicts in RAM.
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        num_envs: int,
        obs_group_names: list[str],
        max_in_flight_writes: int = 8,
        max_workers: int = 2,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._num_envs = num_envs
        self._group_names = list(obs_group_names)
        self._obs_keys = [f"observation.{g}" for g in self._group_names]
        self._step_keys = (*self._obs_keys, "action", "reward", "terminated", "truncated")

        self._per_env_buffer: list[list[dict[str, np.ndarray]]] = [[] for _ in range(num_envs)]
        self._per_env_success: list[bool | None] = [None for _ in range(num_envs)]

        self._writer = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: list[Future[None]] = []
        self._write_sem = threading.BoundedSemaphore(max_in_flight_writes)
        self._num_saved = 0

    @property
    def num_saved(self) -> int:
        return self._num_saved

    def add_step(
        self,
        obs_groups: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        next_obs_groups: dict[str, np.ndarray],
        success: np.ndarray | None = None,
    ) -> int:
        """Append a batched transition; flush any envs whose episode just ended.

        Shapes (with N = num_envs):
            obs_groups[g]      (N, *obs_dim_g)
            action             (N, act_dim)
            reward             (N,)
            terminated         (N,) bool
            truncated          (N,) bool
            next_obs_groups[g] (N, *obs_dim_g)
            success            (N,) bool or None

        Returns the number of episodes flushed (== number of envs whose
        ``done`` flag is True on this call).
        """
        dones = terminated.astype(bool) | truncated.astype(bool)
        flushed = 0
        for i in range(self._num_envs):
            step: dict[str, np.ndarray] = {
                "action": action[i].copy(),
                "reward": reward[i].copy(),
                "terminated": terminated[i].copy(),
                "truncated": truncated[i].copy(),
            }
            for g, arr in obs_groups.items():
                step[f"observation.{g}"] = arr[i].copy()
            self._per_env_buffer[i].append(step)
            if success is not None:
                # Latch the most-recent success flag; on done we use this value.
                self._per_env_success[i] = bool(success[i])

            if dones[i]:
                env_buf = self._per_env_buffer[i]
                episode: dict[str, np.ndarray] = {
                    "action": np.stack([t["action"] for t in env_buf]),
                    "reward": np.stack([t["reward"] for t in env_buf]).astype(np.float32, copy=False),
                    "terminated": np.stack([t["terminated"] for t in env_buf]).astype(bool, copy=False),
                    "truncated": np.stack([t["truncated"] for t in env_buf]).astype(bool, copy=False),
                }
                # Each obs group gets an (N+1, ...) array: N per-step rows + 1
                # trailing terminal (next) row. Matches flash_rl.buffers layout.
                for g in self._group_names:
                    key = f"observation.{g}"
                    body = np.stack([t[key] for t in env_buf])
                    tail = next_obs_groups[g][i].copy()[None]
                    episode[key] = np.concatenate([body, tail], axis=0)
                if self._per_env_success[i] is not None:
                    episode["success"] = np.asarray(self._per_env_success[i], dtype=bool)

                # Reset per-env state BEFORE submitting so the next step can start
                # accumulating even while this episode is being compressed.
                self._per_env_buffer[i] = []
                self._per_env_success[i] = None

                path = _episode_path(self._output_dir, self._num_saved)
                self._num_saved += 1
                # Backpressure: block the collection loop if too many writes are
                # queued. Prevents a synchronised reset burst (common at fixed
                # episode length) from queueing thousands of episode dicts.
                self._write_sem.acquire()
                fut = self._writer.submit(_save_and_release, episode, path, self._write_sem)
                self._pending.append(fut)
                if len(self._pending) > 64:
                    self._pending = [f for f in self._pending if not f.done()]
                flushed += 1
        return flushed

    def close(self) -> None:
        self._writer.shutdown(wait=True)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _to_np(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32, copy=False)
    return arr


def _dot_lookup(root: Any, dotted: str) -> Any:
    """Resolve a dotted path like 'command_manager.object_pose.metrics.consecutive_success'.

    Falls through ``getattr`` first, then ``__getitem__``.
    """
    obj = root
    for part in dotted.split("."):
        if obj is None:
            return None
        if hasattr(obj, part):
            obj = getattr(obj, part)
            continue
        try:
            obj = obj[part]
        except (KeyError, TypeError):
            return None
    return obj


_SUCCESS_WARNED = {"done": False}


def _detect_success(env: Any, success_key: str | None) -> torch.Tensor | None:
    """Best-effort per-env success detection.

    Priority:
        1. CLI-supplied ``--success_key`` dot-path on ``env.unwrapped``.
        2. A term named '*success*' on the termination manager.
        3. ``command_manager.object_pose.metrics.consecutive_success`` > 0
           (reorient convention).
        4. Give up — returns ``None`` and warns once.
    """
    root = env.unwrapped
    if success_key:
        val = _dot_lookup(root, success_key)
        if isinstance(val, torch.Tensor):
            return val.bool()
        if val is not None:
            return torch.as_tensor(val).bool()

    tm = getattr(root, "termination_manager", None)
    if tm is not None:
        term_names = getattr(tm, "active_terms", None) or getattr(tm, "_term_names", None) or []
        for name in term_names:
            if "success" in name:
                state = None
                if hasattr(tm, "get_term"):
                    try:
                        state = tm.get_term(name)
                    except Exception:
                        state = None
                if isinstance(state, torch.Tensor):
                    return state.bool()

    cm = getattr(root, "command_manager", None)
    if cm is not None:
        try:
            term = cm.get_term("object_pose")
            metrics = getattr(term, "metrics", {})
            if "consecutive_success" in metrics:
                return (metrics["consecutive_success"] > 0).bool()
        except Exception:
            pass

    if not _SUCCESS_WARNED["done"]:
        print(
            "[WARN] Could not auto-detect a per-env success signal; `success` "
            "will be omitted from saved episodes. Pass --success_key to override."
        )
        _SUCCESS_WARNED["done"] = True
    return None


def _apply_grasp_overrides(env_cfg: Any, args_cli: argparse.Namespace) -> None:
    """Apply grasp/object overrides for both legacy and current env config schemas.

    Copy of ``_apply_grasp_overrides`` in scripts/rsl_rl/play.py.
    """
    if apply_grasp_overrides_from_cli(env_cfg, args_cli):
        return

    if hasattr(env_cfg, "grasp_data_path"):
        if args_cli.grasp_path is not None:
            env_cfg.grasp_data_path = args_cli.grasp_path
            env_cfg.use_grasp_init = True
        if args_cli.obj_urdf_path is not None:
            env_cfg.object_urdf_path = args_cli.obj_urdf_path
            env_cfg.use_grasp_init = True
        if args_cli.obj_scale is not None:
            env_cfg.object_scale_override = args_cli.obj_scale
            env_cfg.use_grasp_init = True
        if env_cfg.use_grasp_init:
            from src.env.tasks.anygrasp.utils.grasp_init import load_grasp_init

            env_cfg.grasp_init = load_grasp_init(
                env_cfg.grasp_data_path,
                object_scale_override=env_cfg.object_scale_override,
                object_urdf_path=env_cfg.object_urdf_path,
            )


def _resolve_checkpoint(args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: Any) -> str:
    """Resolve a checkpoint path (file / wandb / last-in-experiment).

    Same precedence as scripts/rsl_rl/play.py.
    """
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    if args_cli.wandb_path:
        import pathlib as _pathlib

        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        files = [f.name for f in wandb_run.files() if "model" in f.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is not None and args_cli.motion_file is None \
                and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.motion_file = str(_pathlib.Path(art.download()) / "motion.npz")
        return resume_path

    if args_cli.checkpoint is not None:
        return retrieve_file_path(args_cli.checkpoint)

    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_last_checkpoint:
        run_dir = agent_cfg.load_run if agent_cfg.load_run else ".*"
        return get_checkpoint_path(log_root_path, run_dir, "model_.*.pt")
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _resolve_output_dir(args_cli: argparse.Namespace) -> pathlib.Path:
    if args_cli.output_dir:
        out = pathlib.Path(args_cli.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (args_cli.task or "unknown").replace("/", "_")
        out = pathlib.Path.cwd() / "datasets" / slug / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _write_metadata(
    out_dir: pathlib.Path,
    args_cli: argparse.Namespace,
    groups: list[str],
    obs_manager: Any,
    action_dim: int,
    resume_path: str,
) -> None:
    def _dim(v: Any) -> Any:
        if isinstance(v, (list, tuple)):
            if v and isinstance(v[0], (list, tuple)):
                return [list(x) for x in v]
            return list(v)
        return v

    group_dims: dict[str, Any] = {}
    try:
        all_dims = obs_manager.group_obs_dim
        for g in groups:
            if g in all_dims:
                group_dims[g] = _dim(all_dims[g])
    except Exception:
        pass

    meta = {
        "task": args_cli.task,
        "checkpoint": str(resume_path),
        "num_envs": args_cli.num_envs,
        "num_episodes_target": args_cli.num_episodes,
        "seed": args_cli.seed,
        "deterministic": bool(args_cli.deterministic),
        "obs_groups": groups,
        "obs_group_dims": group_dims,
        "action_dim": int(action_dim),
        "success_key": args_cli.success_key,
        "git_sha": _git_sha(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)


def _select_groups(obs: Mapping[str, torch.Tensor], requested: Iterable[str]) -> list[str]:
    """Return the subset of ``requested`` groups that are present in ``obs``."""
    return [g for g in requested if g in obs]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = agent_cfg.seed
    _apply_grasp_overrides(env_cfg, args_cli)

    if args_cli.motion_file is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env_cfg.log_dir = os.path.dirname(resume_path)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    out_dir = _resolve_output_dir(args_cli)

    # RslRlVecEnvWrapper.get_observations() / .step() return a TensorDict keyed
    # by observation-group name; we use those tensors directly as the
    # policy-consumed observations (same objects the policy sees, no double-sample).
    obs = env.get_observations()
    requested_groups = [g.strip() for g in args_cli.obs_groups.split(",") if g.strip()]
    groups = _select_groups(obs, requested_groups)
    missing = [g for g in requested_groups if g not in groups]
    if missing:
        print(f"[WARN] Requested obs groups not present on this task and will be skipped: {missing}")
    if not groups:
        available = list(env.unwrapped.observation_manager.group_obs_dim.keys())
        raise RuntimeError(
            f"None of the requested --obs_groups {requested_groups!r} exist on this task's "
            f"observation manager. Available: {available}"
        )

    action_dim = int(env.action_space.shape[-1]) if env.action_space.shape else 0
    _write_metadata(out_dir, args_cli, groups, env.unwrapped.observation_manager, action_dim, resume_path)
    print(f"[INFO]: Writing episodes to {out_dir}")
    print(f"[INFO]: Saving obs groups: {groups}")

    writer = EpisodeWriter(
        output_dir=out_dir,
        num_envs=env.num_envs,
        obs_group_names=groups,
        max_in_flight_writes=args_cli.max_in_flight_writes,
    )

    try:
        from tqdm import tqdm

        pbar: Any = tqdm(total=args_cli.num_episodes, desc="Episodes", unit="ep")
    except ImportError:  # pragma: no cover
        class _Bar:
            n = 0

            def update(self, k: int) -> None:
                self.n += k
                print(f"[data_collection] {self.n} episodes saved", flush=True)

            def close(self) -> None:
                pass

        pbar = _Bar()

    device = env.unwrapped.device
    ep_return = torch.zeros(env.num_envs, device=device)
    ep_length = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    try:
        while writer.num_saved < args_cli.num_episodes and simulation_app.is_running():
            with torch.inference_mode():
                if args_cli.deterministic and hasattr(policy_nn, "act_inference"):
                    actions = policy_nn.act_inference(obs)
                else:
                    actions = policy(obs)

                new_obs, rew, dones, _infos = env.step(actions)
                truncated_t = env.unwrapped.termination_manager.time_outs.clone().bool()
                terminated_t = dones.bool() & ~truncated_t
                success_t = _detect_success(env, args_cli.success_key)

                obs_np = {g: _to_np(obs[g]) for g in groups}
                next_obs_np = {g: _to_np(new_obs[g]) for g in groups}
                n_flushed = writer.add_step(
                    obs_groups=obs_np,
                    action=_to_np(actions),
                    reward=_to_np(rew),
                    terminated=_to_np(terminated_t),
                    truncated=_to_np(truncated_t),
                    next_obs_groups=next_obs_np,
                    success=_to_np(success_t) if success_t is not None else None,
                )

                ep_return += rew
                ep_length += 1
                if dones.any():
                    done_mask = dones.bool()
                    finished_idx = torch.where(done_mask)[0].tolist()
                    for i in finished_idx:
                        returns.append(float(ep_return[i].item()))
                        lengths.append(int(ep_length[i].item()))
                        if success_t is not None:
                            successes.append(bool(success_t[i].item()))
                    ep_return[done_mask] = 0.0
                    ep_length[done_mask] = 0
                    pbar.update(n_flushed)

                if hasattr(policy_nn, "reset"):
                    policy_nn.reset(dones)
                obs = new_obs
    finally:
        pbar.close()
        writer.close()
        env.close()

    print(f"Saved {writer.num_saved} episodes to {out_dir}")
    print(f"  avg length : {np.mean(lengths) if lengths else 0.0:.1f}")
    print(f"  avg return : {np.mean(returns) if returns else 0.0:.3f}")
    if successes:
        print(f"  success    : {100.0 * np.mean(successes):.1f}%  ({sum(successes)}/{len(successes)})")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
