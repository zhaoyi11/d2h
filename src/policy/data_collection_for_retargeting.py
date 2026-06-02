"""Collect named LEAP-hand rollout data for downstream human-keypoint retargeting.

This script mirrors the RSL-RL rollout setup in ``src.policy.data_collection``
but writes raw, named kinematic arrays instead of a concatenated policy
observation vector.  It intentionally does not implement retargeting.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any

import numpy as np
import torch


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RSL_RL_SCRIPT_DIR = _REPO_ROOT / "scripts" / "rsl_rl"

FINGERTIP_LINK_NAMES = ("thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3")
CONTACT_SENSOR_NAMES = ("thumb_tip_object_s", "index_tip_object_s", "middle_tip_object_s", "ring_tip_object_s")
HUMAN21_KEYPOINT_NAMES = (
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
OPTIONAL_EPISODE_FIELDS = {
    "leap_fingertip_pose_w",
    "leap_fingertip_pose_b",
    "fingertip_contact_mask",
    "fingertip_contact_force_b",
    "goal_pose_w",
    "joint_target_pos",
}


# --------------------------------------------------------------------------- #
# RetargetingEpisodeWriter
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


def _episode_path(output_dir: pathlib.Path, eps_idx: int) -> pathlib.Path:
    return output_dir / f"{eps_idx:010d}.npz"


class RetargetingEpisodeWriter:
    """Per-env accumulator for retargeting episodes.

    Unlike the generic policy dataset writer, this writer does not append a
    post-reset observation row.  Every saved array has exactly one row per
    recorded pre-step motion frame.
    """

    def __init__(
        self,
        output_dir: pathlib.Path,
        num_envs: int,
        max_in_flight_writes: int = 8,
        max_workers: int = 2,
        max_episodes: int | None = None,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir).expanduser()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._num_envs = int(num_envs)
        self._per_env_buffer: list[list[dict[str, np.ndarray]]] = [[] for _ in range(self._num_envs)]
        self._per_env_success: list[bool | None] = [None for _ in range(self._num_envs)]
        self._recorded_keys: set[str] = set()

        self._writer = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: list[Future[None]] = []
        self._write_sem = threading.BoundedSemaphore(max_in_flight_writes)
        self._max_episodes = max_episodes
        self._num_saved = 0

    @property
    def num_saved(self) -> int:
        return self._num_saved

    @property
    def recorded_keys(self) -> set[str]:
        return set(self._recorded_keys)

    def add_step(
        self,
        step_data: Mapping[str, np.ndarray],
        done: np.ndarray,
        success: np.ndarray | None = None,
    ) -> int:
        """Append a batched transition and flush envs whose episode ended."""
        dones = np.asarray(done).astype(bool)
        flushed = 0
        for env_id in range(self._num_envs):
            step: dict[str, np.ndarray] = {}
            for key, value in step_data.items():
                if value is None:
                    continue
                item = value[env_id]
                step[key] = item.copy() if isinstance(item, np.ndarray) else np.asarray(item).copy()
                self._recorded_keys.add(key)
            self._per_env_buffer[env_id].append(step)

            if success is not None:
                self._per_env_success[env_id] = bool(success[env_id])

            if not dones[env_id]:
                continue

            env_buf = self._per_env_buffer[env_id]
            self._per_env_buffer[env_id] = []
            episode: dict[str, np.ndarray] = {}
            for key in sorted({key for frame in env_buf for key in frame}):
                if all(key in frame for frame in env_buf):
                    episode[key] = np.stack([frame[key] for frame in env_buf])

            if self._per_env_success[env_id] is not None:
                episode["success"] = np.asarray(self._per_env_success[env_id], dtype=bool)
                self._recorded_keys.add("success")
            self._per_env_success[env_id] = None

            if self._max_episodes is not None and self._num_saved >= self._max_episodes:
                continue

            path = _episode_path(self._output_dir, self._num_saved)
            self._num_saved += 1
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


def _select_clean_and_env_actions(
    obs: Any,
    policy: Any,
    policy_nn: Any,
    deterministic: bool,
    action_noise_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return clean policy actions for storage and executed actions for stepping."""
    if deterministic and hasattr(policy_nn, "act_inference"):
        clean_actions = policy_nn.act_inference(obs)
    else:
        clean_actions = policy(obs)

    noise_std = action_noise_std.to(device=clean_actions.device, dtype=clean_actions.dtype)
    if bool((noise_std > 0.0).any()):
        view_shape = (-1, *([1] * (clean_actions.ndim - 1)))
        env_actions = clean_actions + torch.randn_like(clean_actions) * noise_std.view(view_shape)
    else:
        env_actions = clean_actions
    return clean_actions, env_actions


def _sample_action_noise_std(
    num_envs: int,
    device: torch.device | str,
    max_std: float,
    current: torch.Tensor | None = None,
    done: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-env scalar action-noise stds, optionally replacing only done envs."""
    samples = torch.rand(num_envs, device=device, generator=generator) * float(max_std)
    if current is None or done is None:
        return samples
    updated = current.clone()
    done_mask = done.to(device=updated.device).bool()
    updated[done_mask] = samples[done_mask]
    return updated


def _dot_lookup(root: Any, dotted: str) -> Any:
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
    """Best-effort per-env success detection."""
    root = env.unwrapped if hasattr(env, "unwrapped") else env
    if success_key:
        val = _dot_lookup(root, success_key)
        if isinstance(val, torch.Tensor):
            return val.bool()
        if val is not None:
            return torch.as_tensor(val, device=root.device).bool()

    tm = getattr(root, "termination_manager", None)
    if tm is not None:
        term_names = getattr(tm, "active_terms", None) or getattr(tm, "_term_names", None) or []
        for name in term_names:
            if "success" not in name:
                continue
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


def _tensor_attr(data: Any, names: Sequence[str], label: str) -> torch.Tensor:
    for name in names:
        value = getattr(data, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError(f"Could not find required tensor for {label}; tried {list(names)}")


def _optional_tensor_attr(data: Any, names: Sequence[str]) -> torch.Tensor | None:
    for name in names:
        value = getattr(data, name, None)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _pose(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((pos, quat), dim=-1)


def _entity_names(entity: Any, attr_name: str) -> list[str]:
    names = getattr(entity, attr_name, None)
    if names is None:
        names = getattr(getattr(entity, "data", None), attr_name, None)
    if names is None:
        return []
    return [str(name) for name in names]


def _resolve_fingertip_body_ids(robot: Any) -> tuple[list[int], list[str]]:
    body_names = _entity_names(robot, "body_names")
    if body_names:
        if all(name in body_names for name in FINGERTIP_LINK_NAMES):
            return [body_names.index(name) for name in FINGERTIP_LINK_NAMES], list(FINGERTIP_LINK_NAMES)
        matches = [(idx, name) for idx, name in enumerate(body_names) if "fingertip" in name]
        if len(matches) >= 4:
            matches = matches[:4]
            return [idx for idx, _ in matches], [name for _, name in matches]

    finder = getattr(robot, "find_bodies", None)
    if callable(finder):
        try:
            ids, names = finder(list(FINGERTIP_LINK_NAMES))
            if len(ids) == len(FINGERTIP_LINK_NAMES):
                return [int(idx) for idx in ids], [str(name) for name in names]
        except Exception:
            pass
        try:
            ids, names = finder(".*fingertip.*")
            if len(ids) >= 4:
                return [int(idx) for idx in ids[:4]], [str(name) for name in names[:4]]
        except Exception:
            pass

    return [], []


def _base_frame_poses(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    subtract_frame_transforms: Any,
) -> torch.Tensor:
    num_envs, num_bodies = pos_w.shape[:2]
    flat_pos = pos_w.reshape(-1, 3)
    flat_quat = quat_w.reshape(-1, 4)
    root_pos = root_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 3)
    root_quat = root_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).reshape(-1, 4)
    pos_b, quat_b = subtract_frame_transforms(root_pos, root_quat, flat_pos, flat_quat)
    return _pose(pos_b, quat_b).reshape(num_envs, num_bodies, 7)


def _base_frame_vectors(
    vectors_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    quat_apply_inverse: Any,
) -> torch.Tensor:
    num_envs, count = vectors_w.shape[:2]
    flat_vectors = vectors_w.reshape(-1, 3)
    root_quat = root_quat_w.unsqueeze(1).repeat_interleave(count, dim=1).reshape(-1, 4)
    return quat_apply_inverse(root_quat, flat_vectors).reshape(num_envs, count, 3)


def _sensor_force_w(sensor: Any, num_envs: int) -> torch.Tensor | None:
    data = getattr(sensor, "data", None)
    if data is None:
        return None
    force = getattr(data, "force_matrix_w", None)
    if not isinstance(force, torch.Tensor) or force.numel() == 0:
        force = getattr(data, "net_forces_w", None)
    if not isinstance(force, torch.Tensor) or force.numel() == 0:
        return None
    return force.reshape(num_envs, -1, 3).sum(dim=1)


def _contact_arrays(
    env: Any,
    root_quat_w: torch.Tensor,
    quat_apply_inverse: Any,
    force_threshold: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    sensors = getattr(getattr(env, "scene", None), "sensors", {})
    forces_w = []
    for name in CONTACT_SENSOR_NAMES:
        try:
            sensor = sensors[name]
        except (KeyError, TypeError):
            return None, None
        force_w = _sensor_force_w(sensor, env.num_envs)
        if force_w is None:
            return None, None
        forces_w.append(force_w)

    force_w_t = torch.stack(forces_w, dim=1)
    force_b = _base_frame_vectors(force_w_t, root_quat_w, quat_apply_inverse)
    mask = torch.linalg.norm(force_b, dim=-1) > float(force_threshold)
    return mask, force_b


def _goal_pose_w(env: Any) -> torch.Tensor | None:
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        return None
    try:
        term = command_manager.get_term("object_pose")
    except Exception:
        return None
    pos = getattr(term, "pos_command_w", None)
    quat = getattr(term, "quat_command_w", None)
    if isinstance(pos, torch.Tensor) and isinstance(quat, torch.Tensor):
        return _pose(pos, quat)
    command = getattr(term, "command", None)
    if isinstance(command, torch.Tensor) and command.shape[-1] >= 7:
        scene = getattr(env, "scene", None)
        origins = getattr(scene, "env_origins", None)
        pos_w = command[:, 0:3] + origins if isinstance(origins, torch.Tensor) else command[:, 0:3]
        return _pose(pos_w, command[:, 3:7])
    return None


def _joint_target_pos(env: Any, action_term_name: str) -> torch.Tensor | None:
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None or not hasattr(action_manager, "get_term"):
        return None
    try:
        term = action_manager.get_term(action_term_name)
    except Exception:
        return None
    target = getattr(term, "processed_actions", None)
    if not isinstance(target, torch.Tensor):
        target = getattr(term, "_processed_actions", None)
    return target if isinstance(target, torch.Tensor) else None


def _extract_retargeting_state(
    env: Any,
    action_term_name: str,
    contact_force_threshold: float,
    subtract_frame_transforms: Any,
    quat_apply_inverse: Any,
) -> dict[str, np.ndarray]:
    """Extract named, pre-step kinematic state for every env."""
    robot = env.scene["robot"]
    obj = env.scene["object"]

    root_pos_w = _tensor_attr(robot.data, ("root_link_pos_w", "root_pos_w"), "LEAP root position")
    root_quat_w = _tensor_attr(robot.data, ("root_link_quat_w", "root_quat_w"), "LEAP root orientation")
    body_pos_w = _tensor_attr(robot.data, ("body_pos_w",), "LEAP link positions")
    body_quat_w = _tensor_attr(robot.data, ("body_quat_w",), "LEAP link orientations")
    object_pos_w = _tensor_attr(obj.data, ("root_pos_w",), "object root position")
    object_quat_w = _tensor_attr(obj.data, ("root_quat_w",), "object root orientation")

    link_pose_w = _pose(body_pos_w, body_quat_w)
    link_pose_b = _base_frame_poses(body_pos_w, body_quat_w, root_pos_w, root_quat_w, subtract_frame_transforms)
    object_pos_b, object_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, object_pos_w, object_quat_w)

    step: dict[str, np.ndarray] = {
        "time": _to_np(env.episode_length_buf.to(dtype=torch.float32) * env.step_dt),
        "frame_idx": _to_np(env.episode_length_buf),
        "leap_joint_pos": _to_np(_tensor_attr(robot.data, ("joint_pos",), "LEAP joint positions")),
        "leap_joint_vel": _to_np(_tensor_attr(robot.data, ("joint_vel",), "LEAP joint velocities")),
        "leap_root_pose_w": _to_np(_pose(root_pos_w, root_quat_w)),
        "leap_link_pose_w": _to_np(link_pose_w),
        "leap_link_pose_b": _to_np(link_pose_b),
        "object_pose_w": _to_np(_pose(object_pos_w, object_quat_w)),
        "object_pose_b": _to_np(_pose(object_pos_b, object_quat_b)),
    }

    object_lin_vel_w = _optional_tensor_attr(obj.data, ("root_lin_vel_w",))
    object_ang_vel_w = _optional_tensor_attr(obj.data, ("root_ang_vel_w",))
    if object_lin_vel_w is not None:
        step["object_lin_vel_w"] = _to_np(object_lin_vel_w)
    if object_ang_vel_w is not None:
        step["object_ang_vel_w"] = _to_np(object_ang_vel_w)

    fingertip_ids, _ = _resolve_fingertip_body_ids(robot)
    if fingertip_ids:
        ids = torch.as_tensor(fingertip_ids, dtype=torch.long, device=body_pos_w.device)
        step["leap_fingertip_pose_w"] = _to_np(link_pose_w[:, ids])
        step["leap_fingertip_pose_b"] = _to_np(link_pose_b[:, ids])

    contact_mask, contact_force_b = _contact_arrays(env, root_quat_w, quat_apply_inverse, contact_force_threshold)
    if contact_mask is not None and contact_force_b is not None:
        step["fingertip_contact_mask"] = _to_np(contact_mask)
        step["fingertip_contact_force_b"] = _to_np(contact_force_b)

    goal_pose_w = _goal_pose_w(env)
    if goal_pose_w is not None:
        step["goal_pose_w"] = _to_np(goal_pose_w)

    joint_target = _joint_target_pos(env, action_term_name)
    if joint_target is not None:
        step["joint_target_pos"] = _to_np(joint_target)

    return step


def _add_transition_data(
    step: dict[str, np.ndarray],
    clean_actions: torch.Tensor,
    executed_actions: torch.Tensor,
    action_noise_std: torch.Tensor,
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
) -> dict[str, np.ndarray]:
    step = dict(step)
    step.update(
        {
            "policy_action_clean": _to_np(clean_actions),
            "policy_action_executed": _to_np(executed_actions),
            "action_noise_std": _to_np(action_noise_std),
            "reward": _to_np(reward),
            "terminated": _to_np(terminated),
            "truncated": _to_np(truncated),
        }
    )
    return step


def _resolve_output_dir(args_cli: argparse.Namespace) -> pathlib.Path:
    if args_cli.output_dir:
        out = pathlib.Path(args_cli.output_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = (args_cli.task or "unknown").replace("/", "_")
        out = pathlib.Path.cwd() / "datasets" / slug / f"retargeting_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def _build_metadata_payload(
    args_cli: argparse.Namespace,
    env_cfg: Any,
    robot: Any,
    action_dim: int,
    resume_path: str,
    git_sha: str | None,
    timestamp: str,
    recorded_fields: Sequence[str],
) -> dict[str, Any]:
    sim_cfg = getattr(env_cfg, "sim", None)
    sim_dt = getattr(sim_cfg, "dt", None)
    decimation = getattr(env_cfg, "decimation", None)
    dt = float(sim_dt * decimation) if sim_dt is not None and decimation is not None else None
    _, fingertip_link_names = _resolve_fingertip_body_ids(robot)
    recorded = sorted(str(field) for field in recorded_fields)

    return {
        "task": args_cli.task,
        "checkpoint": str(resume_path),
        "num_envs": args_cli.num_envs,
        "num_episodes_target": args_cli.num_episodes,
        "seed": args_cli.seed,
        "deterministic": bool(args_cli.deterministic),
        "action_noise_std_range": [0.0, float(args_cli.action_noise_std_max)],
        "action_dim": int(action_dim),
        "action_term_name": args_cli.action_term_name,
        "contact_force_threshold": float(args_cli.contact_force_threshold),
        "success_key": args_cli.success_key,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "dt": dt,
        "sim_dt": float(sim_dt) if sim_dt is not None else None,
        "decimation": int(decimation) if decimation is not None else None,
        "episode_length_s": getattr(env_cfg, "episode_length_s", None),
        "joint_names": _entity_names(robot, "joint_names"),
        "link_names": _entity_names(robot, "body_names"),
        "fingertip_link_names": fingertip_link_names,
        "units": {
            "position": "meters",
            "angle": "radians",
            "time": "seconds",
            "force": "newtons",
        },
        "quaternion_convention": "wxyz",
        "frame_convention": {
            "_w": "world frame",
            "_b": "LEAP base/root frame",
        },
        "human21_keypoint_names": list(HUMAN21_KEYPOINT_NAMES),
        "human21_pinky_policy": (
            "LEAP source has thumb, index, middle, and ring fingers but no pinky. "
            "Pinky keypoints must be masked or synthesized downstream."
        ),
        "recorded_fields": recorded,
        "omitted_optional_fields": sorted(OPTIONAL_EPISODE_FIELDS.difference(recorded)),
    }


def _write_metadata(
    out_dir: pathlib.Path,
    args_cli: argparse.Namespace,
    env_cfg: Any,
    robot: Any,
    action_dim: int,
    resume_path: str,
    recorded_fields: Sequence[str],
) -> None:
    meta = _build_metadata_payload(
        args_cli=args_cli,
        env_cfg=env_cfg,
        robot=robot,
        action_dim=action_dim,
        resume_path=resume_path,
        git_sha=_git_sha(),
        timestamp=datetime.now().isoformat(timespec="seconds"),
        recorded_fields=recorded_fields,
    )
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)


def _apply_grasp_overrides(env_cfg: Any, args_cli: argparse.Namespace, apply_grasp_overrides_from_cli: Any) -> None:
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


def _resolve_checkpoint(
    args_cli: argparse.Namespace,
    agent_cfg: Any,
    env_cfg: Any,
    retrieve_file_path: Any,
    get_checkpoint_path: Any,
) -> str:
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


def _load_actor_only_checkpoint(runner: Any, resume_path: str, device: str) -> None:
    checkpoint = torch.load(resume_path, weights_only=False, map_location=device)
    loaded_state = checkpoint["model_state_dict"]
    policy = getattr(runner.alg, "policy", None)
    if policy is None:
        policy = getattr(runner.alg, "actor_critic", None)
    if policy is None:
        raise RuntimeError("Runner does not expose a policy module for actor-only checkpoint loading.")

    current_state = policy.state_dict()
    merged_state = current_state.copy()
    loaded_keys: list[str] = []
    skipped_keys: list[str] = []
    critic_prefixes = ("critic.", "critic_obs_normalizer.")

    for key, value in loaded_state.items():
        if key.startswith(critic_prefixes):
            skipped_keys.append(key)
            continue
        if key not in current_state or current_state[key].shape != value.shape:
            skipped_keys.append(key)
            continue
        merged_state[key] = value
        loaded_keys.append(key)

    if not any(key.startswith("actor.") for key in loaded_keys):
        raise RuntimeError(f"No actor weights could be loaded from checkpoint: {resume_path}")

    policy.load_state_dict(merged_state, strict=True)
    print(
        f"[INFO]: Actor-only checkpoint load: loaded {len(loaded_keys)} keys, "
        f"skipped {len(skipped_keys)} incompatible/non-actor keys."
    )


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #


def _build_parser(cli_args: Any, AppLauncher: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect LEAP rollout data for human-keypoint retargeting.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default="Reorient_Play-v0", help="Name of the task.")
    parser.add_argument(
        "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent config entry point."
    )
    parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
    parser.add_argument("--grasp_path", type=str, default=None, help="Path to grasp data file (.npy).")
    parser.add_argument("--obj_urdf_path", type=str, default=None, help="Path to object URDF.")
    parser.add_argument("--obj_scale", type=float, default=None, help="Override object scale from grasp data.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
    parser.add_argument(
        "--use_last_checkpoint", action="store_true", help="When no checkpoint provided, use the last saved model."
    )
    parser.add_argument(
        "--actor_only_checkpoint_load",
        action="store_true",
        help="Load only actor-compatible checkpoint weights; useful when critic privileged observations changed.",
    )
    parser.add_argument("--num_episodes", type=int, default=1000, help="Stop after this many completed episodes.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write {idx:010d}.npz files into. Defaults to datasets/<task>/retargeting_<timestamp>/.",
    )
    parser.add_argument(
        "--action_noise_std_max",
        type=float,
        default=0.0,
        help="Upper bound for per-env Gaussian action-noise std sampled uniformly from [0, this] on reset.",
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
        "--action_term_name",
        type=str,
        default="joint_pos",
        help="Action term whose processed target should be saved as joint_target_pos.",
    )
    parser.add_argument(
        "--contact_force_threshold",
        type=float,
        default=0.25,
        help="Force threshold used for fingertip_contact_mask.",
    )
    parser.add_argument(
        "--max_in_flight_writes",
        type=int,
        default=8,
        help="Maximum number of compressed-write jobs allowed to be queued at once.",
    )
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _run() -> None:
    from isaaclab.app import AppLauncher

    sys.path.insert(0, str(_RSL_RL_SCRIPT_DIR))
    import cli_args  # noqa: E402

    parser = _build_parser(cli_args, AppLauncher)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import src.tasks  # noqa: F401
        from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
        from isaaclab.envs import multi_agent_to_single_agent
        from isaaclab.utils.assets import retrieve_file_path
        from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms
        from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import get_checkpoint_path
        from isaaclab_tasks.utils.hydra import hydra_task_config
        from rsl_rl.runners import DistillationRunner, OnPolicyRunner
        from src.utils import apply_grasp_overrides_from_cli

        @hydra_task_config(args_cli.task, args_cli.agent)
        def main(
            env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
            agent_cfg: RslRlBaseRunnerCfg,
        ) -> None:
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
            env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
            env_cfg.seed = agent_cfg.seed
            _apply_grasp_overrides(env_cfg, args_cli, apply_grasp_overrides_from_cli)

            if args_cli.motion_file is not None and hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
                print(f"[INFO]: Using motion file from CLI: {args_cli.motion_file}")
                env_cfg.commands.motion.motion_file = args_cli.motion_file

            resume_path = _resolve_checkpoint(
                args_cli=args_cli,
                agent_cfg=agent_cfg,
                env_cfg=env_cfg,
                retrieve_file_path=retrieve_file_path,
                get_checkpoint_path=get_checkpoint_path,
            )
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

            if args_cli.actor_only_checkpoint_load:
                if agent_cfg.class_name != "OnPolicyRunner":
                    raise ValueError("--actor_only_checkpoint_load is only supported for OnPolicyRunner checkpoints.")
                _load_actor_only_checkpoint(runner, resume_path, agent_cfg.device)
            else:
                runner.load(resume_path)

            policy = runner.get_inference_policy(device=env.unwrapped.device)
            try:
                policy_nn = runner.alg.policy
            except AttributeError:
                policy_nn = runner.alg.actor_critic

            out_dir = _resolve_output_dir(args_cli)
            obs = env.get_observations()
            action_dim = int(env.action_space.shape[-1]) if env.action_space.shape else 0
            print(f"[INFO]: Writing retargeting episodes to {out_dir}")

            writer = RetargetingEpisodeWriter(
                output_dir=out_dir,
                num_envs=env.num_envs,
                max_in_flight_writes=args_cli.max_in_flight_writes,
                max_episodes=args_cli.num_episodes,
            )

            try:
                from tqdm import tqdm

                pbar: Any = tqdm(total=args_cli.num_episodes, desc="Episodes", unit="ep")
            except ImportError:
                class _Bar:
                    n = 0

                    def update(self, k: int) -> None:
                        self.n += k
                        print(f"[retargeting_collection] {self.n} episodes saved", flush=True)

                    def close(self) -> None:
                        pass

                pbar = _Bar()

            device = env.unwrapped.device
            ep_return = torch.zeros(env.num_envs, device=device)
            ep_length = torch.zeros(env.num_envs, device=device, dtype=torch.long)
            action_noise_std = _sample_action_noise_std(
                num_envs=env.num_envs,
                device=device,
                max_std=args_cli.action_noise_std_max,
            )
            returns: list[float] = []
            lengths: list[int] = []
            successes: list[bool] = []

            try:
                while writer.num_saved < args_cli.num_episodes and simulation_app.is_running():
                    with torch.inference_mode():
                        retargeting_state = _extract_retargeting_state(
                            env=env.unwrapped,
                            action_term_name=args_cli.action_term_name,
                            contact_force_threshold=args_cli.contact_force_threshold,
                            subtract_frame_transforms=subtract_frame_transforms,
                            quat_apply_inverse=quat_apply_inverse,
                        )
                        clean_actions, env_actions = _select_clean_and_env_actions(
                            obs=obs,
                            policy=policy,
                            policy_nn=policy_nn,
                            deterministic=args_cli.deterministic,
                            action_noise_std=action_noise_std,
                        )
                        new_obs, rew, dones, infos = env.step(env_actions)
                        done_t = dones.bool()
                        time_outs = infos.get("time_outs") if isinstance(infos, Mapping) else None
                        if time_outs is None:
                            truncated_t = env.unwrapped.termination_manager.time_outs.clone().bool()
                        elif isinstance(time_outs, torch.Tensor):
                            truncated_t = time_outs.to(device=done_t.device).bool()
                        else:
                            truncated_t = torch.as_tensor(time_outs, device=done_t.device).bool()
                        terminated_t = done_t & ~truncated_t
                        success_t = _detect_success(env, args_cli.success_key)

                        joint_target = _joint_target_pos(env.unwrapped, args_cli.action_term_name)
                        if joint_target is not None:
                            retargeting_state["joint_target_pos"] = _to_np(joint_target)

                        step_data = _add_transition_data(
                            step=retargeting_state,
                            clean_actions=clean_actions,
                            executed_actions=env_actions,
                            action_noise_std=action_noise_std,
                            reward=rew,
                            terminated=terminated_t,
                            truncated=truncated_t,
                        )
                        n_flushed = writer.add_step(
                            step_data=step_data,
                            done=_to_np(done_t),
                            success=_to_np(success_t) if success_t is not None else None,
                        )

                        ep_return += rew
                        ep_length += 1
                        if done_t.any():
                            done_mask = done_t
                            finished_idx = torch.where(done_mask)[0].tolist()
                            for i in finished_idx:
                                returns.append(float(ep_return[i].item()))
                                lengths.append(int(ep_length[i].item()))
                                if success_t is not None:
                                    successes.append(bool(success_t[i].item()))
                            ep_return[done_mask] = 0.0
                            ep_length[done_mask] = 0
                            action_noise_std = _sample_action_noise_std(
                                num_envs=env.num_envs,
                                device=device,
                                max_std=args_cli.action_noise_std_max,
                                current=action_noise_std,
                                done=done_mask,
                            )
                            pbar.update(n_flushed)

                        if hasattr(policy_nn, "reset"):
                            policy_nn.reset(dones)
                        obs = new_obs
            finally:
                pbar.close()
                writer.close()
                _write_metadata(
                    out_dir=out_dir,
                    args_cli=args_cli,
                    env_cfg=env_cfg,
                    robot=env.unwrapped.scene["robot"],
                    action_dim=action_dim,
                    resume_path=resume_path,
                    recorded_fields=writer.recorded_keys,
                )
                env.close()

            print(f"Saved {writer.num_saved} retargeting episodes to {out_dir}")
            print(f"  avg length : {np.mean(lengths) if lengths else 0.0:.1f}")
            print(f"  avg return : {np.mean(returns) if returns else 0.0:.3f}")
            if successes:
                print(f"  success    : {100.0 * np.mean(successes):.1f}%  ({sum(successes)}/{len(successes)})")

        main()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    _run()
