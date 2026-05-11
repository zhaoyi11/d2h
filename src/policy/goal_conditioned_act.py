"""ACT-style context-conditioned action-chunk policy for low-level control."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.policy.vae import (
    DEFAULT_WANDB_PROJECT,
    _build_window_datasets,
    _cuda_autocast,
    _default_output_dir,
    _default_wandb_run_name,
    _init_wandb,
    _move_batch,
    _rng_state,
    _wandb_log,
    _wandb_save,
)


MODEL_TYPE = "goal_conditioned_act_trajectory_chunk_policy"


@dataclass
class GoalConditionedACTConfig:
    state_dim: int
    action_dim: int
    past_length: int = 4
    future_length: int = 8
    condition_dim: int = 256
    hidden_dim: int = 256
    transformer_layers: int = 4
    transformer_heads: int = 4
    transformer_feedforward_dim: int = 1024
    transformer_dropout: float = 0.1

    @property
    def token_dim(self) -> int:
        return self.state_dim + self.action_dim

    @property
    def context_dim(self) -> int:
        return self.past_length * self.token_dim

    @property
    def target_length(self) -> int:
        return self.future_length


def _transformer_encoder(config: GoalConditionedACTConfig) -> nn.TransformerEncoder:
    if config.hidden_dim % config.transformer_heads != 0:
        raise ValueError("hidden_dim must be divisible by transformer_heads")
    layer = nn.TransformerEncoderLayer(
        d_model=config.hidden_dim,
        nhead=config.transformer_heads,
        dim_feedforward=config.transformer_feedforward_dim,
        dropout=config.transformer_dropout,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(layer, num_layers=config.transformer_layers)


class GoalConditionedACTContextEncoder(nn.Module):
    def __init__(self, config: GoalConditionedACTConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.token_dim, config.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, 1 + config.past_length, config.hidden_dim).normal_(std=0.02))
        self.transformer = _transformer_encoder(config)
        self.condition_proj = nn.Linear(config.hidden_dim, config.condition_dim)

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        if context.shape[-1] != self.config.context_dim:
            raise ValueError(f"Expected context dim {self.config.context_dim}, got {context.shape[-1]}.")
        tokens = context.reshape(-1, self.config.past_length, self.config.token_dim)
        hidden = self.input_proj(tokens)
        cls = self.cls_token.expand(hidden.size(0), -1, -1)
        hidden = torch.cat([cls, hidden], dim=1)
        hidden = hidden + self.pos_embedding[:, : hidden.size(1)]
        encoded = self.transformer(hidden)
        return self.condition_proj(encoded[:, 0]), encoded


class GoalConditionedACTQueryDecoder(nn.Module):
    def __init__(self, config: GoalConditionedACTConfig) -> None:
        super().__init__()
        self.config = config
        self.query_embed = nn.Embedding(config.target_length, config.hidden_dim)
        self.condition_memory_proj = nn.Linear(config.condition_dim, config.hidden_dim)
        layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_feedforward_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerDecoder(layer, num_layers=int(config.transformer_layers) * 2)
        self.action_head = nn.Linear(config.hidden_dim, config.action_dim)

    def forward(self, condition: Tensor, context_memory: Tensor) -> Tensor:
        batch_size = condition.size(0)
        condition_token = self.condition_memory_proj(condition).unsqueeze(1)
        memory = torch.cat([condition_token, context_memory], dim=1)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.transformer(queries, memory)
        return self.action_head(decoded)


class GoalConditionedACTPolicy(nn.Module):
    def __init__(self, config: GoalConditionedACTConfig) -> None:
        super().__init__()
        self.config = config
        self.context_encoder = GoalConditionedACTContextEncoder(config)
        self.decoder = GoalConditionedACTQueryDecoder(config)

    def forward(self, context: Tensor) -> Tensor:
        condition, context_memory = self.context_encoder(context)
        return self.decoder(condition, context_memory)


@torch.no_grad()
def _evaluate(
    model: GoalConditionedACTPolicy,
    loader: DataLoader,
    torch_device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = _move_batch(batch, torch_device)
        with _cuda_autocast(amp_enabled):
            pred = model(batch["context"])
            loss = F.mse_loss(pred, batch["target_actions"])
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return {"loss": total_loss / total_batches}


def _save_checkpoint(
    path: Path,
    model: GoalConditionedACTPolicy,
    optimizer: torch.optim.Optimizer,
    config: GoalConditionedACTConfig,
    train_args: dict[str, Any],
    epoch: int,
    global_step: int,
    best_val_loss: float,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    payload: dict[str, Any] = {
        "config": asdict(config),
        "model_type": MODEL_TYPE,
        "training": train_args,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "rng_state": _rng_state(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, path)


def train_goal_conditioned_act(
    dataset_dir: str | Path,
    output_dir: str | Path | None = None,
    past_length: int = 4,
    future_length: int = 8,
    condition_dim: int = 256,
    hidden_dim: int = 512,
    transformer_layers: int = 8,
    transformer_heads: int = 8,
    transformer_feedforward_dim: int = 1024,
    transformer_dropout: float = 0.1,
    batch_size: int = 64,
    epochs: int = 100,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = None,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    train_windows_per_epoch: int = 100_000,
    val_windows: int = 10_000,
    log_every_steps: int = 100,
    checkpoint_every_epochs: int = 10,
    wandb_project: str = DEFAULT_WANDB_PROJECT,
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    wandb_run_name: str | None = None,
    amp: bool = False,
    seed: int = 0,
    device: str = "cuda",
) -> GoalConditionedACTPolicy:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_dataset, val_dataset, dataset_info = _build_window_datasets(
        dataset_dir,
        past_length=past_length,
        future_length=future_length,
        val_ratio=val_ratio,
        train_windows_per_epoch=train_windows_per_epoch,
        val_windows=val_windows,
        seed=seed,
    )
    config = GoalConditionedACTConfig(
        state_dim=int(dataset_info.state_dim),
        action_dim=int(dataset_info.action_dim),
        past_length=past_length,
        future_length=future_length,
        condition_dim=condition_dim,
        hidden_dim=hidden_dim,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        transformer_feedforward_dim=transformer_feedforward_dim,
        transformer_dropout=transformer_dropout,
    )

    torch_device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch_device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    model = GoalConditionedACTPolicy(config).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    amp_enabled = bool(amp and torch_device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None

    output_path = _default_output_dir(dataset_dir) if output_dir is None else Path(output_dir).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    if wandb_run_name is None:
        wandb_run_name = _default_wandb_run_name(dataset_dir)

    train_args = {
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "output_dir": str(output_path),
        "past_length": past_length,
        "future_length": future_length,
        "condition_dim": condition_dim,
        "hidden_dim": hidden_dim,
        "transformer_layers": transformer_layers,
        "transformer_heads": transformer_heads,
        "transformer_feedforward_dim": transformer_feedforward_dim,
        "transformer_dropout": transformer_dropout,
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "grad_clip_norm": grad_clip_norm,
        "num_workers": num_workers,
        "val_ratio": val_ratio,
        "train_windows_per_epoch": train_windows_per_epoch,
        "val_windows": val_windows,
        "log_every_steps": log_every_steps,
        "checkpoint_every_epochs": checkpoint_every_epochs,
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_mode": wandb_mode,
        "wandb_run_name": wandb_run_name,
        "amp": amp,
        "seed": seed,
        "device": str(torch_device),
        "state_keys": dataset_info.state_keys,
        "eligible_episodes": dataset_info.eligible_episodes,
        "train_episodes": dataset_info.train_episodes,
        "val_episodes": dataset_info.val_episodes,
    }
    config_payload = asdict(config) | {"model_type": MODEL_TYPE, "training": train_args}
    with (output_path / "config.json").open("w") as f:
        json.dump(config_payload, f, indent=2)

    wandb_run = _init_wandb(
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        config_payload=config_payload,
        output_path=output_path,
    )

    global_step = 0
    best_val_loss = float("inf")
    log_path = output_path / "train_log.jsonl"
    try:
        with log_path.open("w") as log_file:
            for epoch in range(1, epochs + 1):
                model.train()
                total_loss = 0.0
                total_batches = 0

                for batch in train_loader:
                    batch = _move_batch(batch, torch_device)
                    optimizer.zero_grad(set_to_none=True)
                    with _cuda_autocast(amp_enabled):
                        pred = model(batch["context"])
                        loss = F.mse_loss(pred, batch["target_actions"])

                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if grad_clip_norm is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if grad_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                        optimizer.step()

                    global_step += 1
                    total_loss += float(loss.detach().cpu())
                    total_batches += 1

                    if log_every_steps > 0 and global_step % log_every_steps == 0:
                        _wandb_log(
                            wandb_run,
                            {
                                "train/loss": float(loss.detach().cpu()),
                                "train/lr": float(optimizer.param_groups[0]["lr"]),
                                "epoch": epoch,
                            },
                            step=global_step,
                        )

                train_loss = total_loss / total_batches
                val_metrics = _evaluate(model, val_loader, torch_device, amp_enabled=amp_enabled)
                val_loss = val_metrics["loss"]
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss

                row = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                }
                print(json.dumps(row), file=log_file, flush=True)
                print(
                    f"epoch {epoch:04d} step={global_step} loss={row['loss']:.6f} "
                    f"val_loss={row['val_loss']:.6f}",
                    flush=True,
                )
                _wandb_log(
                    wandb_run,
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_loss,
                        "val/loss": val_loss,
                        "best_val_loss": best_val_loss,
                    },
                    step=global_step,
                )

                last_path = output_path / "last.pt"
                _save_checkpoint(
                    last_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    train_args=train_args,
                    epoch=epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    scaler=scaler,
                )
                _wandb_save(wandb_run, last_path)

                if is_best:
                    best_path = output_path / "best.pt"
                    _save_checkpoint(
                        best_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        train_args=train_args,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        scaler=scaler,
                    )
                    _wandb_save(wandb_run, best_path)

                if checkpoint_every_epochs > 0 and epoch % checkpoint_every_epochs == 0:
                    epoch_path = output_path / f"epoch_{epoch:04d}.pt"
                    _save_checkpoint(
                        epoch_path,
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        train_args=train_args,
                        epoch=epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        scaler=scaler,
                    )
                    _wandb_save(wandb_run, epoch_path)
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    torch.save(
        {
            "config": asdict(config),
            "model_type": MODEL_TYPE,
            "model_state_dict": model.state_dict(),
        },
        output_path / "model.pt",
    )
    return model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ACT-style context-conditioned action-chunk policy.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--past-length", type=int, default=4)
    parser.add_argument("--future-length", type=int, default=8)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-feedforward-dim", type=int, default=1024)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--train-windows-per-epoch", type=int, default=100_000)
    parser.add_argument("--val-windows", type=int, default=10_000)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=10)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_goal_conditioned_act(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        past_length=args.past_length,
        future_length=args.future_length,
        condition_dim=args.condition_dim,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        transformer_feedforward_dim=args.transformer_feedforward_dim,
        transformer_dropout=args.transformer_dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        train_windows_per_epoch=args.train_windows_per_epoch,
        val_windows=args.val_windows,
        log_every_steps=args.log_every_steps,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        wandb_run_name=args.wandb_run_name,
        amp=args.amp,
        seed=args.seed,
        device=args.device,
    )
