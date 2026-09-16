"""Training / validation / test engine.

Plain, explicit PyTorch loops - no Lightning, no high-level trainer. The canonical
sequence is visible in :func:`train_one_epoch`::

    model.train()
    optimizer.zero_grad(set_to_none=True)
    outputs = model(inputs)      # forward (inside torch.autocast when AMP is on)
    loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

and :func:`validate` / :func:`test` run under ``model.eval()`` + ``torch.inference_mode()``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CIFAR10_CLASSES, NUM_CLASSES, TrainConfig
from models import create_model, get_model_summary
from utils import (
    AverageMeter,
    append_csv_row,
    amp_supported,
    autocast_context,
    count_parameters,
    format_seconds,
    gpu_memory_summary,
    iter_batches,
    make_grad_scaler,
    make_summary_writer,
    save_checkpoint,
    write_history_csv,
)

__all__ = [
    "train_one_epoch",
    "validate",
    "test",
    "fit",
    "load_finetuned_model",
    "run_test_on_checkpoint",
]


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Number of correct top-1 predictions in a batch."""
    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Create the optimizer. The benchmark always uses AdamW(lr=0.005, weight_decay=1e-4)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found - did you freeze the whole model?")

    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Expected one of: adamw, adam, sgd.")


def build_scheduler(
    optimizer: torch.optim.Optimizer, name: str, epochs: int
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Optional LR schedule (default ``none``: constant lr=0.005 for a simple, fair recipe)."""
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.1)
    raise ValueError(f"Unknown scheduler '{name}'. Expected one of: none, cosine, step.")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Current learning rate of the first parameter group."""
    return float(optimizer.param_groups[0]["lr"])


# --------------------------------------------------------------------------------------
# One epoch of training
# --------------------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[Any] = None,
    amp: bool = False,
    epoch: Optional[int] = None,
    epochs: Optional[int] = None,
    grad_clip: float = 0.0,
    limit_batches: Optional[int] = None,
    log_interval: int = 0,
) -> Dict[str, float]:
    """Run a single training epoch and return ``{"loss", "accuracy", "lr"}``.

    Accuracy and loss are sample-weighted over the whole epoch (values in ``%`` for accuracy).
    """
    model.train()

    loss_meter = AverageMeter("train_loss")
    correct = 0
    seen = 0
    start = time.perf_counter()
    num_batches = len(loader) if limit_batches is None else min(limit_batches, len(loader))

    for batch_index, (inputs, targets) in enumerate(iter_batches(loader, limit_batches)):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp):
            logits = model(inputs)           # forward
            loss = criterion(logits, targets)  # loss

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()    # backward (scaled)
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)           # optimizer step
            scaler.update()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = targets.size(0)
        loss_meter.update(loss.detach().item(), batch_size)
        correct += _accuracy(logits.detach(), targets)
        seen += batch_size

        if log_interval and (batch_index + 1) % log_interval == 0:
            elapsed = time.perf_counter() - start
            print(
                f"    [epoch {epoch}/{epochs}] batch {batch_index + 1}/{num_batches} "
                f"| loss {loss_meter.avg:.4f} | acc {100.0 * correct / max(seen, 1):.2f}% "
                f"| {elapsed:.1f}s",
                flush=True,
            )

    return {
        "loss": loss_meter.avg,
        "accuracy": 100.0 * correct / max(seen, 1),
        "lr": current_lr(optimizer),
        "epoch_time": time.perf_counter() - start,
        "num_samples": seen,
    }


# --------------------------------------------------------------------------------------
# Evaluation (validation / test)
# --------------------------------------------------------------------------------------
@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
    limit_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate ``model`` on ``loader`` in eval mode without gradient tracking."""
    model.eval()

    loss_meter = AverageMeter("val_loss")
    correct = 0
    seen = 0
    start = time.perf_counter()

    for inputs, targets in iter_batches(loader, limit_batches):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast_context(device, amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        loss_meter.update(loss.detach().item(), batch_size)
        correct += _accuracy(logits, targets)
        seen += batch_size

    return {
        "loss": loss_meter.avg,
        "accuracy": 100.0 * correct / max(seen, 1),
        "num_samples": seen,
        "eval_time": time.perf_counter() - start,
    }


@torch.inference_mode()
def test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool = False,
    limit_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate on the CIFAR-10 test split (same maths as :func:`validate`)."""
    return validate(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        amp=amp,
        limit_batches=limit_batches,
    )


# --------------------------------------------------------------------------------------
# Checkpoint -> model
# --------------------------------------------------------------------------------------
def load_finetuned_model(
    checkpoint_path: str,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild a model from a fine-tuned checkpoint and return ``(model, checkpoint)``.

    The architecture is created *without* downloading ImageNet weights because every
    parameter is overwritten by the fine-tuned ``model_state_dict`` stored in the
    checkpoint (this is the only place where ``pretrained=False`` is legitimate - the
    ImageNet pretrained weights are exactly what was fine-tuned into this file).
    """
    from utils import load_checkpoint  # local import keeps the module import graph flat

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name")
    if not model_name:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' has no 'model_name' field; cannot rebuild the architecture."
        )

    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    state_dict = checkpoint["model_state_dict"]
    # Tolerate checkpoints saved from nn.DataParallel / DDP wrappers.
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[engine] load_state_dict warnings -> missing={list(missing)} unexpected={list(unexpected)}")

    model.to(device)
    model.eval()
    return model, checkpoint


def run_test_on_checkpoint(
    checkpoint_path: str,
    test_loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    num_classes: int = NUM_CLASSES,
    limit_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Load the best checkpoint and evaluate it on the test split.

    Returns a JSON-ready dict with the fields required by the lab report.
    """
    model, checkpoint = load_finetuned_model(checkpoint_path, device=device, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()

    metrics = test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp=amp,
        limit_batches=limit_batches,
    )
    total_params, trainable_params = count_parameters(model)

    return {
        "model": checkpoint.get("model_name", "unknown"),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_val_accuracy": round(float(checkpoint.get("val_accuracy", float("nan"))), 4),
        "test_loss": round(float(metrics["loss"]), 4),
        "test_accuracy": round(float(metrics["accuracy"]), 4),
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "checkpoint": str(checkpoint_path),
        "test_samples": int(metrics["num_samples"]),
    }


# --------------------------------------------------------------------------------------
# Full training run
# --------------------------------------------------------------------------------------
def fit(
    model: nn.Module,
    loaders: Mapping[str, DataLoader],
    cfg: TrainConfig,
    device: torch.device,
    paths: Mapping[str, str],
    writer: Optional[Any] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Full fine-tuning loop with TensorBoard, CSV history and best-checkpoint saving.

    Args:
        model: the pretrained model (already on ``device`` with a 10-class head).
        loaders: dict with ``"train"`` / ``"val"`` (and optionally ``"test"``) DataLoaders.
        cfg: the run configuration.
        device: compute device.
        paths: output paths from :func:`utils.resolve_paths`.
        writer: optional ``SummaryWriter``. When ``None`` and ``cfg.tensorboard`` is set,
            a writer is created on ``paths["run_dir"]``.
        verbose: print the per-epoch report.

    Returns:
        Dict with ``history``, ``best_val_accuracy``, ``best_epoch``, ``checkpoint_path``,
        ``training_time_seconds`` and the model summary.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model, name=cfg.optimizer, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, name=cfg.scheduler, epochs=cfg.epochs)

    use_amp = bool(cfg.amp and amp_supported(device))
    scaler = make_grad_scaler(use_amp) if use_amp else None

    owns_writer = False
    tb_message = "TensorBoard logging disabled (--no-tensorboard)."
    if writer is None and cfg.tensorboard:
        writer, owns_writer, tb_message = make_summary_writer(paths["run_dir"], enabled=True)
    if verbose:
        print(f"[engine] {tb_message}", flush=True)

    if writer is not None:
        writer.add_text("config", "```\n" + str(cfg.to_dict()) + "\n```", 0)

    summary = get_model_summary(model, cfg.model_name)
    total_params, trainable_params = count_parameters(model)
    head = summary["classifier"]

    if verbose:
        print(f"Model: {summary['display_name']}")
        print(f"Source: {summary['source']} | ImageNet-1K weights: {summary['weights']}")
        print(f"Classifier/head: {head}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
        print(
            f"Optimizer: {cfg.optimizer} | lr={cfg.lr} | weight_decay={cfg.weight_decay} "
            f"| scheduler={cfg.scheduler}"
        )
        print(f"Batch size: {cfg.batch_size} | Epochs: {cfg.epochs} | AMP: {'on' if use_amp else 'off'}")
        print("-" * 78, flush=True)

    history: List[Dict[str, Any]] = []
    best_val_accuracy = -1.0
    best_epoch = -1
    best_train_accuracy = float("nan")
    best_val_loss = float("nan")
    training_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            amp=use_amp,
            epoch=epoch,
            epochs=cfg.epochs,
            limit_batches=cfg.limit_train_batches,
            log_interval=cfg.log_interval,
        )
        val_metrics = validate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            amp=use_amp,
            limit_batches=cfg.limit_eval_batches,
        )
        epoch_time = train_metrics["epoch_time"] + val_metrics["eval_time"]
        lr = current_lr(optimizer)

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 4),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 4),
            "learning_rate": lr,
            "epoch_time": round(epoch_time, 3),
        }
        history.append(row)
        append_csv_row(paths["history_csv"], row)  # incremental: a crash keeps the history

        if writer is not None:
            writer.add_scalar("Loss/train", row["train_loss"], epoch)
            writer.add_scalar("Loss/val", row["val_loss"], epoch)
            writer.add_scalar("Accuracy/train", row["train_accuracy"], epoch)
            writer.add_scalar("Accuracy/val", row["val_accuracy"], epoch)
            writer.add_scalar("LearningRate", lr, epoch)
            writer.add_scalar("Time/epoch_seconds", epoch_time, epoch)

        if verbose:
            print(f"Epoch {epoch:02d}/{cfg.epochs}")
            print(f"Train Loss: {row['train_loss']:.4f}")
            print(f"Train Acc : {row['train_accuracy']:.2f}%")
            print(f"Val Loss  : {row['val_loss']:.4f}")
            print(f"Val Acc   : {row['val_accuracy']:.2f}%")
            extra = gpu_memory_summary(device)
            print(
                f"LR        : {lr:.6f} | time {format_seconds(epoch_time)}"
                + (f" | {extra}" if extra else "")
            )
            print("-" * 78, flush=True)

        # ---- best model selection: validation accuracy only (never the test set) ----
        if row["val_accuracy"] > best_val_accuracy:
            best_val_accuracy = row["val_accuracy"]
            best_epoch = epoch
            best_train_accuracy = row["train_accuracy"]
            best_val_loss = row["val_loss"]
            save_checkpoint(
                paths["checkpoint_path"],
                {
                    "epoch": epoch,
                    "model_name": cfg.model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "val_accuracy": row["val_accuracy"],
                    "val_loss": row["val_loss"],
                    "train_accuracy": row["train_accuracy"],
                    "train_loss": row["train_loss"],
                    "args": cfg.to_dict(),
                    "model_summary": summary,
                    "history": history,
                    "class_names": list(CIFAR10_CLASSES),
                    "num_classes": NUM_CLASSES,
                },
            )
            if verbose:
                print(f"  -> saved new best checkpoint ({paths['checkpoint_path']}, val_acc={best_val_accuracy:.2f}%)")

        if scheduler is not None:
            scheduler.step()

    training_time = time.perf_counter() - training_start

    if writer is not None:
        writer.add_hparams(
            {
                "model": cfg.model_name,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "optimizer": cfg.optimizer,
                "epochs": cfg.epochs,
                "freeze_backbone": int(cfg.freeze_backbone),
                "seed": cfg.seed,
            },
            {"hparam/best_val_accuracy": best_val_accuracy, "hparam/best_epoch": best_epoch},
        )
        writer.flush()
        if owns_writer:
            writer.close()

    # Guarantee a fully written history file even if the run was interrupted early.
    write_history_csv(paths["history_csv"], history)

    return {
        "model": cfg.model_name,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_accuracy, 4),
        "best_val_loss": best_val_loss,
        "best_train_accuracy": best_train_accuracy,
        "checkpoint_path": paths["checkpoint_path"],
        "training_time_seconds": round(training_time, 2),
        "training_time_minutes": round(training_time / 60.0, 4),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_summary": summary,
        "tensorboard_used": writer is not None,
        "tensorboard_message": tb_message,
    }
