#!/usr/bin/env python
"""Benchmark the four backbones on CIFAR-10 with an identical recipe (Lab 2, requirement 19).

Every model is trained with the *same* configuration so the comparison is fair:

* identical CIFAR-10 45,000/5,000/10,000 split (same ``--seed``)
* identical transforms (RandomResizedCrop(224) + HFlip / Resize(256) + CenterCrop(224))
* AdamW, lr=0.005, weight_decay=1e-4, batch size 32, CrossEntropyLoss
* same number of epochs, full fine-tuning, AMP only as an implementation detail

It writes ``results/model_comparison.csv`` with the columns::

    model,total_params,trainable_params,best_epoch,best_val_accuracy,test_accuracy,training_time_minutes,model_size_mb

Usage
-----
::

    python benchmark.py --epochs 20 --batch-size 32 --lr 0.005
    python benchmark.py --models resnet18 vgg16 --epochs 5     # subset
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from config import (
    BENCHMARK_MODELS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OPTIMIZER,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RUNS_DIR,
    DEFAULT_SCHEDULER,
    DEFAULT_SEED,
    DEFAULT_VAL_SPLIT,
    DEFAULT_WEIGHT_DECAY,
    SUPPORTED_OPTIMIZERS,
    SUPPORTED_SCHEDULERS,
    TrainConfig,
)
from dataset import build_dataloaders, describe_dataset
from engine import fit, run_test_on_checkpoint
from models import SUPPORTED_MODELS, create_model, freeze_backbone, get_model_summary, unfreeze_model
from utils import (
    count_parameters,
    describe_device,
    format_parameters,
    format_seconds,
    get_device,
    get_model_size_mb,
    resolve_paths,
    save_json,
    set_seed,
    write_csv_rows,
)

COMPARISON_COLUMNS: List[str] = [
    "model",
    "total_params",
    "trainable_params",
    "best_epoch",
    "best_val_accuracy",
    "test_accuracy",
    "training_time_minutes",
    "model_size_mb",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare MobileNetV4-Small / VGG16 / ResNet18 / DenseNet121 on CIFAR-10.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", type=str, nargs="+", default=list(BENCHMARK_MODELS),
                        choices=list(SUPPORTED_MODELS), help="Models to benchmark, in order.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epochs per model (same for all).")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (same for all).")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (same for all).")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay (same for all).")
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER, choices=list(SUPPORTED_OPTIMIZERS),
                        help="Optimizer (same for all).")
    parser.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER, choices=list(SUPPORTED_SCHEDULERS),
                        help="LR schedule (same for all).")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Optional experiment: train heads only (not the main benchmark).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="CIFAR-10 directory.")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint folder.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Results folder.")
    parser.add_argument("--runs-dir", type=str, default=DEFAULT_RUNS_DIR, help="TensorBoard runs folder.")
    parser.add_argument("--comparison-csv", type=str, default=None,
                        help="Output CSV (default: <results-dir>/model_comparison.csv).")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--val-split", type=int, default=DEFAULT_VAL_SPLIT, help="Validation size (5,000).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed (identical for every model).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Mixed precision (CUDA).")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", default=True,
                        help="Disable TensorBoard logging.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Debug: N training batches per epoch.")
    parser.add_argument("--limit-eval-batches", type=int, default=None, help="Debug: N eval batches.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Keep going with the next model if one fails (default: stop).")
    parser.add_argument("--pandas-table", action="store_true",
                        help="Print the final comparison table with pandas (plain text by default).")
    return parser


def make_config(args: argparse.Namespace, model_name: str) -> TrainConfig:
    """Build the per-model config; only ``model_name`` differs between models."""
    return TrainConfig(
        model_name=model_name,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
        amp=args.amp,
        device=args.device,
        limit_train_batches=args.limit_train_batches,
        limit_eval_batches=args.limit_eval_batches,
        tensorboard=args.tensorboard,
    )


def benchmark_one_model(
    cfg: TrainConfig,
    device: torch.device,
    print_dataset_info: bool = False,
) -> Dict[str, Any]:
    """Train + test a single model with the shared recipe and return its comparison row."""
    # Same seed for every model => identical split, identical augmentation stream.
    set_seed(cfg.seed)

    loaders, datasets = build_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        seed=cfg.seed,
        num_workers=cfg.num_workers,
    )
    if print_dataset_info:
        print(describe_dataset(*datasets))
        print("-" * 78, flush=True)

    paths = resolve_paths(cfg, create=True)
    model = create_model(cfg.model_name, num_classes=10, pretrained=cfg.pretrained)
    if cfg.freeze_backbone:
        freeze_backbone(model)
    else:
        unfreeze_model(model)
    model.to(device)

    total_params, trainable_params = count_parameters(model)
    summary = get_model_summary(model, cfg.model_name)
    print(f"### {summary['display_name']} | weights: {summary['weights']}")
    print(f"Total parameters: {format_parameters(total_params)} | "
          f"trainable: {format_parameters(trainable_params)}")

    start = time.perf_counter()
    result = fit(model, loaders, cfg, device, paths, verbose=True)
    elapsed_minutes = (time.perf_counter() - start) / 60.0

    test_result = run_test_on_checkpoint(
        checkpoint_path=paths["checkpoint_path"],
        test_loader=loaders["test"],
        device=device,
        amp=cfg.amp,
        limit_batches=cfg.limit_eval_batches,
    )

    # Persist the per-model test JSON exactly like train.py does.
    test_result["training_time_minutes"] = round(elapsed_minutes, 4)
    test_result["history_csv"] = paths["history_csv"]
    test_result["tensorboard_dir"] = paths["run_dir"]
    save_json(paths["test_json"], test_result)

    row: Dict[str, Any] = {
        "model": cfg.model_name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_epoch": result["best_epoch"],
        "best_val_accuracy": result["best_val_accuracy"],
        "test_accuracy": test_result["test_accuracy"],
        "training_time_minutes": round(elapsed_minutes, 4),
        "model_size_mb": round(get_model_size_mb(paths["checkpoint_path"]), 2),
    }

    # free memory before the next backbone
    del model, loaders
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return row


def print_comparison_table(rows: List[Dict[str, Any]], use_pandas: bool = False) -> None:
    """Print the comparison table.

    The plain-text table is the default; importing pandas is opt-in (``--pandas-table``)
    because in some environments its native dependency chain (pyarrow/tensorflow) is
    broken and can crash the interpreter. The CSV file itself is always written with the
    standard library and is unaffected.
    """
    if not rows:
        return

    if use_pandas:
        try:
            import pandas as pd  # optional dependency

            frame = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
            print("\n" + frame.to_string(index=False))
            best = frame.loc[frame["test_accuracy"].idxmax()]
            print(f"\nBest test accuracy: {best['model']} ({best['test_accuracy']:.2f}%)")
            return
        except ImportError:
            print("[benchmark] pandas is not installed - falling back to the plain-text table.")

    headers = COMPARISON_COLUMNS
    widths = [max(len(str(h)), *(len(str(r.get(h, ""))) for r in rows)) for h in headers]
    print("\n" + "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(w) for h, w in zip(headers, widths)))
    best = max(rows, key=lambda r: float(r["test_accuracy"]))
    print(f"\nBest test accuracy: {best['model']} ({float(best['test_accuracy']):.2f}%)")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    device = get_device(args.device)
    results_dir = Path(args.results_dir)
    comparison_csv = args.comparison_csv or str(results_dir / "model_comparison.csv")

    print("=" * 78)
    print("Lab 2 - Benchmark of 4 ImageNet-1K pretrained backbones on CIFAR-10")
    print("=" * 78)
    print(describe_device(device))
    print(f"Models          : {', '.join(args.models)}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Optimizer       : {args.optimizer} (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Scheduler       : {args.scheduler}")
    print("Loss            : CrossEntropyLoss")
    print(f"Fine-tuning     : {'classification head only' if args.freeze_backbone else 'full network'}")
    print(f"Seed            : {args.seed} (identical for every model)")
    print(f"AMP             : {'on' if args.amp else 'off'}")
    print("=" * 78, flush=True)

    rows: List[Dict[str, Any]] = []
    failures: List[str] = []
    benchmark_start = time.perf_counter()

    for index, model_name in enumerate(args.models, start=1):
        print("\n" + "#" * 78)
        print(f"# [{index}/{len(args.models)}] {model_name}")
        print("#" * 78, flush=True)
        cfg = make_config(args, model_name)
        try:
            row = benchmark_one_model(cfg, device, print_dataset_info=(index == 1))
            rows.append(row)
            print(f"--> {model_name}: test accuracy {float(row['test_accuracy']):.2f}% "
                  f"in {float(row['training_time_minutes']):.2f} min")
        except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
            print(f"[benchmark] ERROR on '{model_name}': {exc}", file=sys.stderr)
            failures.append(model_name)
            if not args.continue_on_error:
                print("[benchmark] Stopping. Use --continue-on-error to skip failing models.", file=sys.stderr)
                break

    if rows:
        write_csv_rows(comparison_csv, rows, fieldnames=COMPARISON_COLUMNS)
        print_comparison_table(rows, use_pandas=args.pandas_table)
        print(f"\nSaved comparison CSV: {comparison_csv}")

    total_minutes = (time.perf_counter() - benchmark_start) / 60.0
    print(f"Total benchmark time: {format_seconds(total_minutes * 60)} ({total_minutes:.2f} minutes)")
    if failures:
        print(f"Failed models: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
