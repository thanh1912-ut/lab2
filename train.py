#!/usr/bin/env python
"""Train one pretrained backbone on CIFAR-10 (Lab 2 main entry point).

Examples
--------
Main benchmark recipe (ImageNet-1K pretrained, full fine-tuning, AdamW lr=0.005, bs=32)::

    python train.py --model resnet18 --epochs 20 --batch-size 32 --lr 0.005

All four backbones::

    python train.py --model mobilenetv4_small --epochs 20
    python train.py --model vgg16            --epochs 20
    python train.py --model resnet18         --epochs 20
    python train.py --model densenet121      --epochs 20

Optional experiment (head-only training)::

    python train.py --model resnet18 --epochs 20 --freeze-backbone

Outputs: ``checkpoints/<model>_best.pt``, ``results/<model>_history.csv``,
``results/<model>_test.json``, ``runs/<model>/`` (TensorBoard).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import (
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
from models import SUPPORTED_MODELS, create_model, freeze_backbone, normalize_model_name, unfreeze_model
from utils import (
    Timer,
    describe_device,
    format_parameters,
    format_seconds,
    get_device,
    resolve_paths,
    save_json,
    set_seed,
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Command line interface of ``train.py``."""
    parser = argparse.ArgumentParser(
        description="Fine-tune an ImageNet-1K pretrained CNN on CIFAR-10 (Lab 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / data
    parser.add_argument("--model", type=str, default="resnet18", choices=list(SUPPORTED_MODELS),
                        help="Pretrained backbone to fine-tune.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="CIFAR-10 directory (downloaded automatically if missing).")
    parser.add_argument("--val-split", type=int, default=DEFAULT_VAL_SPLIT,
                        help="Number of CIFAR-10 train images held out for validation.")
    parser.add_argument("--no-download", dest="download", action="store_false", default=True,
                        help="Never download CIFAR-10; fail if it is not already in --data-dir.")

    # Optimisation (mandatory benchmark values are the defaults)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate (AdamW).")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument("--optimizer", type=str, default=DEFAULT_OPTIMIZER, choices=list(SUPPORTED_OPTIMIZERS),
                        help="Optimizer (adamw for the main benchmark).")
    parser.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER, choices=list(SUPPORTED_SCHEDULERS),
                        help="Optional LR schedule; 'none' keeps a constant lr for a simple recipe.")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="CrossEntropyLoss label smoothing (0.0 = plain CrossEntropyLoss).")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Gradient norm clipping value (0.0 = disabled).")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Train only the classification head (optional experiment; default is full fine-tuning).")

    # Runtime
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed (same for all backbones).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device; 'auto' prefers CUDA > MPS > CPU.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True,
                        help="Use mixed precision on CUDA (default).")
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable mixed precision (always off on CPU/MPS).")

    # Outputs
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint folder.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="CSV/JSON results folder.")
    parser.add_argument("--runs-dir", type=str, default=DEFAULT_RUNS_DIR, help="TensorBoard runs folder.")
    parser.add_argument("--run-name", type=str, default="", help="TensorBoard run name (default: model name).")
    parser.add_argument("--no-tensorboard", dest="tensorboard", action="store_false", default=True,
                        help="Disable TensorBoard logging.")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip the final test-set evaluation after training.")
    parser.add_argument("--log-interval", type=int, default=0,
                        help="Print a batch-level line every N batches (0 = only per-epoch report).")

    # Debug helpers (useful for smoke tests / CI; they do not change the recipe)
    parser.add_argument("--limit-train-batches", type=int, default=None,
                        help="Debug: use only the first N training batches per epoch.")
    parser.add_argument("--limit-eval-batches", type=int, default=None,
                        help="Debug: use only the first N validation/test batches.")
    return parser


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    """Convert parsed CLI arguments into a validated :class:`config.TrainConfig`."""
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive, got {args.lr}.")
    if args.num_workers < 0:
        raise ValueError(f"--num-workers must be >= 0, got {args.num_workers}.")
    if not 0 < args.val_split < 50_000:
        raise ValueError(f"--val-split must be in (0, 50000), got {args.val_split}.")
    if args.freeze_backbone and args.scheduler != "none":
        print("[train] Warning: --freeze-backbone with a scheduler is an unusual combination.")

    return TrainConfig(
        model_name=normalize_model_name(args.model),
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
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        val_split=args.val_split,
        seed=args.seed,
        pretrained=True,  # this lab never trains from scratch
        download=args.download,
        freeze_backbone=args.freeze_backbone,
        amp=args.amp,
        device=args.device,
        limit_train_batches=args.limit_train_batches,
        limit_eval_batches=args.limit_eval_batches,
        tensorboard=args.tensorboard,
        run_name=args.run_name,
        log_interval=args.log_interval,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = config_from_args(args)

    print("=" * 78)
    print("Lab 2 - Fine-tuning pretrained CNNs on CIFAR-10")
    print("=" * 78)
    print(f"Model: {cfg.model_name}")
    print(f"Pretrained: ImageNet-1K (pretrained=True) | fine-tuning: "
          f"{'classification head only' if cfg.freeze_backbone else 'full network'}")
    print(f"Seed: {cfg.seed}")

    # Reproducibility + device
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(describe_device(device))

    paths = resolve_paths(cfg, create=True)

    # Data (CIFAR-10 is downloaded automatically when missing, unless --no-download)
    try:
        loaders, datasets = build_dataloaders(
            data_dir=cfg.data_dir,
            batch_size=cfg.batch_size,
            val_split=cfg.val_split,
            seed=cfg.seed,
            num_workers=cfg.num_workers,
            download=cfg.download,
        )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"[train] ERROR while preparing CIFAR-10 in '{cfg.data_dir}':\n{exc}", file=sys.stderr)
        return 1

    print(describe_dataset(*datasets))
    print("-" * 78, flush=True)

    # Model (ImageNet-1K pretrained weights, 10-class head)
    try:
        model = create_model(cfg.model_name, num_classes=10, pretrained=cfg.pretrained)
    except (RuntimeError, ValueError, ImportError, KeyError) as exc:
        print(f"[train] ERROR while creating model '{cfg.model_name}':\n{exc}", file=sys.stderr)
        return 1

    if cfg.freeze_backbone:
        freeze_backbone(model)
    else:
        unfreeze_model(model)
    model.to(device)

    # Parameter counts, head, optimizer and AMP status are printed by engine.fit()
    # right before the first epoch (both train.py and benchmark.py share that report).

    # Train
    result: Dict[str, Any] = {}
    try:
        with Timer() as timer:
            result = fit(model, loaders, cfg, device, paths, verbose=True)
        print(f"Training finished in {format_seconds(timer.elapsed)} "
              f"({timer.minutes:.2f} minutes)")
        print(f"Best epoch: {result['best_epoch']} | Best val accuracy: {result['best_val_accuracy']:.2f}%")
    except KeyboardInterrupt:
        print("\n[train] Interrupted by user - keeping the best checkpoint written so far.")
        if not Path(paths["checkpoint_path"]).is_file():
            print("[train] No checkpoint was written yet (interrupted during the first epoch).")
            return 130
    except (RuntimeError, ValueError) as exc:
        print(f"\n[train] ERROR while training '{cfg.model_name}': {exc}", file=sys.stderr)
        if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
            print(
                "[train] CUDA out of memory. Try: --batch-size 16 --amp, "
                "or free the GPU (`nvidia-smi`) and retry.",
                file=sys.stderr,
            )
            traceback.print_exc()
            return 2
        return 1

    # Final test on the best checkpoint (validation is never used for this)
    if not args.skip_test:
        print("-" * 78)
        print(f"[train] Loading best checkpoint: {paths['checkpoint_path']}")
        try:
            test_result = run_test_on_checkpoint(
                checkpoint_path=paths["checkpoint_path"],
                test_loader=loaders["test"],
                device=device,
                amp=cfg.amp,
                limit_batches=cfg.limit_eval_batches,
            )
        except FileNotFoundError as exc:
            print(f"[train] {exc}", file=sys.stderr)
            return 1

        test_result["training_time_minutes"] = result.get("training_time_minutes")
        test_result["history_csv"] = paths["history_csv"]
        test_result["tensorboard_dir"] = paths["run_dir"]
        if cfg.freeze_backbone:
            test_result["note"] = "classification head only (--freeze-backbone), backbone frozen"
        save_json(paths["test_json"], test_result)

        print("=" * 78)
        print(f"Test Loss    : {test_result['test_loss']:.4f}")
        print(f"Test Accuracy: {test_result['test_accuracy']:.2f}%")
        print(f"Best epoch   : {test_result['best_epoch']} "
              f"(val acc {test_result['best_val_accuracy']:.2f}%)")
        print(f"Parameters   : {format_parameters(test_result['parameters'])}")
        print(f"Saved: {paths['test_json']}")
        print(f"TensorBoard: tensorboard --logdir {cfg.runs_dir}")
        print("=" * 78)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
