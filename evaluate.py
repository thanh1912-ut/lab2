#!/usr/bin/env python
"""Evaluate a fine-tuned checkpoint on the CIFAR-10 test split (Lab 2).

Examples
--------
Evaluate the best checkpoint of a model::

    python evaluate.py --model resnet18

Evaluate an explicit checkpoint file::

    python evaluate.py --checkpoint checkpoints/vgg16_best.pt

Also evaluate the 5,000-image validation split::

    python evaluate.py --model resnet18 --split val

Writes ``results/<model>_test.json`` (or the file given by ``--output-json``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.nn as nn

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_NUM_WORKERS,
    DEFAULT_RESULTS_DIR,
)
from dataset import build_dataloaders
from engine import load_finetuned_model, test, validate
from models import SUPPORTED_MODELS, normalize_model_name
from utils import (
    count_parameters,
    describe_device,
    format_parameters,
    get_device,
    load_checkpoint,
    load_json,
    save_json,
    set_seed,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned CIFAR-10 checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None, choices=list(SUPPORTED_MODELS),
                        help="Model name (used to locate checkpoints/<model>_best.pt).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit checkpoint path (default: <checkpoint-dir>/<model>_best.pt).")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory holding '<model>_best.pt'.")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="CIFAR-10 directory.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR, help="Output folder for the JSON.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Explicit output JSON path (default: <results-dir>/<model>_test.json).")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"], help="Split to evaluate.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size (32 for the lab).")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the val split (must match training).")
    parser.add_argument("--val-split", type=int, default=5_000, help="Validation size used during training.")
    parser.add_argument("--no-download", dest="download", action="store_false", default=True,
                        help="Never download CIFAR-10; fail if it is not already in --data-dir.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
                        help="Compute device.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Mixed precision (CUDA).")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision.")
    parser.add_argument("--limit-eval-batches", type=int, default=None, help="Debug: only N batches.")
    parser.add_argument("--no-save", action="store_true", help="Do not write the result JSON.")
    return parser


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
    """Work out which checkpoint file to evaluate, with clear errors."""
    if args.checkpoint:
        return args.checkpoint
    if not args.model:
        raise SystemExit(
            "Provide either --model (to use <checkpoint-dir>/<model>_best.pt) or --checkpoint PATH.\n"
            f"Supported models: {', '.join(SUPPORTED_MODELS)}"
        )
    return str(Path(args.checkpoint_dir) / f"{normalize_model_name(args.model)}_best.pt")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    set_seed(args.seed)
    device = get_device(args.device)
    print(describe_device(device))

    checkpoint_path = resolve_checkpoint_path(args)
    try:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        print(f"[evaluate] ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        model_name = normalize_model_name(args.model) if args.model else normalize_model_name(checkpoint["model_name"])
    except (KeyError, ValueError) as exc:
        print(f"[evaluate] ERROR: cannot determine the model name ({exc})", file=sys.stderr)
        return 1

    if args.model and normalize_model_name(checkpoint.get("model_name", args.model)) != model_name:
        print(
            f"[evaluate] ERROR: checkpoint '{checkpoint_path}' was trained for "
            f"'{checkpoint.get('model_name')}' but --model {args.model} was given.",
            file=sys.stderr,
        )
        return 1

    print(f"Model: {model_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        f"Checkpoint info -> epoch {checkpoint.get('epoch')} | "
        f"val_acc {float(checkpoint.get('val_accuracy', float('nan'))):.2f}% | "
        f"train_acc {float(checkpoint.get('train_accuracy', float('nan'))):.2f}%"
    )

    try:
        loaders, datasets = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            seed=args.seed,
            num_workers=args.num_workers,
            download=args.download,
        )
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"[evaluate] ERROR while preparing CIFAR-10 in '{args.data_dir}':\n{exc}", file=sys.stderr)
        return 1
    loader = loaders[args.split]

    try:
        model, _ = load_finetuned_model(checkpoint_path, device=device, num_classes=10)
    except (RuntimeError, ValueError, KeyError, ImportError) as exc:
        print(f"[evaluate] ERROR while rebuilding '{model_name}' from the checkpoint:\n{exc}", file=sys.stderr)
        return 1

    criterion = nn.CrossEntropyLoss()

    if args.split == "test":
        metrics = test(model, loader, criterion, device, amp=args.amp, limit_batches=args.limit_eval_batches)
    else:
        metrics = validate(model, loader, criterion, device, amp=args.amp, limit_batches=args.limit_eval_batches)

    total_params, trainable_params = count_parameters(model)

    print("=" * 78)
    print(f"Split        : {args.split} ({metrics['num_samples']:,} images)")
    print(f"Loss         : {metrics['loss']:.4f}")
    print(f"Accuracy     : {metrics['accuracy']:.2f}%")
    print(f"Parameters   : {format_parameters(total_params)}")
    print("=" * 78)

    if not args.no_save:
        json_path = args.output_json or str(Path(args.results_dir) / f"{model_name}_{args.split}.json")
        payload: Dict[str, Any] = {}
        if Path(json_path).is_file():
            try:  # keep additional fields written during training
                payload.update(load_json(json_path))
            except (ValueError, OSError):
                payload = {}

        payload.update(
            {
                "model": model_name,
                "best_epoch": int(checkpoint.get("epoch", -1)),
                "best_val_accuracy": round(float(checkpoint.get("val_accuracy", float("nan"))), 4),
                "parameters": total_params,
                "trainable_parameters": trainable_params,
                "checkpoint": checkpoint_path,
            }
        )
        if args.split == "test":
            payload["test_loss"] = round(float(metrics["loss"]), 4)
            payload["test_accuracy"] = round(float(metrics["accuracy"]), 4)
            payload["test_samples"] = int(metrics["num_samples"])
        else:
            payload["val_loss"] = round(float(metrics["loss"]), 4)
            payload["val_accuracy_eval"] = round(float(metrics["accuracy"]), 4)
            payload["val_samples"] = int(metrics["num_samples"])

        save_json(json_path, payload)
        print(f"Saved: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
