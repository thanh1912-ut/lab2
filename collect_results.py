#!/usr/bin/env python
"""Collect the per-model results into ``results/model_comparison.csv`` (no re-training).

``benchmark.py`` writes the comparison table itself, but if you trained the four backbones
with four separate ``train.py`` runs (the usual Colab workflow, one model per session) the
per-model ``results/<model>_test.json`` files exist without the summary table. This script
reads them back and builds the table, so nothing has to be trained again.

Columns produced (same as ``benchmark.py``)::

    model,total_params,trainable_params,best_epoch,best_val_accuracy,test_accuracy,training_time_minutes,model_size_mb

Usage
-----
::

    python collect_results.py                                   # ./results + ./checkpoints
    python collect_results.py --results-dir ./results --checkpoint-dir ./checkpoints
    python collect_results.py --pandas-table                    # prettier console output
    python collect_results.py --models resnet18 vgg16           # subset

Missing values are left empty rather than invented: if a model has no ``*_test.json`` it is
simply skipped and reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import BENCHMARK_MODELS, DEFAULT_CHECKPOINT_DIR, DEFAULT_RESULTS_DIR
from models import SUPPORTED_MODELS, normalize_model_name
from utils import format_parameters, load_json, write_csv_rows

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
        description="Build results/model_comparison.csv from existing per-model results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Folder with <model>_test.json / <model>_history.csv.")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Folder with <model>_best.pt (used for the size column).")
    parser.add_argument("--models", type=str, nargs="+", default=list(BENCHMARK_MODELS),
                        choices=list(SUPPORTED_MODELS), help="Models to include, in order.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV (default: <results-dir>/model_comparison.csv).")
    parser.add_argument("--pandas-table", action="store_true",
                        help="Print the table with pandas instead of plain text.")
    return parser


def read_history_best(results_dir: Path, model: str) -> Dict[str, Any]:
    """Fallback: best validation row from ``<model>_history.csv`` when the JSON lacks it."""
    from utils import read_csv_rows

    rows = read_csv_rows(results_dir / f"{model}_history.csv")
    if not rows:
        return {}
    try:
        best = max(rows, key=lambda r: float(r["val_accuracy"]))
    except (KeyError, ValueError):
        return {}
    return {
        "best_epoch": int(float(best.get("epoch", 0))),
        "best_val_accuracy": round(float(best["val_accuracy"]), 4),
    }


def collect_one(model: str, results_dir: Path, checkpoint_dir: Path) -> Optional[Dict[str, Any]]:
    """Build one comparison row from the artefacts of ``model`` (None when nothing found)."""
    json_path = results_dir / f"{model}_test.json"
    data: Dict[str, Any] = {}
    if json_path.is_file():
        try:
            data = load_json(json_path)
        except (ValueError, OSError) as exc:
            print(f"[collect] warning: cannot read {json_path}: {exc}", file=sys.stderr)

    if not data:
        # No test JSON: still report what the history CSV knows, so the row is not lost.
        history = read_history_best(results_dir, model)
        if not history:
            print(f"[collect] no results for '{model}' in {results_dir} - skipped")
            return None
        data = history

    for key, value in read_history_best(results_dir, model).items():
        data.setdefault(key, value)

    checkpoint = checkpoint_dir / f"{model}_best.pt"
    model_size = round(checkpoint.stat().st_size / 1024**2, 2) if checkpoint.is_file() else ""

    row: Dict[str, Any] = {
        "model": model,
        "total_params": data.get("parameters", ""),
        "trainable_params": data.get("trainable_parameters", ""),
        "best_epoch": data.get("best_epoch", ""),
        "best_val_accuracy": data.get("best_val_accuracy", ""),
        "test_accuracy": data.get("test_accuracy", ""),
        "training_time_minutes": data.get("training_time_minutes", ""),
        "model_size_mb": model_size,
    }
    return row


def print_table(rows: List[Dict[str, Any]], use_pandas: bool = False) -> None:
    if not rows:
        return
    if use_pandas:
        try:
            import pandas as pd  # optional

            print("\n" + pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_string(index=False))
            return
        except ImportError:
            print("[collect] pandas is not installed - falling back to plain text.")

    widths = [max(len(str(h)), *(len(str(r.get(h, ""))) for r in rows)) for h in COMPARISON_COLUMNS]
    print("\n" + "  ".join(str(h).ljust(w) for h, w in zip(COMPARISON_COLUMNS, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(w) for h, w in zip(COMPARISON_COLUMNS, widths)))


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    output = Path(args.output) if args.output else results_dir / "model_comparison.csv"

    print("=" * 78)
    print("Collect results -> model_comparison.csv (no training)")
    print("=" * 78)
    print(f"results dir    : {results_dir}")
    print(f"checkpoint dir : {checkpoint_dir}")

    if not results_dir.is_dir():
        print(f"[collect] ERROR: results dir '{results_dir}' does not exist.\n"
              "Train at least one model first, e.g. python train.py --model resnet18 --epochs 20",
              file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    for name in args.models:
        row = collect_one(normalize_model_name(name), results_dir, checkpoint_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"[collect] ERROR: no '*_test.json' or '*_history.csv' found in '{results_dir}'.",
              file=sys.stderr)
        return 1

    write_csv_rows(output, rows, fieldnames=COMPARISON_COLUMNS)
    print_table(rows, use_pandas=args.pandas_table)

    ranked = [r for r in rows if r["test_accuracy"] != ""]
    if ranked:
        best = max(ranked, key=lambda r: float(r["test_accuracy"]))
        print(f"\nBest test accuracy: {best['model']} ({float(best['test_accuracy']):.2f}%)")
    if rows and rows[0]["total_params"] != "":
        print(f"Largest model     : {max(rows, key=lambda r: int(r['total_params']))['model']} "
              f"({format_parameters(max(int(r['total_params']) for r in rows))} parameters)")

    print(f"\nSaved: {output}")
    print(f"Rows : {len(rows)}/{len(args.models)} models")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
