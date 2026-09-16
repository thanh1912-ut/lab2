"""Utility helpers: reproducibility, device selection, parameter counting, I/O.

This module deliberately contains no model- or dataset-specific logic so that it can
be reused by ``train.py``, ``evaluate.py``, ``benchmark.py`` and ``inspect_model.py``.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "set_seed",
    "get_device",
    "describe_device",
    "count_parameters",
    "format_parameters",
    "ensure_dir",
    "resolve_paths",
    "save_checkpoint",
    "load_checkpoint",
    "save_json",
    "load_json",
    "write_history_csv",
    "append_csv_row",
    "read_csv_rows",
    "write_csv_rows",
    "get_model_size_mb",
    "AverageMeter",
    "format_seconds",
    "amp_supported",
    "make_grad_scaler",
    "autocast_context",
    "check_tensorboard",
    "make_summary_writer",
    "TENSORBOARD_ENV_VAR",
]


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Seed ``random``, ``numpy`` and ``torch`` (CPU + CUDA) for reproducibility.

    Args:
        seed: the seed used everywhere in the project (default 42).
        deterministic: when True, asks cuDNN for deterministic kernels. This is slower,
            so it is off by default; the train/val split is reproducible either way
            because it is driven by an explicit ``torch.Generator`` seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:  # pragma: no cover - tiny helper
    """DataLoader ``worker_init_fn`` so that augmentation is worker-reproducible."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------
def get_device(preferred: str = "auto") -> torch.device:
    """Return the compute device, preferring CUDA > MPS > CPU.

    Args:
        preferred: ``"auto"``, or one of ``"cuda"``, ``"mps"``, ``"cpu"``. An explicit
            request that is unavailable raises a clear error instead of silently
            falling back (fallbacks would make the benchmark hard to interpret).
    """
    preferred = (preferred or "auto").lower()

    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested (--device cuda) but torch.cuda.is_available() is False. "
                f"Installed torch={torch.__version__} (CUDA build: {torch.version.cuda}). "
                "On Google Colab select Runtime > Change runtime type > GPU."
            )
        return torch.device("cuda")

    if preferred == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested (--device mps) but it is not available on this machine.")
        return torch.device("mps")

    if preferred == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device '{preferred}'. Expected one of: auto, cuda, mps, cpu.")


def describe_device(device: torch.device) -> str:
    """Human readable one-liner describing the selected device."""
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total_gb = torch.cuda.get_device_properties(index).total_memory / (1024**3)
        return f"Device: cuda\nGPU: {name} ({total_gb:.1f} GB VRAM, CUDA {torch.version.cuda})"
    if device.type == "mps":
        return "Device: mps\nGPU: Apple Silicon (Metal Performance Shaders)"
    return "Device: cpu\nGPU: none (AMP disabled)"


# --------------------------------------------------------------------------------------
# Parameters / model size
# --------------------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_parameters(n: int) -> str:
    """Format a parameter count with thousands separators (e.g. ``11,181,642``)."""
    return f"{n:,}"


def get_model_size_mb(model_or_path: nn.Module | str | os.PathLike) -> float:
    """Approximate model size in MB.

    Accepts either an ``nn.Module`` (parameters + buffers are summed) or a path to a
    checkpoint file (its size on disk is used).
    """
    if isinstance(model_or_path, (str, os.PathLike)):
        path = Path(model_or_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cannot compute model size, file not found: {path}")
        return path.stat().st_size / (1024**2)

    total_bytes = 0
    for tensor in list(model_or_path.parameters()) + list(model_or_path.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024**2)


# --------------------------------------------------------------------------------------
# Filesystem helpers
# --------------------------------------------------------------------------------------
def ensure_dir(path: str | os.PathLike) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_paths(cfg: Any, create: bool = True) -> Dict[str, str]:
    """Build the standard output paths for a run from a :class:`config.TrainConfig`.

    Returns a dict with the checkpoint / history CSV / test JSON / TensorBoard run dir.
    """
    model_name = cfg.model_name
    run_name = cfg.effective_run_name

    checkpoint_dir = Path(cfg.checkpoint_dir)
    results_dir = Path(cfg.results_dir)
    runs_dir = Path(cfg.runs_dir) / run_name

    if create:
        ensure_dir(checkpoint_dir)
        ensure_dir(results_dir)
        ensure_dir(runs_dir)

    return {
        "checkpoint_path": str(checkpoint_dir / f"{model_name}_best.pt"),
        "history_csv": str(results_dir / f"{model_name}_history.csv"),
        "test_json": str(results_dir / f"{model_name}_test.json"),
        "run_dir": str(runs_dir),
    }


# --------------------------------------------------------------------------------------
# Checkpoint helpers
# --------------------------------------------------------------------------------------
def save_checkpoint(path: str | os.PathLike, payload: Mapping[str, Any]) -> Path:
    """Save a rich checkpoint dict (never a bare ``state_dict``) atomically."""
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(dict(payload), tmp)
    tmp.replace(p)  # atomic on POSIX -> a crash can never leave a truncated .pt
    return p


def load_checkpoint(path: str | os.PathLike, map_location: Any = "cpu") -> Dict[str, Any]:
    """Load a checkpoint produced by :func:`save_checkpoint` with clear error messages."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {p}\n"
            "Train a model first, e.g.:\n"
            "    python train.py --model resnet18 --epochs 20\n"
            "or point to the right file with --checkpoint."
        )
    try:
        obj = torch.load(p, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001 - re-raise with context
        raise RuntimeError(f"Failed to load checkpoint '{p}': {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError(
            f"Checkpoint '{p}' does not contain a dict. Expected keys: "
            "'model_state_dict', 'model_name', 'epoch', 'val_accuracy', ..."
        )
    if "model_state_dict" not in obj:
        raise KeyError(
            f"Checkpoint '{p}' has no 'model_state_dict' key (found: {sorted(obj.keys())}). "
            "This project always saves full checkpoints, not raw state_dicts."
        )
    return obj


# --------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------
def save_json(path: str | os.PathLike, data: Mapping[str, Any], indent: int = 4) -> Path:
    """Write ``data`` as pretty-printed JSON."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
        fh.write("\n")
    return p


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    """Read a JSON file, with an explicit error if it is missing."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------
def write_csv_rows(
    path: str | os.PathLike,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Optional[Sequence[str]] = None,
) -> Path:
    """Overwrite ``path`` with ``rows`` (used for ``*_history.csv`` and comparison CSV)."""
    p = Path(path)
    ensure_dir(p.parent)

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return p


def append_csv_row(path: str | os.PathLike, row: Mapping[str, Any]) -> Path:
    """Append one row to a CSV, writing the header when the file does not exist yet."""
    p = Path(path)
    ensure_dir(p.parent)
    file_exists = p.is_file() and p.stat().st_size > 0
    with p.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(dict(row))
    return p


def read_csv_rows(path: str | os.PathLike) -> List[Dict[str, str]]:
    """Read a CSV into a list of dicts (empty list when the file is absent)."""
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_history_csv(path: str | os.PathLike, history: Sequence[Mapping[str, Any]]) -> Path:
    """Write the per-epoch training history with the columns required by the lab."""
    fieldnames = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
        "epoch_time",
    ]
    return write_csv_rows(path, history, fieldnames=fieldnames)


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------
class AverageMeter:
    """Track a running mean of a scalar metric (loss, accuracy, ...)."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.name}: {self.avg:.4f}"


def format_seconds(seconds: float) -> str:
    """Format a duration as ``HH:MM:SS`` (or ``MM:SS`` when under an hour)."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class Timer:
    """Minimal context manager measuring wall-clock seconds."""

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed = time.perf_counter() - self.start

    @property
    def minutes(self) -> float:
        return self.elapsed / 60.0


# --------------------------------------------------------------------------------------
# Mixed precision (AMP) - version tolerant
# --------------------------------------------------------------------------------------
def amp_supported(device: torch.device) -> bool:
    """AMP (float16 autocast + GradScaler) is only used on CUDA devices."""
    return device.type == "cuda" and torch.cuda.is_available()


def make_grad_scaler(enabled: bool) -> Any:
    """Create a ``GradScaler`` with the modern ``torch.amp`` API and a legacy fallback.

    PyTorch >= 2.4 exposes ``torch.amp.GradScaler(device, ...)``; older versions only
    have ``torch.cuda.amp.GradScaler``. ``enabled=False`` makes it a harmless no-op,
    which is what happens on CPU/MPS.
    """
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:  # older signature: torch.amp.GradScaler(device=..., enabled=...)
            return torch.amp.GradScaler(device="cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover - legacy torch


def autocast_context(device: torch.device, enabled: bool) -> Any:
    """Return the autocast context manager for the given device (disabled on CPU/MPS)."""
    device_type = "cuda" if device.type == "cuda" else "cpu"
    enabled = bool(enabled and device_type == "cuda")
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device_type, dtype=torch.float16, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)  # pragma: no cover - legacy torch


def gpu_memory_summary(device: torch.device) -> str:
    """Short VRAM usage string (empty when not on CUDA)."""
    if device.type != "cuda":
        return ""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    return f"VRAM: {allocated:.2f} GB allocated / {peak:.2f} GB peak"


# --------------------------------------------------------------------------------------
# TensorBoard availability
# --------------------------------------------------------------------------------------
# ``from torch.utils.tensorboard import SummaryWriter`` is not safe everywhere:
# TensorBoard's compat layer may import TensorFlow, and a broken TensorFlow / Keras /
# pyarrow installation in the environment crashes the interpreter with a native SIGSEGV
# that Python cannot catch. The import is therefore validated once, out-of-process,
# before it is done in-process.
_TB_STATUS: Optional[bool] = None
_TB_REASON: str = ""

TENSORBOARD_ENV_VAR = "LAB2_TENSORBOARD"  # auto (default) | 1 (assume ok) | 0 (disable)


def check_tensorboard(force: bool = False, timeout: int = 300) -> Tuple[bool, str]:
    """Return ``(available, reason)`` for TensorBoard logging, cached per process.

    Controlled by the ``LAB2_TENSORBOARD`` environment variable:

    * ``auto`` (default): probe ``from torch.utils.tensorboard import SummaryWriter`` in a
      subprocess. Because the probe runs out-of-process, a native crash caused by a broken
      TensorFlow/Keras/pyarrow disables TensorBoard logging instead of killing training.
    * ``1``/``true``: import in-process without probing (fast path on known-good machines).
    * ``0``/``false``: disable TensorBoard entirely (same as ``--no-tensorboard``).
    """
    global _TB_STATUS, _TB_REASON
    if _TB_STATUS is not None and not force:
        return _TB_STATUS, _TB_REASON

    mode = os.environ.get(TENSORBOARD_ENV_VAR, "auto").strip().lower()

    if mode in {"0", "false", "off", "no"}:
        _TB_STATUS = False
        _TB_REASON = f"disabled by {TENSORBOARD_ENV_VAR}={mode}"
        return _TB_STATUS, _TB_REASON

    if mode in {"1", "true", "on", "yes"}:
        try:
            importlib.import_module("torch.utils.tensorboard")

            _TB_STATUS, _TB_REASON = True, "imported in-process"
        except Exception as exc:  # noqa: BLE001
            _TB_STATUS, _TB_REASON = False, f"in-process import failed: {exc!r}"
        return _TB_STATUS, _TB_REASON

    probe = "from torch.utils.tensorboard import SummaryWriter"
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        _TB_STATUS, _TB_REASON = False, f"probe timed out after {timeout}s"
        return _TB_STATUS, _TB_REASON
    except OSError as exc:
        _TB_STATUS, _TB_REASON = False, f"probe could not start: {exc}"
        return _TB_STATUS, _TB_REASON

    if completed.returncode == 0:
        _TB_STATUS, _TB_REASON = True, "probe succeeded"
        return _TB_STATUS, _TB_REASON

    if completed.returncode < 0:
        detail = f"the probe process was killed by signal {-completed.returncode} (native crash)"
    else:
        lines = (completed.stderr or "").strip().splitlines()
        detail = f"probe exited with code {completed.returncode}: {lines[-1] if lines else 'no stderr'}"

    _TB_STATUS = False
    _TB_REASON = (
        f"{detail}. Importing torch.utils.tensorboard pulls in TensorFlow/Keras/pyarrow, so a "
        f"broken install of one of them can crash the interpreter. Training continues without "
        f"TensorBoard; fix the environment (e.g. `pip install -U tensorflow` or "
        f"`pip uninstall -y tensorflow keras`) or set {TENSORBOARD_ENV_VAR}=1 to force the "
        f"in-process import."
    )
    return _TB_STATUS, _TB_REASON


def make_summary_writer(log_dir: str | os.PathLike, enabled: bool = True) -> Tuple[Any, bool, str]:
    """Create a TensorBoard ``SummaryWriter`` when possible.

    Returns ``(writer_or_None, owns_writer, message)``. ``writer`` is ``None`` when
    TensorBoard is disabled or unavailable; callers must then simply skip logging.
    """
    if not enabled:
        return None, False, "TensorBoard logging disabled (--no-tensorboard)."

    available, reason = check_tensorboard()
    if not available:
        return None, False, f"TensorBoard logging disabled: {reason}"

    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415 - validated above

        ensure_dir(log_dir)
        return SummaryWriter(log_dir=str(log_dir)), True, f"TensorBoard logging to {log_dir}"
    except Exception as exc:  # noqa: BLE001 - logging must never break training
        return None, False, f"TensorBoard logging disabled: could not create SummaryWriter ({exc!r})"


def iter_batches(loader: Iterable[Any], limit: Optional[int] = None) -> Iterable[Any]:
    """Iterate a DataLoader, optionally stopping after ``limit`` batches (debug helper)."""
    if limit is None:
        yield from loader
        return
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        yield batch
