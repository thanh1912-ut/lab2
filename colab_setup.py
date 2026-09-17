#!/usr/bin/env python
"""Persist this Lab 2 project to Google Drive (made for Google Colab).

Goal: keep the working copy at ``/content/lab2`` (fast local disk) while every artefact -
dataset, checkpoints, CSV/JSON results, TensorBoard runs and a snapshot of the source -
lives in Google Drive, so nothing is lost when a Colab session is recycled.

What it does (idempotent - safe to re-run at any time):

1. checks that Drive is mounted (it cannot mount it for you: ``drive.mount()`` needs to run
   in a notebook cell, see below);
2. creates ``<drive-root>/{data,checkpoints,results,runs,code}``;
3. replaces the project's ``data/``, ``checkpoints/``, ``results/`` and ``runs/`` folders with
   symlinks to those Drive folders (existing local files are moved to Drive first, never
   deleted), so the default relative paths in ``train.py`` keep working unchanged;
4. copies the source files (``*.py``, ``README.md``, ``requirements.txt``, ``.gitignore``) into
   ``<drive-root>/code/`` as a snapshot/backup of the code.

Typical Colab usage
-------------------
Cell 1 (mount Drive - must be a notebook cell, not a subprocess)::

    from google.colab import drive
    drive.mount('/content/drive')

Cell 2::

    !git clone https://github.com/thanh1912-ut/lab2.git /content/lab2
    %cd /content/lab2
    !python colab_setup.py

Then train exactly as usual - everything is written to Drive::

    !python train.py --model resnet18 --epochs 20 --batch-size 32 --lr 0.005
    !python benchmark.py --epochs 20 --batch-size 32 --lr 0.005
    %load_ext tensorboard
    %tensorboard --logdir runs

Next session only needs cells 1 + 2 again (``git clone`` or restore the code snapshot from
``/content/drive/MyDrive/lab2/code``): the dataset and all results are already on Drive.

Local (non-Colab) use is also supported for testing::

    python colab_setup.py --project-dir ./some_copy --drive-root /tmp/fake_drive/lab2
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

OUTPUT_DIRS: Tuple[str, ...] = ("data", "checkpoints", "results", "runs")
CODE_FILES: Tuple[str, ...] = ("README.md", "requirements.txt", ".gitignore")
DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/lab2"
DEFAULT_PROJECT_DIR = "/content/lab2"
DRIVE_MOUNT_POINT = Path("/content/drive/MyDrive")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def in_colab() -> bool:
    """True when running inside Google Colab."""
    try:
        importlib.import_module("google.colab")

        return True
    except ImportError:
        return False


def drive_mounted() -> bool:
    return DRIVE_MOUNT_POINT.is_dir()


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def dir_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def move_contents(src: Path, dst: Path) -> int:
    """Move everything from ``src`` into ``dst`` (copy + delete across filesystems)."""
    moved = 0
    for item in list(src.iterdir()):
        target = dst / item.name
        if target.exists():
            print(f"    skip {item.name} (already on Drive)")
            continue
        shutil.move(str(item), str(target))
        moved += 1
    return moved


def link_dir(project_dir: Path, name: str, drive_dir: Path, dry_run: bool = False) -> str:
    """Make ``project_dir/name`` point at ``drive_dir/name`` and return what happened."""
    local = project_dir / name
    target = drive_dir / name

    if local.is_symlink():
        if local.resolve() == target.resolve():
            return f"{name}/: already linked -> {target}"
        local.unlink()
        if not dry_run:
            local.symlink_to(target, target_is_directory=True)
        return f"{name}/: relinked -> {target}"

    if local.is_dir():
        contents = [p for p in local.iterdir()]
        if contents:
            size = dir_size(local)
            print(f"  moving existing {name}/ ({len(contents)} items, {human_size(size)}) to Drive ...")
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
                move_contents(local, target)
        if not dry_run:
            shutil.rmtree(local, ignore_errors=True)
            local.symlink_to(target, target_is_directory=True)
        return f"{name}/: local folder moved to Drive and symlinked -> {target}"

    if local.exists():  # a plain file where a folder is expected
        raise RuntimeError(f"'{local}' exists and is not a directory - refusing to touch it.")

    if not dry_run:
        local.symlink_to(target, target_is_directory=True)
    return f"{name}/: symlinked -> {target}"


def copy_code(project_dir: Path, code_dir: Path, include_git: bool = False) -> List[str]:
    """Copy the source files into ``code_dir`` and return the list of copied names."""
    copied: List[str] = []
    code_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("*.py",):
        for path in sorted(project_dir.glob(pattern)):
            if path.is_file():
                shutil.copy2(path, code_dir / path.name)
                copied.append(path.name)
    for name in CODE_FILES:
        path = project_dir / name
        if path.is_file():
            shutil.copy2(path, code_dir / name)
            copied.append(name)

    if include_git:
        git_dir = project_dir / ".git"
        if git_dir.is_dir():
            shutil.rmtree(code_dir / ".git", ignore_errors=True)
            shutil.copytree(git_dir, code_dir / ".git")
            copied.append(".git/")
    return copied


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save the Lab 2 project (code + data + results) to Google Drive on Colab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--project-dir", type=str, default=None,
                        help=f"Working copy of the project (default: {DEFAULT_PROJECT_DIR} if it exists, else cwd).")
    parser.add_argument("--drive-root", type=str, default=DEFAULT_DRIVE_ROOT,
                        help="Folder on Drive that stores data/checkpoints/results/runs/code.")
    parser.add_argument("--no-data", action="store_true",
                        help="Keep the dataset on the local (ephemeral) disk instead of Drive.")
    parser.add_argument("--no-code", action="store_true", help="Do not snapshot the source code to Drive.")
    parser.add_argument("--copy-git", action="store_true",
                        help="Also copy the .git folder (lets you commit/push from Drive later).")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would happen.")
    return parser


def resolve_project_dir(value: Optional[str]) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    if Path(DEFAULT_PROJECT_DIR).is_dir():
        return Path(DEFAULT_PROJECT_DIR)
    return Path.cwd()


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_dir = resolve_project_dir(args.project_dir)
    drive_root = Path(args.drive_root).expanduser()

    print("=" * 78)
    print("Lab 2 - save project to Google Drive")
    print("=" * 78)
    print(f"project dir : {project_dir}")
    print(f"drive root  : {drive_root}")
    print(f"colab       : {in_colab()} | drive mounted: {drive_mounted()}")
    if args.dry_run:
        print("mode        : DRY RUN (nothing is changed)")
    print("-" * 78)

    if not project_dir.is_dir():
        print(f"[colab_setup] ERROR: project dir '{project_dir}' does not exist.", file=sys.stderr)
        print("Clone or copy the project first, e.g.:\n"
              "    !git clone https://github.com/thanh1912-ut/lab2.git /content/lab2", file=sys.stderr)
        return 1

    # --- Drive must be mounted by the user in a notebook cell -------------------------
    if in_colab() and not drive_mounted() and str(drive_root).startswith("/content/drive"):
        print(
            "[colab_setup] Google Drive is not mounted yet.\n"
            "Run this in a notebook cell (mounting needs the notebook's auth prompt, so it\n"
            "cannot be done from inside this script):\n\n"
            "    from google.colab import drive\n"
            "    drive.mount('/content/drive')\n\n"
            "then re-run:  !python colab_setup.py",
            file=sys.stderr,
        )
        return 1

    # --- create the Drive layout -------------------------------------------------------
    subdirs = ["code"] if not args.no_code else []
    subdirs += [d for d in OUTPUT_DIRS if not (d == "data" and args.no_data)]
    print("1) creating Drive folders")
    for name in subdirs:
        target = drive_root / name
        if not args.dry_run:
            target.mkdir(parents=True, exist_ok=True)
        print(f"   {'(dry) ' if args.dry_run else ''}{target}")

    # --- symlink output folders --------------------------------------------------------
    print("2) linking output folders into the project")
    for name in OUTPUT_DIRS:
        if name == "data" and args.no_data:
            print("   data/: kept local (--no-data)")
            continue
        print(f"   {link_dir(project_dir, name, drive_root, dry_run=args.dry_run)}")

    # --- snapshot the code -------------------------------------------------------------
    if not args.no_code:
        print("3) snapshotting the source code")
        code_dir = drive_root / "code"
        if args.dry_run:
            print(f"   (dry) copy *.py, README.md, requirements.txt, .gitignore -> {code_dir}")
        else:
            copied = copy_code(project_dir, code_dir, include_git=args.copy_git)
            print(f"   {len(copied)} items -> {code_dir}: {', '.join(copied)}")
    else:
        print("3) source snapshot skipped (--no-code)")

    # --- summary -----------------------------------------------------------------------
    print("-" * 78)
    print("Done. Everything now persists on Drive:")
    print(f"  data        : {drive_root / 'data'}          (CIFAR-10, downloaded once)")
    print(f"  checkpoints : {drive_root / 'checkpoints'}     (<model>_best.pt)")
    print(f"  results     : {drive_root / 'results'}         (<model>_history.csv, <model>_test.json, model_comparison.csv)")
    print(f"  runs        : {drive_root / 'runs'}            (tensorboard --logdir runs)")
    if not args.no_code:
        print(f"  code        : {drive_root / 'code'}            (source snapshot)")
    print("\nNext steps:")
    print("  !python train.py --model resnet18 --epochs 20 --batch-size 32 --lr 0.005")
    print("  !python benchmark.py --epochs 20 --batch-size 32 --lr 0.005")
    print("  %load_ext tensorboard")
    print("  %tensorboard --logdir runs")
    print("\nNext session: mount Drive, restore/re-clone the code, re-run this script - the")
    print("dataset and all previous results are already there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
