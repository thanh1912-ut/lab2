"""CIFAR-10 data pipeline: transforms, reproducible train/val/test split, DataLoaders.

Protocol used by the lab (identical for all four backbones):

* CIFAR-10 train split (50,000 images) -> 45,000 train + 5,000 validation
* CIFAR-10 official test split (10,000 images) -> test only, never used for tuning
* Images are resized from 32x32 to 224x224 to match the ImageNet pretrained backbones
* Normalisation uses the ImageNet mean/std for every backbone

The split depends only on ``seed`` and the dataset length (it is produced with an
explicit ``torch.Generator``), so all four models see exactly the same split regardless
of how much randomness was consumed before.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from config import (
    CIFAR10_CLASSES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    RANDOM_RESIZED_CROP_SCALE,
    RESIZE_SIZE,
)
from utils import ensure_dir, seed_worker

__all__ = [
    "build_train_transform",
    "build_eval_transform",
    "build_transforms",
    "load_cifar10_datasets",
    "build_dataloaders",
    "describe_dataset",
    "ensure_cifar10",
    "cifar10_is_available",
    "CIFAR10_MIRROR_ENV_VAR",
]


# --------------------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------------------
def build_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Training transform: mild, backbone-agnostic augmentation + ImageNet normalisation.

    ``RandomResizedCrop`` + ``RandomHorizontalFlip`` only. No RandAugment / MixUp / CutMix
    is used on purpose: strong augmentation would change the ranking between backbones and
    make the benchmark harder to interpret.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=RANDOM_RESIZED_CROP_SCALE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = IMAGE_SIZE, resize_size: int = RESIZE_SIZE) -> transforms.Compose:
    """Validation/test transform: ``Resize(256)`` + ``CenterCrop(224)`` + normalisation."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_transforms(
    image_size: int = IMAGE_SIZE,
    resize_size: int = RESIZE_SIZE,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Return ``(train_transform, eval_transform)``."""
    return (
        build_train_transform(image_size=image_size),
        build_eval_transform(image_size=image_size, resize_size=resize_size),
    )


# --------------------------------------------------------------------------------------
# Download handling
# --------------------------------------------------------------------------------------
# torchvision downloads CIFAR-10 from a single official URL (cs.toronto.edu) and verifies
# the archive md5 (``CIFAR10.tgz_md5``) before extracting it, so a truncated or tampered
# download fails loudly instead of silently corrupting a run. In some networks that host is
# slow, therefore:
#   * a custom mirror can be supplied through the LAB2_CIFAR10_MIRROR environment variable
#     (same file name + same md5 is enforced), and
#   * failures raise an actionable error instead of a bare traceback.
CIFAR10_ARCHIVE: str = "cifar-10-python.tar.gz"
CIFAR10_FOLDER: str = "cifar-10-batches-py"
CIFAR10_MIRROR_ENV_VAR: str = "LAB2_CIFAR10_MIRROR"
CIFAR10_OFFICIAL_URL: str = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def cifar10_is_available(data_dir: str = "./data") -> bool:
    """True when the CIFAR-10 python batches are already extracted in ``data_dir``."""
    folder = Path(data_dir) / CIFAR10_FOLDER
    return (folder / "data_batch_1").is_file() and (folder / "test_batch").is_file()


def ensure_cifar10(data_dir: str = "./data", download: bool = True) -> None:
    """Make sure CIFAR-10 is extracted in ``data_dir``, downloading it if needed.

    Mirrors ``torchvision.datasets.CIFAR10(download=...)`` but with an explicit, actionable
    failure message and optional mirror support (``LAB2_CIFAR10_MIRROR``). Whichever URL is
    used, the archive must match torchvision's published md5 (``CIFAR10.tgz_md5``) before it
    is extracted, so a mirror can never silently corrupt the dataset.
    """
    data_dir = str(data_dir)
    ensure_dir(data_dir)

    if cifar10_is_available(data_dir):
        return

    if not download:
        raise FileNotFoundError(
            f"CIFAR-10 is not present in '{data_dir}' and downloading is disabled (--no-download).\n"
            f"Either drop the extracted '{CIFAR10_FOLDER}/' folder (or the "
            f"'{CIFAR10_ARCHIVE}' archive) into '{data_dir}', or run without --no-download."
        )

    from torchvision.datasets.utils import download_and_extract_archive  # local import

    mirror = os.environ.get(CIFAR10_MIRROR_ENV_VAR, "").strip()
    # When a mirror is configured explicitly it is the only candidate: falling back to the
    # official host would silently ignore the user's choice and can hang a session for a long
    # time on a slow link. Unset the variable to use the official URL again.
    candidates = [mirror] if mirror else [CIFAR10_OFFICIAL_URL]

    errors: List[str] = []
    for url in candidates:
        try:
            download_and_extract_archive(
                url,
                download_root=data_dir,
                filename=CIFAR10_ARCHIVE,
                md5=CIFAR10.tgz_md5,  # enforced verification, whatever the source
            )
        except Exception as exc:  # noqa: BLE001 - collected and reported below
            errors.append(f"  - {url} -> {type(exc).__name__}: {exc}")
            continue

        if cifar10_is_available(data_dir):
            if url != CIFAR10_OFFICIAL_URL:
                print(f"[dataset] CIFAR-10 downloaded from mirror: {url}")
            return
        errors.append(f"  - {url} -> archive extracted but '{CIFAR10_FOLDER}' is incomplete")

    raise RuntimeError(
        "Failed to download CIFAR-10 into "
        f"'{data_dir}'. Attempts:\n" + "\n".join(errors) + "\n"
        "Options:\n"
        "  1. retry - transient network errors are common (the official host can be slow)\n"
        "  2. download it once into a persistent folder and reuse it, e.g. on Colab:\n"
        "       --data-dir /content/drive/MyDrive/lab2/data\n"
        "  3. fetch the archive manually, then let torchvision verify and extract it:\n"
        f"       mkdir -p {data_dir} && wget -O {data_dir}/{CIFAR10_ARCHIVE} {CIFAR10_OFFICIAL_URL}\n"
        f"     (md5 must be {CIFAR10.tgz_md5})\n"
        "  4. use your own mirror (any host serving the same file name):\n"
        f"       export {CIFAR10_MIRROR_ENV_VAR}=https://your-mirror/{CIFAR10_ARCHIVE}\n"
        f"       (currently {'set to ' + repr(mirror) if mirror else 'unset -> official URL'}; "
        f"unset it with `unset {CIFAR10_MIRROR_ENV_VAR}` to use the official URL)\n"
        f"  5. delete a possibly corrupted archive and retry: rm -f "
        f"'{Path(data_dir) / CIFAR10_ARCHIVE}'"
    )


# --------------------------------------------------------------------------------------
# Datasets / split
# --------------------------------------------------------------------------------------
def load_cifar10_datasets(
    data_dir: str = "./data",
    val_split: int = 5_000,
    seed: int = 42,
    image_size: int = IMAGE_SIZE,
    resize_size: int = RESIZE_SIZE,
    download: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Build the ``(train, val, test)`` datasets.

    Args:
        data_dir: where CIFAR-10 lives / is downloaded to (``./data`` by default).
        val_split: number of training images held out for validation (5,000).
        seed: seed of the deterministic split.
        image_size: network input resolution (224).
        resize_size: resize size used by the evaluation transform (256).
        download: download CIFAR-10 when it is not present yet (see :func:`ensure_cifar10`).

    Returns:
        ``train_dataset`` (augmented, 45,000), ``val_dataset`` (deterministic, 5,000),
        ``test_dataset`` (deterministic, 10,000).
    """
    if not 0 < val_split < 50_000:
        raise ValueError(f"val_split must be in (0, 50000), got {val_split}.")

    ensure_dir(data_dir)
    ensure_cifar10(data_dir, download=download)
    train_transform, eval_transform = build_transforms(image_size=image_size, resize_size=resize_size)

    # Two views of the same training images: the augmented one for training and the
    # deterministic one for validation (so validation is comparable to the test set).
    train_base = CIFAR10(root=data_dir, train=True, download=False, transform=train_transform)
    val_base = CIFAR10(root=data_dir, train=True, download=False, transform=eval_transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=False, transform=eval_transform)

    generator = torch.Generator().manual_seed(seed)  # reproducible split, independent of global RNG
    indices = torch.randperm(len(train_base), generator=generator).tolist()
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]

    train_dataset: Dataset = Subset(train_base, train_indices)
    val_dataset: Dataset = Subset(val_base, val_indices)
    return train_dataset, val_dataset, test_dataset


# --------------------------------------------------------------------------------------
# DataLoaders
# --------------------------------------------------------------------------------------
def build_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 32,
    val_split: int = 5_000,
    seed: int = 42,
    num_workers: int = 2,
    image_size: int = IMAGE_SIZE,
    resize_size: int = RESIZE_SIZE,
    pin_memory: Optional[bool] = None,
    download: bool = True,
) -> Tuple[Dict[str, DataLoader], Tuple[Dataset, Dataset, Dataset]]:
    """Create the ``train`` / ``val`` / ``test`` DataLoaders plus the underlying datasets.

    Shuffling is on for training only. ``drop_last=False`` everywhere so validation and
    test accuracy are computed on the complete splits. ``worker_init_fn`` + a seeded
    generator keep the training augmentation reproducible across runs.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}.")

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset, val_dataset, test_dataset = load_cifar10_datasets(
        data_dir=data_dir,
        val_split=val_split,
        seed=seed,
        image_size=image_size,
        resize_size=resize_size,
        download=download,
    )

    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(train_dataset, shuffle=True, generator=generator, drop_last=False, **common),
        "val": DataLoader(val_dataset, shuffle=False, drop_last=False, **common),
        "test": DataLoader(test_dataset, shuffle=False, drop_last=False, **common),
    }
    return loaders, (train_dataset, val_dataset, test_dataset)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def describe_dataset(train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset) -> str:
    """Human readable summary of the split (printed at the start of every run)."""
    lines: List[str] = [
        "Dataset: CIFAR-10",
        f"  classes      : {len(CIFAR10_CLASSES)} -> {', '.join(CIFAR10_CLASSES)}",
        f"  train        : {len(train_dataset):,} images (augmented)",
        f"  validation   : {len(val_dataset):,} images",
        f"  test         : {len(test_dataset):,} images (never used for tuning)",
        f"  input size   : {IMAGE_SIZE}x{IMAGE_SIZE} (ImageNet normalisation)",
    ]
    return "\n".join(lines)
