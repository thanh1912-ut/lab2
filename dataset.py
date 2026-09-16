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
        download: download CIFAR-10 when it is not present yet.

    Returns:
        ``train_dataset`` (augmented, 45,000), ``val_dataset`` (deterministic, 5,000),
        ``test_dataset`` (deterministic, 10,000).
    """
    if not 0 < val_split < 50_000:
        raise ValueError(f"val_split must be in (0, 50000), got {val_split}.")

    ensure_dir(data_dir)
    train_transform, eval_transform = build_transforms(image_size=image_size, resize_size=resize_size)

    # Two views of the same training images: the augmented one for training and the
    # deterministic one for validation (so validation is comparable to the test set).
    train_base = CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    val_base = CIFAR10(root=data_dir, train=True, download=download, transform=eval_transform)
    test_dataset = CIFAR10(root=data_dir, train=False, download=download, transform=eval_transform)

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
