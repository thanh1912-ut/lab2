"""Central configuration for Lab 2: fine-tuning pretrained CNN backbones on CIFAR-10.

Everything that the benchmark must keep identical across backbones lives here
(hyperparameters, transforms constants, paths, model registry names) so that the
comparison between MobileNetV4-Small / VGG16 / ResNet18 / DenseNet121 is fair.

No absolute paths are used anywhere in the project: every path is relative and can
be overridden from the command line, which keeps the code portable to Google Colab.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------

CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

NUM_CLASSES: int = len(CIFAR10_CLASSES)  # 10
CIFAR10_TRAIN_SIZE: int = 50_000
CIFAR10_TEST_SIZE: int = 10_000

# --------------------------------------------------------------------------------------
# Image preprocessing (identical for all four backbones -> fair benchmark)
# --------------------------------------------------------------------------------------

# ImageNet statistics: all four backbones were pretrained on ImageNet-1K.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

IMAGE_SIZE: int = 224          # network input resolution
RESIZE_SIZE: int = 256         # val/test: Resize(256) then CenterCrop(224) -> 0.875 crop pct
RANDOM_RESIZED_CROP_SCALE: Tuple[float, float] = (0.6, 1.0)  # mild, backbone-agnostic

# --------------------------------------------------------------------------------------
# Optimisation defaults (MANDATORY values for the main benchmark)
# --------------------------------------------------------------------------------------

DEFAULT_EPOCHS: int = 20
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LR: float = 5e-3          # 0.005, identical for every backbone
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_OPTIMIZER: str = "adamw"
DEFAULT_SCHEDULER: str = "none"   # keep the recipe simple for the lab report
DEFAULT_SEED: int = 42
DEFAULT_NUM_WORKERS: int = 2
DEFAULT_VAL_SPLIT: int = 5_000    # 45,000 train / 5,000 validation / 10,000 test

SUPPORTED_OPTIMIZERS: Tuple[str, ...] = ("adamw", "sgd", "adam")
SUPPORTED_SCHEDULERS: Tuple[str, ...] = ("none", "cosine", "step")

# --------------------------------------------------------------------------------------
# Default output directories (relative -> Colab friendly)
# --------------------------------------------------------------------------------------

DEFAULT_DATA_DIR: str = "./data"
DEFAULT_CHECKPOINT_DIR: str = "./checkpoints"
DEFAULT_RESULTS_DIR: str = "./results"
DEFAULT_RUNS_DIR: str = "./runs"

# --------------------------------------------------------------------------------------
# Default model order used by benchmark.py
# --------------------------------------------------------------------------------------

BENCHMARK_MODELS: Tuple[str, ...] = (
    "mobilenetv4_small",
    "vgg16",
    "resnet18",
    "densenet121",
)


@dataclass
class TrainConfig:
    """All knobs of one training run.

    The fields default to the mandatory benchmark recipe, so
    ``TrainConfig(model_name="resnet18")`` is already a valid main-benchmark run.
    """

    model_name: str = "resnet18"
    data_dir: str = DEFAULT_DATA_DIR
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR
    results_dir: str = DEFAULT_RESULTS_DIR
    runs_dir: str = DEFAULT_RUNS_DIR

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    optimizer: str = DEFAULT_OPTIMIZER
    scheduler: str = DEFAULT_SCHEDULER
    label_smoothing: float = 0.0

    num_workers: int = DEFAULT_NUM_WORKERS
    val_split: int = DEFAULT_VAL_SPLIT
    seed: int = DEFAULT_SEED

    pretrained: bool = True          # ImageNet-1K weights, never train from scratch
    download: bool = True             # auto-download CIFAR-10 when it is missing
    freeze_backbone: bool = False    # full fine-tuning by default
    amp: bool = True                 # auto-disabled on CPU/MPS
    device: str = "auto"             # auto | cuda | mps | cpu

    log_interval: int = 0            # print a batch-level line every N batches (0 = off)

    # Debug / smoke-test helpers (None => use the full split)
    limit_train_batches: int | None = None
    limit_eval_batches: int | None = None

    tensorboard: bool = True
    run_name: str = ""               # defaults to model_name

    def to_dict(self) -> Dict[str, Any]:
        """JSON/checkpoint friendly view of the config."""
        data = asdict(self)
        data["effective_run_name"] = self.effective_run_name
        return data

    @property
    def effective_run_name(self) -> str:
        return self.run_name or self.model_name
