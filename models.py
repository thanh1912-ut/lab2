"""Model factory: ImageNet-1K pretrained backbones with a 10-class CIFAR-10 head.

Supported ``model_name`` values
-------------------------------
===================  ==============  ==================================================================
model_name           source          ImageNet-1K pretrained weights
===================  ==============  ==================================================================
``mobilenetv4_small``  timm          ``mobilenetv4_conv_small.e2400_r224_in1k`` (MobileNetV4 Conv Small)
``vgg16``              torchvision   ``torchvision.models.VGG16_Weights.IMAGENET1K_V1``
``resnet18``           torchvision   ``torchvision.models.ResNet18_Weights.IMAGENET1K_V1``
``densenet121``        torchvision   ``torchvision.models.DenseNet121_Weights.IMAGENET1K_V1``
===================  ==============  ==================================================================

Why timm for MobileNetV4?
-------------------------
MobileNetV4 was published in 2024 (ECCV). The torchvision version installed for this
lab does **not** ship a MobileNetV4 implementation, therefore the official
``timm`` (PyTorch Image Models, HuggingFace) implementation of the *MobileNetV4 Conv
Small* variant is used with its ImageNet-1K pretrained weights. MobileNetV3 is **never**
used as a stand-in. The exact resolution of that decision happens at runtime in
:func:`_resolve_timm_model_name` / :func:`create_model`, which reports a clear error and
lists the available MobileNetV4 candidates when the requested tag is missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch.nn as nn

from config import NUM_CLASSES
from utils import count_parameters

__all__ = [
    "ModelSpec",
    "MODEL_SPECS",
    "SUPPORTED_MODELS",
    "create_model",
    "get_classifier",
    "replace_classifier",
    "freeze_backbone",
    "unfreeze_model",
    "count_parameters",
    "get_model_summary",
    "list_timm_mobilenetv4",
    "check_timm_available",
]


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    """Static description of one supported backbone."""

    name: str                      # CLI name, e.g. "resnet18"
    display_name: str              # pretty name, e.g. "ResNet18"
    source: str                    # "torchvision" | "timm"
    arch_name: str                 # constructor name in the source library, e.g. "mobilenetv4_conv_small"
    weights_id: str                # exact ImageNet-1K pretrained weight identifier
    head_attribute: str            # name of the classification head module
    docs: str = ""                 # extra note stored in checkpoints/README


MODEL_SPECS: Dict[str, ModelSpec] = {
    "mobilenetv4_small": ModelSpec(
        name="mobilenetv4_small",
        display_name="MobileNetV4-Small",
        source="timm",
        arch_name="mobilenetv4_conv_small",
        weights_id="mobilenetv4_conv_small.e2400_r224_in1k",
        head_attribute="classifier",
        docs=(
            "MobileNetV4 Conv Small from timm (torchvision has no MobileNetV4 in the "
            "installed version). ImageNet-1K pretrained."
        ),
    ),
    "vgg16": ModelSpec(
        name="vgg16",
        display_name="VGG16",
        source="torchvision",
        arch_name="vgg16",
        weights_id="VGG16_Weights.IMAGENET1K_V1",
        head_attribute="classifier",
        docs="torchvision VGG16 with ImageNet-1K V1 weights.",
    ),
    "resnet18": ModelSpec(
        name="resnet18",
        display_name="ResNet18",
        source="torchvision",
        arch_name="resnet18",
        weights_id="ResNet18_Weights.IMAGENET1K_V1",
        head_attribute="fc",
        docs="torchvision ResNet18 with ImageNet-1K V1 weights.",
    ),
    "densenet121": ModelSpec(
        name="densenet121",
        display_name="DenseNet121",
        source="torchvision",
        arch_name="densenet121",
        weights_id="DenseNet121_Weights.IMAGENET1K_V1",
        head_attribute="classifier",
        docs="torchvision DenseNet121 with ImageNet-1K V1 weights.",
    ),
}

SUPPORTED_MODELS: Tuple[str, ...] = tuple(MODEL_SPECS.keys())

# Aliases so that a typo-friendly call still resolves to the right architecture.
_MODEL_ALIASES: Dict[str, str] = {
    "mobilenetv4": "mobilenetv4_small",
    "mobilenet_v4_small": "mobilenetv4_small",
    "mobilenetv4_conv_small": "mobilenetv4_small",
    "mobilenet_v4": "mobilenetv4_small",
    "resnet_18": "resnet18",
    "resnet-18": "resnet18",
    "vgg_16": "vgg16",
    "vgg-16": "vgg16",
    "densenet_121": "densenet121",
    "densenet-121": "densenet121",
}


def normalize_model_name(model_name: str) -> str:
    """Map a user supplied name (case/alias tolerant) to a registry key."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    key = model_name.strip().lower().replace(" ", "_")
    key = _MODEL_ALIASES.get(key, key)

    if key not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: {', '.join(SUPPORTED_MODELS)}."
        )
    return key


# --------------------------------------------------------------------------------------
# timm availability helpers
# --------------------------------------------------------------------------------------
def check_timm_available() -> Any:
    """Import timm lazily with an actionable error message (only needed for MobileNetV4)."""
    try:
        import timm  # noqa: PLC0415 - intentional lazy import
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "MobileNetV4 requires the 'timm' package (it is not in torchvision for this "
            "version). Install it with:\n"
            "    pip install timm\n"
            "or install all dependencies with:\n"
            "    pip install -r requirements.txt"
        ) from exc
    return timm


def list_timm_mobilenetv4(pretrained_only: bool = True) -> List[str]:
    """Return the MobileNetV4 model names known to the installed timm version."""
    timm = check_timm_available()
    try:
        names = timm.list_models("*mobilenetv4*", pretrained=pretrained_only)
    except Exception:  # noqa: BLE001 - very old timm signatures
        names = [n for n in timm.list_models("*mobilenetv4*")]
    return sorted(names)


def _resolve_timm_model_name(requested: str) -> str:
    """Validate that ``requested`` exists in the installed timm and return it.

    If the exact tag is missing, the function tries the plain architecture name
    (which makes timm pick the default pretrained tag) and, failing that, raises an
    error that lists every MobileNetV4 candidate found in the installed timm.
    There is deliberately **no** fallback to another architecture (e.g. MobileNetV3).
    """
    timm = check_timm_available()
    available = list_timm_mobilenetv4(pretrained_only=True)

    # 1) exact pretrained tag requested and available -> use it as-is.
    if requested in available:
        return requested

    try:
        known = set(timm.list_models("*mobilenetv4*"))
    except Exception:  # noqa: BLE001 - older timm signatures
        known = set(available)

    # 2) a bare architecture name (no ".tag" suffix) that timm knows -> let timm pick its
    #    own default pretrained tag for that architecture.
    if "." not in requested and requested in known:
        return requested

    # 3) anything else (unknown architecture, or an explicit tag that this timm version
    #    does not ship) is an error. We never silently resolve to a different checkpoint.
    raise RuntimeError(
        f"timm (version {getattr(timm, '__version__', 'unknown')}) does not provide the "
        f"MobileNetV4 model '{requested}'"
        + (" (that exact pretrained tag is unknown)." if "." in requested else ".")
        + "\nAvailable MobileNetV4 models with ImageNet-1K pretrained weights:\n  - "
        + "\n  - ".join(available or ["<none found>"])
        + "\nInstall/upgrade timm with:  pip install -U timm\n"
        "No substitute architecture (e.g. MobileNetV3) is used automatically."
    )


# --------------------------------------------------------------------------------------
# Head handling
# --------------------------------------------------------------------------------------
def get_classifier(model: nn.Module) -> nn.Module:
    """Return the classification head module of any supported backbone."""
    for attr in ("fc", "classifier"):
        head = getattr(model, attr, None)
        if isinstance(head, nn.Module):
            return head
    raise AttributeError(
        f"Cannot locate a classification head on {type(model).__name__}; expected an "
        "attribute named 'fc' (ResNet) or 'classifier' (VGG/DenseNet/timm)."
    )


def replace_classifier(model: nn.Module, num_classes: int = NUM_CLASSES, model_name: str = "") -> nn.Module:
    """Replace the final classification layer so the model outputs ``num_classes`` logits.

    Handles each architecture explicitly instead of hard-coding a single pattern:

    * **ResNet18**      -> ``model.fc = nn.Linear(512, num_classes)``
    * **VGG16**         -> ``model.classifier[6] = nn.Linear(4096, num_classes)``
                           (the ``classifier`` is a ``Sequential`` of Linear/ReLU/Dropout)
    * **DenseNet121**   -> ``model.classifier = nn.Linear(1024, num_classes)``
    * **MobileNetV4**   -> timm style: ``model.reset_classifier(num_classes)`` which rebuilds
                           the conv-pool head + final ``Linear`` (timm's own API, so the
                           architecture stays exactly as published).
    """
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")

    # -- timm models (MobileNetV4): use timm's own head-reset API ----------------------
    if hasattr(model, "reset_classifier") and hasattr(model, "forward_head"):
        model.reset_classifier(num_classes)
        return model

    # -- torchvision ResNet -----------------------------------------------------------
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # -- torchvision VGG: classifier is a Sequential whose last Linear must be swapped --
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Sequential):
        last_linear_index = None
        for index, module in enumerate(classifier):
            if isinstance(module, nn.Linear):
                last_linear_index = index
        if last_linear_index is None:
            raise AttributeError(f"{type(model).__name__}.classifier contains no nn.Linear layer.")
        in_features = classifier[last_linear_index].in_features
        classifier[last_linear_index] = nn.Linear(in_features, num_classes)
        return model

    # -- torchvision DenseNet (and anything else with a single Linear 'classifier') ----
    if isinstance(classifier, nn.Linear):
        model.classifier = nn.Linear(classifier.in_features, num_classes)
        return model

    raise AttributeError(
        f"Unsupported classification head for {model_name or type(model).__name__}: "
        f"found attribute(s) {[a for a in ('fc', 'classifier') if hasattr(model, a)]}."
    )


# --------------------------------------------------------------------------------------
# Freeze / unfreeze
# --------------------------------------------------------------------------------------
def _head_parameter_ids(model: nn.Module) -> set:
    head = get_classifier(model)
    return {id(p) for p in head.parameters()}


def freeze_backbone(model: nn.Module) -> nn.Module:
    """Freeze everything except the classification head (optional experiment).

    Not used by default: the main benchmark performs full fine-tuning.
    """
    head_ids = _head_parameter_ids(model)
    for param in model.parameters():
        param.requires_grad = id(param) in head_ids
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Make every parameter trainable (default behaviour)."""
    for param in model.parameters():
        param.requires_grad = True
    return model


# --------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------
def _build_torchvision_model(spec: ModelSpec, num_classes: int, pretrained: bool) -> nn.Module:
    import torchvision.models as tvm  # local import keeps `import models` cheap

    if not hasattr(tvm, spec.arch_name):
        raise RuntimeError(
            f"torchvision {getattr(tvm, '__version__', 'unknown')} does not provide "
            f"'{spec.arch_name}'. Upgrade with: pip install -U torchvision"
        )

    builder: Callable[..., nn.Module] = getattr(tvm, spec.arch_name)
    weights_enum_name = spec.weights_id.split(".")[0]          # e.g. "ResNet18_Weights"
    member_name = spec.weights_id.split(".")[-1]               # e.g. "IMAGENET1K_V1"

    weights_enum = getattr(tvm, weights_enum_name, None)
    if weights_enum is None:
        # Very old torchvision without the weights-API: explicit but still pretrained.
        model = builder(pretrained=pretrained)
    else:
        if not hasattr(weights_enum, member_name):
            raise RuntimeError(
                f"torchvision {tvm.__version__} has no weights entry "
                f"'{weights_enum_name}.{member_name}'. Available: {list(weights_enum.__members__)}"
            )
        weights = getattr(weights_enum, member_name) if pretrained else None
        model = builder(weights=weights)

    return replace_classifier(model, num_classes=num_classes, model_name=spec.name)


def _build_timm_model(spec: ModelSpec, num_classes: int, pretrained: bool) -> nn.Module:
    timm = check_timm_available()

    if not pretrained:
        # Only used for architecture inspection, offline smoke tests, and rebuilding a
        # network whose weights are then overwritten by a fine-tuned checkpoint - never to
        # train from scratch. Note: the timm *architecture* name differs from the CLI name
        # (mobilenetv4_small -> mobilenetv4_conv_small), hence spec.arch_name.
        try:
            model = timm.create_model(spec.arch_name, pretrained=False, num_classes=num_classes)
        except Exception as exc:  # noqa: BLE001 - wrap with actionable context
            raise RuntimeError(
                f"Failed to create timm model '{spec.arch_name}' without pretrained weights: {exc}\n"
                f"Available MobileNetV4 architectures: {list_timm_mobilenetv4(pretrained_only=False)[:10]}"
            ) from exc
        return replace_classifier(model, num_classes=num_classes, model_name=spec.name)

    resolved = _resolve_timm_model_name(spec.weights_id)
    try:
        model = timm.create_model(resolved, pretrained=True, num_classes=num_classes)
    except Exception as exc:  # noqa: BLE001 - wrap with actionable context
        raise RuntimeError(
            f"Failed to create timm model '{resolved}' with ImageNet-1K pretrained weights: {exc}\n"
            "Check your internet connection / HuggingFace cache, or run "
            "`python -c \"import timm; print(timm.list_models('*mobilenetv4*', pretrained=True))\"`."
        ) from exc

    if resolved != spec.weights_id:
        print(f"[models] Note: timm resolved '{spec.weights_id}' -> '{resolved}' (default pretrained tag).")
    return model


def create_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """Create one of the four pretrained backbones with a ``num_classes`` output head.

    Args:
        model_name: one of ``mobilenetv4_small``, ``vgg16``, ``resnet18``, ``densenet121``
            (aliases such as ``mobilenetv4`` are accepted, see :data:`_MODEL_ALIASES`).
        num_classes: number of output logits, 10 for CIFAR-10.
        pretrained: load ImageNet-1K weights. **Always True for training in this lab.**
            It is only set to False when rebuilding a network whose weights are about to be
            overwritten by a fine-tuned checkpoint (see ``evaluate.py``), or for offline
            architecture inspection.

    Returns:
        The model, on CPU, in train mode, with a fresh ``num_classes`` head.
    """
    key = normalize_model_name(model_name)
    spec = MODEL_SPECS[key]

    if spec.source == "torchvision":
        model = _build_torchvision_model(spec, num_classes=num_classes, pretrained=pretrained)
    elif spec.source == "timm":
        model = _build_timm_model(spec, num_classes=num_classes, pretrained=pretrained)
    else:  # pragma: no cover - registry is fixed
        raise ValueError(f"Unknown model source '{spec.source}' for '{key}'.")

    model.train()
    return model


# --------------------------------------------------------------------------------------
# Reporting helpers
# --------------------------------------------------------------------------------------
def get_model_summary(model: nn.Module, model_name: str = "") -> Dict[str, Any]:
    """Collect the metadata written into checkpoints and the comparison CSV."""
    key = normalize_model_name(model_name) if model_name else ""
    spec = MODEL_SPECS.get(key)
    total, trainable = count_parameters(model)
    head = get_classifier(model)

    return {
        "model": key or type(model).__name__,
        "display_name": spec.display_name if spec else type(model).__name__,
        "source": spec.source if spec else "unknown",
        "weights": spec.weights_id if spec else "unknown",
        "classifier": repr(head).replace("\n", " "),
        "head_attribute": spec.head_attribute if spec else "unknown",
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
    }
