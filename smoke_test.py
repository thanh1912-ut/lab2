#!/usr/bin/env python
"""Smoke test for Lab 2: build all four backbones and run a dummy forward pass.

Checks (requirement 29 of the lab):

1. the project modules import correctly
2. each model is created successfully
3. the ImageNet-1K pretrained weights API resolves for the installed torchvision/timm
4. the classification head outputs exactly ``num_classes`` logits
5. ``torch.randn(2, 3, 224, 224)`` -> ``output.shape == (2, 10)``

Examples
--------
::

    python smoke_test.py                       # downloads ImageNet weights if needed
    python smoke_test.py --no-pretrained       # offline: architecture + head shape only
    python smoke_test.py --models resnet18 vgg16
"""

from __future__ import annotations

import argparse
import gc
import sys
from typing import Dict, List, Optional

import torch

from config import NUM_CLASSES
from models import (
    MODEL_SPECS,
    SUPPORTED_MODELS,
    count_parameters,
    create_model,
    get_classifier,
    list_timm_mobilenetv4,
    normalize_model_name,
)
from models import check_timm_available
from utils import format_parameters, get_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Smoke test: create all backbones and forward a dummy batch.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", type=str, nargs="+", default=list(SUPPORTED_MODELS),
                        choices=list(SUPPORTED_MODELS), help="Models to test.")
    parser.add_argument("--batch-size", type=int, default=2, help="Dummy batch size.")
    parser.add_argument("--image-size", type=int, default=224, help="Dummy square image size.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="Expected output classes.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device for the forward pass (cpu is enough for a smoke test).")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip the ImageNet checkpoint download (checks architecture only).")
    return parser


def check_environment(pretrained: bool) -> None:
    """Print versions and verify that the pretrained-weights APIs resolve."""
    import torchvision


    print("=" * 78)
    print("Environment")
    print("=" * 78)
    print(f"python      : {sys.version.split()[0]}")
    print(f"torch       : {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print(f"torchvision : {torchvision.__version__}")
    print(f"timm        : {check_timm_available().__version__} (needed for MobileNetV4 only)")

    has_v4 = hasattr(torchvision.models, "mobilenet_v4")
    print(f"torchvision.models.mobilenet_v4 present: {has_v4}")
    if not has_v4:
        candidates = list_timm_mobilenetv4(pretrained_only=True)
        print(f"-> MobileNetV4 comes from timm. Available MobileNetV4 ImageNet-1K weights ({len(candidates)}):")
        for name in candidates:
            marker = "  <== used by this project" if name == MODEL_SPECS["mobilenetv4_small"].weights_id else ""
            print(f"     - {name}{marker}")
    if pretrained:
        print("Pretrained weights will be downloaded on first use "
              "(~528 MB for VGG16, ~45 MB ResNet18, ~31 MB DenseNet121, ~7 MB MobileNetV4).")
    print("=" * 78, flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    pretrained = not args.no_pretrained

    check_environment(pretrained)
    device = get_device(args.device)

    failures: List[str] = []
    results: List[Dict[str, object]] = []

    for model_name in args.models:
        key = normalize_model_name(model_name)
        spec = MODEL_SPECS[key]
        print(f"\n--- {spec.display_name} ({key}) ---")
        print(f"source   : {spec.source}")
        print(f"weights  : {spec.weights_id if pretrained else 'none (--no-pretrained)'}")
        try:
            model = create_model(key, num_classes=args.num_classes, pretrained=pretrained)
            model.to(device)
            model.eval()

            head = get_classifier(model)
            total, trainable = count_parameters(model)
            dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
            with torch.inference_mode():
                output = model(dummy)

            expected = (args.batch_size, args.num_classes)
            shape_ok = tuple(output.shape) == expected
            head_last = head[-1] if isinstance(head, torch.nn.Sequential) else head
            classes_ok = getattr(head_last, "out_features", None) == args.num_classes

            print(f"classifier: {head_last}")
            print(f"params    : total {format_parameters(total)} | trainable {format_parameters(trainable)}")
            print(f"forward   : {list(dummy.shape)} -> {list(output.shape)} (expected {list(expected)})")
            print(f"head classes == {args.num_classes}: {classes_ok}")
            print(f"RESULT    : {'PASS' if (shape_ok and classes_ok) else 'FAIL'}")

            if not (shape_ok and classes_ok):
                failures.append(key)
            results.append(
                {
                    "model": key,
                    "total_params": total,
                    "shape": list(output.shape),
                    "ok": shape_ok and classes_ok,
                }
            )
            del model, dummy, output
            gc.collect()
        except Exception as exc:  # noqa: BLE001 - the smoke test must report every failure
            print(f"RESULT    : FAIL -> {type(exc).__name__}: {exc}")
            failures.append(key)

    print("\n" + "=" * 78)
    print("Smoke test summary")
    print("=" * 78)
    for entry in results:
        print(f"  {'PASS' if entry['ok'] else 'FAIL'}  {entry['model']:<20} shape={entry['shape']} "
              f"params={format_parameters(int(entry['total_params']))}")
    for name in failures:
        if name not in [e["model"] for e in results]:
            print(f"  FAIL  {name}")

    if failures:
        print(f"\n{len(failures)} model(s) failed: {', '.join(failures)}")
        return 1
    print(f"\nAll {len(results)} models passed the smoke test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
