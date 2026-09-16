#!/usr/bin/env python
"""Inspect the architecture of one pretrained backbone (Lab 2, requirement 18).

Prints:
  * where the model comes from and which ImageNet-1K weights are used
  * the full model architecture
  * total / trainable parameters
  * the classifier (final layer)
  * the output shape for a dummy input ``torch.randn(1, 3, 224, 224)`` (expected ``[1, 10]``)

Examples
--------
::

    python inspect_model.py --model resnet18
    python inspect_model.py --model mobilenetv4_small --num-classes 10
    python inspect_model.py --model vgg16 --no-pretrained     # offline, architecture only
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import torch

from config import IMAGE_SIZE
from models import (
    MODEL_SPECS,
    SUPPORTED_MODELS,
    check_timm_available,
    create_model,
    get_classifier,
    normalize_model_name,
)
from utils import count_parameters, format_parameters, get_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a pretrained backbone architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default="resnet18", choices=list(SUPPORTED_MODELS),
                        help="Backbone to inspect.")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes.")
    parser.add_argument("--input-size", type=int, default=IMAGE_SIZE, help="Square dummy input size.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device used for the dummy forward pass.")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Build the architecture without downloading ImageNet-1K weights.")
    parser.add_argument("--no-arch", action="store_true", help="Do not print the full architecture.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model_name = normalize_model_name(args.model)
    spec = MODEL_SPECS[model_name]

    print("=" * 78)
    print(f"Model        : {spec.display_name} ({model_name})")
    print(f"Source       : {spec.source}")
    print(f"Weights      : {spec.weights_id}")
    print(f"Note         : {spec.docs}")
    if spec.source == "timm":
        timm = check_timm_available()  # raises an actionable error when timm is missing

        print(f"timm version : {timm.__version__} (torchvision has no MobileNetV4 in this environment)")
    else:
        import torchvision  # noqa: PLC0415

        print(f"torchvision  : {torchvision.__version__}")
    print("=" * 78, flush=True)

    device = get_device(args.device)
    model = create_model(model_name, num_classes=args.num_classes, pretrained=not args.no_pretrained)
    model.to(device)
    model.eval()

    total, trainable = count_parameters(model)
    head = get_classifier(model)

    if not args.no_arch:
        print("\n--- Full architecture ---")
        print(model)

    print("\n--- Summary ---")
    print(f"Classifier/final layer ({spec.head_attribute}): {head}")
    print(f"Total parameters     : {format_parameters(total)}")
    print(f"Trainable parameters : {format_parameters(trainable)}")
    print(f"Model size           : {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB (fp32 params)")

    # --- dummy forward pass -----------------------------------------------------------
    dummy = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)
    print("\n--- Dummy forward ---")
    print(f"Input tensor : torch.randn({args.batch_size}, 3, {args.input_size}, {args.input_size})")
    with torch.inference_mode():
        output = model(dummy)

    print(f"Output shape : {list(output.shape)}")
    print(f"Output dtype : {output.dtype}")
    expected = (args.batch_size, args.num_classes)
    ok = tuple(output.shape) == expected
    print(f"Expected     : {list(expected)} -> {'OK' if ok else 'MISMATCH'}")

    if not ok:
        print(
            f"[inspect_model] ERROR: output shape {list(output.shape)} != {list(expected)}.",
            file=sys.stderr,
        )
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
