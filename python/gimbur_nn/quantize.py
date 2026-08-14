"""Create inference-only precision variants of trained checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def export_fp16_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("architecture") != "catan_policy_value_v1"
        or checkpoint.get("checkpoint_version") != 5
        or "model_state_dict" not in checkpoint
    ):
        raise ValueError("incompatible checkpoint; expected catan_policy_value_v1 version 5")
    converted = dict(checkpoint)
    converted["model_state_dict"] = {
        name: tensor.half() if torch.is_floating_point(tensor) else tensor
        for name, tensor in checkpoint["model_state_dict"].items()
    }
    converted["inference_precision"] = "fp16"
    converted["source_checkpoint"] = source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(converted, temporary)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an FP16 inference checkpoint.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    export_fp16_checkpoint(args.source, args.out)
    print(f"FP16 checkpoint exported to {args.out}")


if __name__ == "__main__":
    main()
