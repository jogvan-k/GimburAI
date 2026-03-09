"""
Training loop for GimburTransformer.

Currently implements a dummy training mode that initialises a model with
random weights and saves it to disk.  Real training (reading JSONL data
exported by ``gimbur simulate --export``) will be added later.

Usage::

    python -m gimbur_nn.train \
        --game-config mini_2p \
        --model-config small \
        --out model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .game_config import CONFIGS_BY_NAME
from .transformer_model import (
    GimburTransformer,
    MODEL_CONFIGS_BY_NAME,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GimburTransformer on simulation data.")
    parser.add_argument(
        "--game-config",
        type=str,
        required=True,
        choices=sorted(CONFIGS_BY_NAME),
        help="Game configuration preset.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        choices=sorted(MODEL_CONFIGS_BY_NAME),
        help="Model size preset.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("model.pt"),
        help="Output checkpoint path (default: model.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    game_cfg = CONFIGS_BY_NAME[args.game_config]
    model_cfg = MODEL_CONFIGS_BY_NAME[args.model_config]

    model = GimburTransformer(game_cfg, model_cfg)
    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"Initialised {args.model_config} model for {args.game_config} ({param_count:,} parameters)"
    )

    torch.save(model.state_dict(), args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
