"""
Training loop for CatanNet.

Reads JSONL training data exported by ``gimbur simulate --export``,
tokenizes each state, and trains the model to predict win probabilities
from MCTS search results.

Usage:
    python -m gimbur_nn.train --data games.jsonl --epochs 10
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GimburNet on simulation data.")
    parser.add_argument(
        "--data", type=Path, required=True, help="Path to JSONL training data."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--out", type=Path, default=Path("model.pt"), help="Output model path."
    )
    return parser.parse_args()


def main() -> None:
    _args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
