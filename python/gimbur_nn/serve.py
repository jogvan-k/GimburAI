"""
Inference server for GimburNet.

Loads a trained model checkpoint and exposes an HTTP endpoint that
accepts serialized board+state strings and returns win probability
predictions. The Kjarni MCTS engine can call this endpoint as a
learned leaf evaluator.

Usage:
    python -m gimbur_nn.serve --model model.pt --port 8000
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GimburNet for inference.")
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to trained model checkpoint."
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    return parser.parse_args()


def main() -> None:
    _args = parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
