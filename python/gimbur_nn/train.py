"""
Training loop for GimburTransformer.

Reads JSONL data exported by ``gimbur simulate --export``, expands
states via symmetry permutations and player rotation, and trains the
model to predict a win-probability bucket distribution via
cross-entropy loss.

Usage::

    # Train until val loss plateaus (default: patience=5)
    python -m gimbur_nn.train \
        --data exports/ \
        --game-config mini_2p \
        --model-config small \
        --out model.pt

    # Train for a fixed number of epochs (still stops early if val loss plateaus)
    python -m gimbur_nn.train \
        --data exports/ \
        --game-config mini_2p \
        --model-config small \
        --out model.pt \
        --epochs 20 \
        --patience 10

``--data`` may point to a single ``.jsonl`` file **or** a directory
containing one or more ``.jsonl`` files.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data_loader import SimulationDataset, load_games, split_games
from .game_config import CONFIGS_BY_NAME
from .transformer_model import (
    MODEL_CONFIGS_BY_NAME,
    GimburTransformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GimburTransformer on simulation data.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to a JSONL file or a directory of JSONL files.",
    )
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
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing checkpoint to resume training from.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Max training epochs (0 = unlimited, stop only via patience).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Stop after N epochs with no val loss improvement (requires --val-split > 0).",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of games to hold out for validation (0 to disable).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help="Fraction of games to hold out for testing (0 to disable).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print training loss every N batches (0 to disable).",
    )
    return parser.parse_args()


def _run_epoch(
    model: GimburTransformer,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    log_interval: int,
    epoch: int,
    phase: str,
) -> float:
    """Run one epoch (train or eval).

    When *optimizer* is ``None`` the model runs in eval mode with no
    gradient updates (validation).

    Returns the mean loss over the epoch.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch_idx, (token_ids, targets) in enumerate(loader):
            token_ids = token_ids.to(device)
            targets = targets.to(device)

            logits = model(token_ids)  # (batch, seq_len, n_buckets)
            last_logits = logits[:, -1, :]  # (batch, n_buckets)

            loss = F.cross_entropy(last_logits, targets)

            if is_train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = token_ids.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if is_train and log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg = total_loss / total_samples
                print(
                    f"  [{phase}] epoch {epoch} | batch {batch_idx + 1} | "
                    f"loss {loss.item():.4f} (running avg {avg:.4f})"
                )

    return total_loss / total_samples if total_samples > 0 else 0.0


def main() -> None:
    args = parse_args()

    game_cfg = CONFIGS_BY_NAME[args.game_config]
    model_cfg = MODEL_CONFIGS_BY_NAME[args.model_config]

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────
    model = GimburTransformer(game_cfg, model_cfg)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))
        print(f"Resumed from checkpoint {args.resume}")
    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"Model: {args.model_config} for {args.game_config} "
        f"({param_count:,} parameters) on {device}"
    )

    # ── Data ─────────────────────────────────────────────────────────
    print(f"Loading games from {args.data} ...")
    t0 = time.monotonic()
    all_games = load_games(args.data)
    elapsed = time.monotonic() - t0
    print(f"Loaded {len(all_games):,} games in {elapsed:.1f}s")

    if not all_games:
        print("No games found — nothing to train on.")
        return

    # ── Game-level train / val / test split ──────────────────────────
    train_games, val_games, test_games = split_games(
        all_games, val=args.val_split, test=args.test_split
    )

    train_dataset = SimulationDataset(train_games, game_cfg, n_buckets=model_cfg.n_buckets)
    print(f"Train: {len(train_games):,} games -> {len(train_dataset):,} samples")

    val_dataset: SimulationDataset | None = None
    if val_games:
        val_dataset = SimulationDataset(val_games, game_cfg, n_buckets=model_cfg.n_buckets)
        print(f"Val:   {len(val_games):,} games -> {len(val_dataset):,} samples")

    test_dataset: SimulationDataset | None = None
    if test_games:
        test_dataset = SimulationDataset(test_games, game_cfg, n_buckets=model_cfg.n_buckets)
        print(f"Test:  {len(test_games):,} games -> {len(test_dataset):,} samples")

    if len(train_dataset) == 0:
        print("No training samples — nothing to train on.")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    max_epochs = args.epochs if args.epochs > 0 else None
    epoch = 0

    while True:
        epoch += 1
        if max_epochs is not None and epoch > max_epochs:
            break

        t_start = time.monotonic()

        train_loss = _run_epoch(
            model, train_loader, device, optimizer, args.log_interval, epoch, "train"
        )

        label = f"{epoch}/{max_epochs}" if max_epochs else str(epoch)
        msg = f"Epoch {label} | train loss {train_loss:.4f}"

        if val_loader is not None:
            val_loss = _run_epoch(model, val_loader, device, None, 0, epoch, "val")
            msg += f" | val loss {val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), args.out)
                msg += " (best, saved)"
            else:
                epochs_without_improvement += 1
        else:
            torch.save(model.state_dict(), args.out)

        elapsed = time.monotonic() - t_start
        msg += f" | {elapsed:.1f}s"
        print(msg)

        if val_loader is not None and epochs_without_improvement >= args.patience:
            print(f"Early stopping: no val loss improvement for {args.patience} epochs")
            break

    print(f"Training complete ({epoch} epochs). Checkpoint at {args.out}")

    # ── Test evaluation ──────────────────────────────────────────────
    if test_dataset is not None and len(test_dataset) > 0:
        test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        # Reload best checkpoint for test evaluation.
        model.load_state_dict(torch.load(args.out, map_location=device, weights_only=True))
        test_loss = _run_epoch(model, test_loader, device, None, 0, 0, "test")
        print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
