"""
Training loop for GimburTransformer and GimburPlacementTransformer.

Reads data exported by ``gimbur simulate --export``, expands states via
symmetry permutations and player rotation, and trains the model to predict
a win-probability bucket distribution.  The loss function is
configurable — see ``--loss`` and :mod:`gimbur_nn.loss_config`.

``--data`` may point to a single ``.jsonl`` file, a single ``.json``
file, **or** a directory containing any mix of ``.jsonl`` and ``.json``
files.  The pipeline's default export format writes one ``.json`` file
per game into a directory.

Usage::

    # Train a state model until val loss plateaus (default: patience=5)
    python -m gimbur_nn.train \
        --data exports/ \
        --game-config mini_2p \
        --model-config small \
        --out model.pt

    # Train a placement model from a JSON config file
    python -m gimbur_nn.train \
        --config train_config.json

    # Train with per-epoch checkpointing for recovery
    python -m gimbur_nn.train \
        --data exports/ \
        --game-config mini_2p \
        --model-config small \
        --out model.pt \
        --checkpoint-dir checkpoints/
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data_loader import PlacementDataset, SimulationDataset, load_games, split_games
from .game_config import CONFIGS_BY_NAME
from .loss_config import LOSS_MODES, LossConfig, LossFn, build_loss_fn
from .transformer_model import (
    MODEL_CONFIGS_BY_NAME,
    GimburPlacementTransformer,
    GimburTransformer,
)

# ── Config file helpers ──────────────────────────────────────────────


def _strip_json_comments(text: str) -> str:
    """Remove single-line // comments from JSON text (outside strings)."""
    lines = []
    for line in text.splitlines():
        in_string = False
        escape = False
        for i, ch in enumerate(line):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
            elif ch == "/" and not in_string and i + 1 < len(line) and line[i + 1] == "/":
                line = line[:i].rstrip()
                break
        lines.append(line)
    return "\n".join(lines)


def _to_snake(camel: str) -> str:
    """Convert camelCase to snake_case."""
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", camel).lower()


def _load_config(path: Path) -> dict[str, object]:
    """Read a JSON config file (with // comment support) and return a dict.

    Keys are left in their original camelCase form.
    """
    text = _strip_json_comments(path.read_text())
    return json.loads(text)


# ── Config keys recognised in the JSON file (camelCase → snake_case) ─
_CONFIG_KEYS: dict[str, str] = {
    "data": "data",
    "gameConfig": "game_config",
    "modelConfig": "model_config",
    "modelType": "model_type",
    "out": "out",
    "resume": "resume",
    "epochs": "epochs",
    "patience": "patience",
    "batchSize": "batch_size",
    "lr": "lr",
    "valSplit": "val_split",
    "testSplit": "test_split",
    "logInterval": "log_interval",
    "checkpointDir": "checkpoint_dir",
    "loss": "loss",
    "lossSigma": "loss_sigma",
    "target": "target",
}

# Attributes whose CLI type is Path.
_PATH_ATTRS = {"data", "out", "resume", "checkpoint_dir"}


def _apply_config(args: argparse.Namespace, config: dict[str, object]) -> None:
    """Set *args* attributes from *config* without overriding explicit CLI values.

    Only keys listed in ``_CONFIG_KEYS`` are considered.  Values that
    were **not** provided on the command line (i.e. still at their
    parser default) are set from the config dict.

    Because ``argparse`` does not expose which values were explicitly
    provided, we compare against the parser defaults.  This means a CLI
    arg that happens to equal the default will **not** override a
    config-file value — but that is an acceptable trade-off since the
    user can always re-specify the value on the command line.
    """
    defaults = _ARG_DEFAULTS

    for json_key, attr in _CONFIG_KEYS.items():
        if json_key not in config:
            continue

        current = getattr(args, attr, None)
        default = defaults.get(attr)

        # If the CLI value differs from the default, the user explicitly
        # provided it — keep the CLI value.
        if current != default:
            continue

        value = config[json_key]
        if attr in _PATH_ATTRS and value is not None:
            value = Path(str(value))
        setattr(args, attr, value)


# ── Argument parsing ─────────────────────────────────────────────────


def _build_dataset(
    dataset_class: type,
    games: list[dict],
    game_cfg,
    n_buckets: int,
    args: argparse.Namespace,
):
    """Construct a dataset, forwarding the placement-specific ``target`` arg
    only when training a placement model. The state dataset doesn't accept a
    ``target`` keyword.
    """
    if args.model_type == "placement":
        return dataset_class(
            games, game_cfg, n_buckets=n_buckets, target=args.target
        )
    return dataset_class(games, game_cfg, n_buckets=n_buckets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GimburTransformer on simulation data.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file. CLI args override config values.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a JSONL file, a JSON file, or a directory of JSONL/JSON files.",
    )
    parser.add_argument(
        "--game-config",
        type=str,
        default=None,
        choices=sorted(CONFIGS_BY_NAME),
        help="Game configuration preset.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        choices=sorted(MODEL_CONFIGS_BY_NAME),
        help="Model size preset.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="state",
        choices=["state", "placement"],
        help="Model type: 'state' (default) or 'placement'.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="hard",
        choices=sorted(LOSS_MODES),
        help="Loss function mode (default: hard).",
    )
    parser.add_argument(
        "--loss-sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma for label smoothing (only used with --loss=gaussian, default: 2.0).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="winrate",
        choices=["winrate", "policy"],
        help=(
            "Training target for placement models: 'winrate' (default — predict "
            "per-action expected win probability) or 'policy' (predict each "
            "action's normalised share of MCTS visits among its siblings, "
            "yielding peaked PUCT-style action priors). Ignored for state models."
        ),
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
        help=(
            "Resume training. A .pt file loads model weights only. "
            "A directory loads the last epoch checkpoint (model + optimizer state)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for per-epoch checkpoints and stats.jsonl.",
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


# Capture parser defaults for _apply_config.
_ARG_DEFAULTS: dict[str, object] = {
    "config": None,
    "data": None,
    "game_config": None,
    "model_config": None,
    "model_type": "state",
    "out": Path("model.pt"),
    "resume": None,
    "checkpoint_dir": None,
    "epochs": 0,
    "patience": 5,
    "batch_size": 64,
    "lr": 1e-4,
    "val_split": 0.1,
    "test_split": 0.0,
    "log_interval": 50,
    "loss": "hard",
    "loss_sigma": 2.0,
    "target": "winrate",
}


# ── Checkpoint helpers ───────────────────────────────────────────────


def _find_latest_epoch_checkpoint(directory: Path) -> Path | None:
    """Return the ``epoch_N.pt`` file with the highest N, or ``None``."""
    best: tuple[int, Path] | None = None
    for p in directory.glob("epoch_*.pt"):
        stem = p.stem  # e.g. "epoch_12"
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            n = int(parts[1])
            if best is None or n > best[0]:
                best = (n, p)
    return best[1] if best is not None else None


def _save_epoch_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    epochs_without_improvement: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def _save_final_model(
    path: Path,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> None:
    """Save the final/best model checkpoint with metadata.

    The on-disk format is a dict containing ``model_state_dict`` plus
    metadata fields (``model_type``, ``model_config``, ``game_config``,
    ``target``) that downstream consumers (training resume, the inference
    server) need to interpret the model correctly. Existing checkpoints
    saved as a bare ``state_dict()`` can still be loaded via
    :func:`_load_model_state` for backward compatibility.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": args.model_type,
            "model_config": args.model_config,
            "game_config": args.game_config,
            "target": args.target,
        },
        path,
    )


def _load_model_state(path: Path, device: torch.device) -> dict:
    """Load a checkpoint and normalise to the new dict format.

    Accepts both:
      * Legacy bare ``state_dict()`` files (returned with ``model_state_dict``
        key only and ``target='winrate'`` for backward compatibility).
      * New format dicts with ``model_state_dict`` and metadata keys.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model_state_dict" in raw:
        return raw
    # Legacy: bare state_dict.
    return {"model_state_dict": raw, "target": "winrate"}


def _append_epoch_stats(
    stats_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    is_best: bool,
    elapsed_s: float,
) -> None:
    entry: dict[str, object] = {
        "epoch": epoch,
        "train_loss": round(train_loss, 6),
    }
    if val_loss is not None:
        entry["val_loss"] = round(val_loss, 6)
    entry["best"] = is_best
    entry["elapsed_s"] = round(elapsed_s, 2)
    with stats_path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


# ── Training epoch ───────────────────────────────────────────────────


def _run_epoch(
    model: GimburTransformer,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    log_interval: int,
    epoch: int,
    phase: str,
    loss_fn: LossFn = F.cross_entropy,
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

            loss = loss_fn(last_logits, targets)

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


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # ── Apply config file if provided ────────────────────────────────
    if args.config is not None:
        config = _load_config(args.config)
        _apply_config(args, config)

    # Validate required arguments.
    if args.data is None:
        raise SystemExit("Error: --data is required (via CLI or config file).")
    if args.game_config is None:
        raise SystemExit("Error: --game-config is required (via CLI or config file).")
    if args.model_config is None:
        raise SystemExit("Error: --model-config is required (via CLI or config file).")

    game_cfg = CONFIGS_BY_NAME[args.game_config]
    model_cfg = MODEL_CONFIGS_BY_NAME[args.model_config]

    # ── Select model and dataset class based on model type ───────────
    if args.model_type == "placement":
        model_class = GimburPlacementTransformer
        dataset_class = PlacementDataset
    else:
        model_class = GimburTransformer
        dataset_class = SimulationDataset

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Model ────────────────────────────────────────────────────────
    model = model_class(game_cfg, model_cfg)

    # ── Loss function ────────────────────────────────────────────────
    loss_cfg = LossConfig(mode=args.loss, sigma=args.loss_sigma)
    loss_fn = build_loss_fn(loss_cfg, n_buckets=model_cfg.n_buckets)

    # ── Resume handling ──────────────────────────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    resume_optimizer_state: dict | None = None

    if args.resume is not None:
        resume_path = args.resume
        if resume_path.is_dir():
            # Directory: find latest epoch checkpoint.
            ckpt_file = _find_latest_epoch_checkpoint(resume_path)
            if ckpt_file is None:
                raise SystemExit(
                    f"Error: --resume directory {resume_path} contains no epoch_*.pt checkpoints."
                )
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            resume_optimizer_state = ckpt["optimizer_state_dict"]
            start_epoch = ckpt["epoch"]
            best_val_loss = ckpt["best_val_loss"]
            epochs_without_improvement = ckpt["epochs_without_improvement"]
            print(
                f"Resumed from checkpoint {ckpt_file} "
                f"(epoch {start_epoch}, best_val_loss {best_val_loss:.4f})"
            )
        else:
            # .pt file: load model weights only (handles both legacy and new format).
            ckpt = _load_model_state(resume_path, device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Resumed model weights from {resume_path}")

    model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    loss_label = args.loss if args.loss == "hard" else f"{args.loss}(sigma={args.loss_sigma})"
    print(
        f"Model: {args.model_config} ({args.model_type}) for {args.game_config} "
        f"({param_count:,} parameters) on {device}, loss={loss_label}"
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

    train_dataset = _build_dataset(
        dataset_class, train_games, game_cfg, model_cfg.n_buckets, args
    )
    print(f"Train: {len(train_games):,} games -> {len(train_dataset):,} samples")

    val_dataset: SimulationDataset | PlacementDataset | None = None
    if val_games:
        val_dataset = _build_dataset(
            dataset_class, val_games, game_cfg, model_cfg.n_buckets, args
        )
        print(f"Val:   {len(val_games):,} games -> {len(val_dataset):,} samples")

    test_dataset: SimulationDataset | PlacementDataset | None = None
    if test_games:
        test_dataset = _build_dataset(
            dataset_class, test_games, game_cfg, model_cfg.n_buckets, args
        )
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
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)
        print("Restored optimizer state from checkpoint.")

    # ── Checkpoint directory setup ───────────────────────────────────
    checkpoint_dir: Path | None = args.checkpoint_dir
    stats_path: Path | None = None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_path = checkpoint_dir / "stats.jsonl"

    # ── Training loop ────────────────────────────────────────────────
    max_epochs = args.epochs if args.epochs > 0 else None
    epoch = start_epoch

    while True:
        epoch += 1
        if max_epochs is not None and epoch > max_epochs:
            break

        t_start = time.monotonic()

        train_loss = _run_epoch(
            model, train_loader, device, optimizer, args.log_interval, epoch, "train",
            loss_fn=loss_fn,
        )

        label = f"{epoch}/{max_epochs}" if max_epochs else str(epoch)
        msg = f"Epoch {label} | train loss {train_loss:.4f}"

        is_best = False
        val_loss: float | None = None
        if val_loader is not None:
            val_loss = _run_epoch(
                model, val_loader, device, None, 0, epoch, "val", loss_fn=loss_fn,
            )
            msg += f" | val loss {val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                is_best = True
                _save_final_model(args.out, model, args)
                msg += " (best, saved)"
            else:
                epochs_without_improvement += 1
        else:
            _save_final_model(args.out, model, args)
            is_best = True

        elapsed = time.monotonic() - t_start
        msg += f" | {elapsed:.1f}s"
        print(msg)

        # ── Per-epoch checkpoint ─────────────────────────────────────
        if checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / f"epoch_{epoch}.pt"
            _save_epoch_checkpoint(
                ckpt_path, epoch, model, optimizer, best_val_loss, epochs_without_improvement
            )
            assert stats_path is not None
            _append_epoch_stats(stats_path, epoch, train_loss, val_loss, is_best, elapsed)

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
        # Reload best checkpoint for test evaluation (handles both legacy and new format).
        ckpt = _load_model_state(args.out, device)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss = _run_epoch(
            model, test_loader, device, None, 0, 0, "test", loss_fn=loss_fn,
        )
        print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
