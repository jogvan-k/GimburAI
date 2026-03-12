"""
Data loader for training on JSONL files exported by ``gimbur simulate --export``.

Each JSONL line represents one game.  The loader expands each state entry
into ``(1 + n_permutations) * n_players`` training samples by combining
board symmetry permutations with player rotation.

Each sample is a ``(token_ids, target_bucket)`` pair where:

- ``token_ids`` is the tokenized compact state rotated so the target
  player occupies the player-1 slot.
- ``target_bucket`` is the index of the nearest bucket centre to the
  target player's win probability derived from ``bestActionWins``.

Usage::

    from gimbur_nn.data_loader import SimulationDataset

    dataset = SimulationDataset("games.jsonl", game_cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .tokenizer import rotate_player_state, tokenize

if TYPE_CHECKING:
    from .game_config import GameConfig

_STRIP = str.maketrans("", "", "|/")


def _compact(human_readable: str) -> str:
    """Strip ``|`` and ``/`` separators to produce the compact form."""
    return human_readable.translate(_STRIP)


def _win_probability(best_action_wins: list[float], player: int) -> float:
    """Compute win probability for *player* (1-based) from MCTS win counts.

    Returns ``bestActionWins[player - 1] / sum(bestActionWins)``.
    If the total is zero, returns ``1 / n_players`` (uniform).
    """
    total = sum(best_action_wins)
    if total == 0.0:
        return 1.0 / len(best_action_wins)
    return best_action_wins[player - 1] / total


def _prob_to_bucket(prob: float, n_buckets: int) -> int:
    """Map a win probability to the nearest bucket index.

    Bucket centres are at ``(i + 0.5) / n_buckets`` for ``i`` in
    ``0 .. n_buckets - 1``.  Returns the index of the closest centre.
    """
    bucket = int(prob * n_buckets)
    return min(bucket, n_buckets - 1)


def load_samples(
    path: str | Path,
    cfg: GameConfig,
    *,
    n_buckets: int = 128,
) -> list[tuple[torch.Tensor, int]]:
    """Load all training samples from a JSONL file.

    Each sample is a ``(token_ids, target_bucket)`` tuple.

    Args:
        path: Path to the JSONL file exported by ``gimbur simulate --export``.
        cfg: Game configuration matching the exported data.
        n_buckets: Number of output buckets (must match the model).

    Returns:
        A flat list of ``(token_ids, target_bucket)`` pairs.
    """
    path = Path(path)
    samples: list[tuple[torch.Tensor, int]] = []
    n_players = cfg.player_count

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            game = json.loads(line)
            _process_game(game, cfg, n_players, n_buckets, samples)

    return samples


def _process_game(
    game: dict,
    cfg: GameConfig,
    n_players: int,
    n_buckets: int,
    samples: list[tuple[torch.Tensor, int]],
) -> None:
    """Expand one game record into training samples."""
    board_serialized: str = game["board"]["serialized"]
    board_permutations: list[str] = game["board"]["permutations"]

    for state_entry in game["states"]:
        best_action_wins: list[float] = state_entry["bestActionWins"]
        state_serialized: str = state_entry["serializedState"]
        state_permutations: list[str] = state_entry["permutations"]

        # Identity combo + one combo per symmetry permutation.
        board_variants = [board_serialized, *board_permutations]
        state_variants = [state_serialized, *state_permutations]

        for board_str, state_str in zip(board_variants, state_variants):
            # Reconstruct full human-readable form, then compact.
            full_hr = board_str + "|" + state_str
            compact = _compact(full_hr)

            for player in range(1, n_players + 1):
                rotated = rotate_player_state(compact, player, cfg)
                token_ids = tokenize(rotated)
                prob = _win_probability(best_action_wins, player)
                bucket = _prob_to_bucket(prob, n_buckets)
                samples.append((token_ids, bucket))


def load_samples_from_dir(
    directory: str | Path,
    cfg: GameConfig,
    *,
    n_buckets: int = 128,
) -> list[tuple[torch.Tensor, int]]:
    """Load training samples from all ``.jsonl`` files in *directory*.

    Files are discovered with ``*.jsonl`` glob and processed in sorted
    order.  Returns a flat list identical in format to :func:`load_samples`.
    """
    directory = Path(directory)
    files = sorted(directory.glob("*.jsonl"))
    samples: list[tuple[torch.Tensor, int]] = []
    for f in files:
        samples.extend(load_samples(f, cfg, n_buckets=n_buckets))
    return samples


class SimulationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset backed by JSONL export files.

    Loads all samples into memory on construction.  Each item is a
    ``(token_ids, target_bucket)`` pair where ``token_ids`` is a 1-D
    ``int`` tensor and ``target_bucket`` is a scalar ``long`` tensor.

    *path* may be a single ``.jsonl`` file **or** a directory containing
    one or more ``.jsonl`` files.

    Args:
        path: Path to a JSONL file or a directory of JSONL files.
        cfg: Game configuration matching the exported data.
        n_buckets: Number of output buckets (default 128).
    """

    def __init__(
        self,
        path: str | Path,
        cfg: GameConfig,
        *,
        n_buckets: int = 128,
    ) -> None:
        p = Path(path)
        if p.is_dir():
            raw = load_samples_from_dir(p, cfg, n_buckets=n_buckets)
        else:
            raw = load_samples(p, cfg, n_buckets=n_buckets)
        self._tokens = [t for t, _ in raw]
        self._targets = torch.tensor([b for _, b in raw], dtype=torch.long)

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tokens[idx], self._targets[idx]
