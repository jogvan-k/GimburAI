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

The train/val/test split is performed at the **game** level so that all
samples derived from a single game belong to exactly one split.

Usage::

    from gimbur_nn.data_loader import load_games, split_games, SimulationDataset

    games = load_games("exports/")
    train_games, val_games, test_games = split_games(games, val=0.1, test=0.1)
    train_ds = SimulationDataset(train_games, game_cfg)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .state_tokenizer import StateTokenizer

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


# ── Loading games ────────────────────────────────────────────────────


def load_games(path: str | Path) -> list[dict]:
    """Load game records from a JSONL file, JSON file, or a directory.

    Supported inputs:

    - A single ``.jsonl`` file — each line is one game JSON object.
    - A single ``.json`` file — the file contains one game JSON object.
    - A directory containing ``.jsonl`` and/or ``.json`` files.

    Args:
        path: A ``.jsonl`` file, a ``.json`` file, or a directory
            containing one or more of either.

    Returns:
        A list of parsed game dicts.
    """
    p = Path(path)
    if p.is_dir():
        jsonl_files = sorted(p.glob("*.jsonl"))
        json_files = sorted(p.glob("*.json"))
    elif p.suffix == ".json":
        jsonl_files = []
        json_files = [p]
    else:
        # Assume JSONL (covers .jsonl and any other extension).
        jsonl_files = [p]
        json_files = []

    games: list[dict] = []

    # Load JSONL files (one game per line).
    for f in jsonl_files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                games.append(json.loads(line))

    # Load per-game JSON files (one game per file).
    for f in json_files:
        with f.open() as fh:
            games.append(json.load(fh))

    return games


# ── Splitting ────────────────────────────────────────────────────────


def split_games(
    games: list[dict],
    *,
    val: float = 0.1,
    test: float = 0.0,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split games into train / val / test sets.

    The split is performed at the **game** level so samples from the
    same game never leak across sets.

    Args:
        games: Full list of game dicts.
        val: Fraction of games for validation (0 to skip).
        test: Fraction of games for test (0 to skip).
        seed: Random seed for reproducibility.

    Returns:
        ``(train_games, val_games, test_games)`` — three disjoint lists.
    """
    if val < 0 or test < 0 or val + test > 1.0:
        msg = f"val ({val}) + test ({test}) must be in [0, 1]"
        raise ValueError(msg)

    n = len(games)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)

    n_test = int(n * test)
    n_val = int(n * val)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    return (
        [games[i] for i in train_idx],
        [games[i] for i in val_idx],
        [games[i] for i in test_idx],
    )


# ── Sample expansion ────────────────────────────────────────────────


def expand_games(
    games: list[dict],
    cfg: GameConfig,
    *,
    n_buckets: int = 128,
    tokenizer: StateTokenizer | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """Expand a list of game dicts into ``(token_ids, bucket)`` samples.

    Each game's states are expanded via symmetry permutations and player
    rotation.

    Args:
        games: List of parsed game dicts.
        cfg: Game configuration matching the exported data.
        n_buckets: Number of output buckets (must match the model).

    Returns:
        A flat list of ``(token_ids, target_bucket)`` pairs.
    """
    if tokenizer is None:
        tokenizer = StateTokenizer(cfg)
    samples: list[tuple[torch.Tensor, int]] = []
    n_players = cfg.player_count
    for game in games:
        _process_game(game, cfg, n_players, n_buckets, samples, tokenizer)
    return samples


def _process_game(
    game: dict,
    cfg: GameConfig,
    n_players: int,
    n_buckets: int,
    samples: list[tuple[torch.Tensor, int]],
    tokenizer: StateTokenizer,
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
                rotated = tokenizer.rotate_player_state(compact, player)
                token_ids = tokenizer.tokenize(rotated)
                prob = _win_probability(best_action_wins, player)
                bucket = _prob_to_bucket(prob, n_buckets)
                samples.append((token_ids, bucket))


# ── Legacy convenience (used by tests) ───────────────────────────────


def load_samples(
    path: str | Path,
    cfg: GameConfig,
    *,
    n_buckets: int = 128,
) -> list[tuple[torch.Tensor, int]]:
    """Load and expand all samples from a JSONL file.

    Convenience wrapper: ``load_games(path)`` → ``expand_games(…)``.
    """
    tok = StateTokenizer(cfg)
    return expand_games(load_games(path), cfg, n_buckets=n_buckets, tokenizer=tok)


# ── Dataset ──────────────────────────────────────────────────────────


class SimulationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset backed by a list of game records.

    Expands games into samples on construction and holds them in memory.
    Each item is a ``(token_ids, target_bucket)`` pair where
    ``token_ids`` is a 1-D ``int`` tensor and ``target_bucket`` is a
    scalar ``long`` tensor.

    Args:
        games: List of parsed game dicts (from :func:`load_games`).
        cfg: Game configuration matching the exported data.
        n_buckets: Number of output buckets (default 128).
    """

    def __init__(
        self,
        games: list[dict],
        cfg: GameConfig,
        *,
        n_buckets: int = 128,
    ) -> None:
        tok = StateTokenizer(cfg)
        raw = expand_games(games, cfg, n_buckets=n_buckets, tokenizer=tok)
        self._tokens = [t for t, _ in raw]
        self._targets = torch.tensor([b for _, b in raw], dtype=torch.long)

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tokens[idx], self._targets[idx]
