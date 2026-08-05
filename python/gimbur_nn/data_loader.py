"""
Data loader for training on JSONL files exported by ``gimbur simulate --export``.

Each JSONL line represents one game.  The loader expands each state entry
into ``(1 + n_permutations) * n_players`` training samples by combining
board symmetry permutations with player rotation.

Each sample is a ``(token_ids, player_value_target)`` pair where:

- ``token_ids`` is the tokenized compact state rotated so the target
  player occupies the player-1 slot.
- ``player_value_target`` blends normalized MCTS wins with the final winner,
  then rotates consistently with the serialized player identities.

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
import math
import random
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from .placement_tokenizer import PlacementTokenizer
from .state_tokenizer import StateTokenizer

if TYPE_CHECKING:
    from .game_config import GameConfig

_STRIP = str.maketrans("", "", "|/")
_COMPOSITE_PLACEMENT_ACTION = re.compile(r"^(\d+)(N|NE|SE|S|SW|NW)$")


def _compact(human_readable: str) -> str:
    """Strip ``|`` and ``/`` separators to produce the compact form."""
    return human_readable.translate(_STRIP)


def _composite_export_vertex(action: str) -> int:
    """Extract a settlement vertex from the current composite export format."""
    match = _COMPOSITE_PLACEMENT_ACTION.fullmatch(action)
    if match is None:
        raise ValueError(f"invalid composite placement action: {action}")
    return int(match.group(1))


def _normalize_wins(wins: list[float], player_count: int) -> torch.Tensor | None:
    """Normalize valid MCTS wins to a player distribution."""
    if len(wins) != player_count:
        return None
    target = torch.tensor(wins, dtype=torch.float32).clamp_min(0)
    total = target.sum()
    if total <= 0:
        return None
    return target / total


def _value_target(
    wins: list[float],
    winner: object,
    player_count: int,
    mcts_value_weight: float,
) -> torch.Tensor | None:
    """Blend MCTS evidence and a valid 1-based terminal winner."""
    if not 0.0 <= mcts_value_weight <= 1.0:
        raise ValueError("mcts_value_weight must be between 0 and 1")

    mcts_target = _normalize_wins(wins, player_count)
    terminal_target = None
    if isinstance(winner, int) and not isinstance(winner, bool) and 1 <= winner <= player_count:
        terminal_target = torch.zeros(player_count, dtype=torch.float32)
        terminal_target[winner - 1] = 1.0

    if mcts_target is None:
        return terminal_target
    if terminal_target is None:
        return mcts_target
    return mcts_value_weight * mcts_target + (1.0 - mcts_value_weight) * terminal_target


def rotate_player_target(target: torch.Tensor, player_turn: int) -> torch.Tensor:
    """Rotate an original-player target so the acting player's slot is first."""
    if not 1 <= player_turn <= target.numel():
        raise ValueError("playerTurn must identify a target player slot")
    return torch.roll(target, shifts=-(player_turn - 1))


def _scheduled_mcts_value_weight(
    turn_number: int,
    total_turns: int,
    start: float,
    end: float,
) -> float:
    """Linearly interpolate the MCTS blend by turn / exported total turns."""
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError("MCTS value weights must be between 0 and 1")
    progress = min(max(turn_number / max(1, total_turns), 0.0), 1.0)
    return start + (end - start) * progress


def _select_state_entries(
    games: list[dict],
    statistic: str,
    upper_percentage: float,
) -> list[list[dict]]:
    """Select full-game roots using dataset-level total-VP bucket caps."""
    if statistic not in ("median", "average"):
        raise ValueError("victory_point_sampling_statistic must be 'median' or 'average'")
    if upper_percentage < 0:
        raise ValueError("victory_point_sampling_upper_percentage must be non-negative")

    buckets: dict[int, list[tuple[int, int]]] = {}
    mandatory: set[tuple[int, int]] = set()
    for game_index, game in enumerate(games):
        states = game["states"]
        # Adaptive VP sampling requires authoritative score metadata. Older
        # exports may still feed placement replay, but not state replay.
        if any(not (state.get("scores") or []) for state in states):
            continue
        if states:
            mandatory.add((game_index, len(states) - 1))
        post_placement_index = next(
            (
                state_index
                for state_index, state in enumerate(states)
                if state["turnNumber"] == 1 and state["stage"] == "r"
            ),
            None,
        )
        if post_placement_index is not None:
            mandatory.add((game_index, post_placement_index))

        for state_index, state in enumerate(states):
            scores = state["scores"]
            total_victory_points = math.floor(sum(float(score) for score in scores))
            buckets.setdefault(total_victory_points, []).append((game_index, state_index))

    if not buckets:
        return [[] for _ in games]

    bucket_sizes = [len(identities) for identities in buckets.values()]
    reference = (
        statistics.median(bucket_sizes) if statistic == "median" else statistics.fmean(bucket_sizes)
    )
    cap = math.ceil(reference * (1 + upper_percentage))
    dataset_seed = ":".join(str(game.get("seed", "")) for game in games)
    selected: set[tuple[int, int]] = set()
    for total_victory_points, identities in buckets.items():
        if len(identities) <= cap:
            selected.update(identities)
            continue

        required = mandatory.intersection(identities)
        selected.update(required)
        available = max(0, cap - len(required))
        candidates = [identity for identity in identities if identity not in required]
        rng = random.Random(f"{dataset_seed}:{total_victory_points}")
        selected.update(rng.sample(candidates, min(available, len(candidates))))

    return [
        [
            state
            for state_index, state in enumerate(game["states"])
            if (game_index, state_index) in selected
        ]
        for game_index, game in enumerate(games)
    ]


# ── Loading games ────────────────────────────────────────────────────


def load_games(path: str | Path | list[str | Path]) -> list[dict]:
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
    if isinstance(path, list):
        return [game for item in path for game in load_games(item)]

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
    tokenizer: StateTokenizer | None = None,
    mcts_value_weight_start: float = 0.9,
    mcts_value_weight_end: float = 0.1,
    victory_point_sampling_statistic: str = "median",
    victory_point_sampling_upper_percentage: float = 0.10,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Expand games into player-rotated state and value-distribution samples.

    Each game's states are expanded via symmetry permutations and player
    rotation.

    Args:
        games: List of parsed game dicts.
        cfg: Game configuration matching the exported data.
    Returns:
        A flat list of ``(token_ids, player_value_target)`` pairs.
    """
    if tokenizer is None:
        tokenizer = StateTokenizer(cfg)
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    n_players = cfg.player_count
    selected_entries = _select_state_entries(
        games,
        victory_point_sampling_statistic,
        victory_point_sampling_upper_percentage,
    )
    for game, state_entries in zip(games, selected_entries):
        _process_game(
            game,
            cfg,
            n_players,
            samples,
            tokenizer,
            mcts_value_weight_start,
            mcts_value_weight_end,
            state_entries,
        )
    return samples


def _process_game(
    game: dict,
    cfg: GameConfig,
    n_players: int,
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    tokenizer: StateTokenizer,
    mcts_value_weight_start: float,
    mcts_value_weight_end: float,
    state_entries: list[dict],
) -> None:
    """Expand one game record into training samples."""
    board_serialized: str = game["board"]["serialized"]
    board_permutations: list[str] = game["board"]["permutations"]

    for state_entry in state_entries:
        wins: list[float] = state_entry.get("wins") or []
        state_serialized: str = state_entry["serializedState"]
        state_permutations: list[str] = state_entry["permutations"]

        # Identity combo + one combo per symmetry permutation.
        board_variants = [board_serialized, *board_permutations]
        state_variants = [state_serialized, *state_permutations]

        weight = _scheduled_mcts_value_weight(
            state_entry["turnNumber"],
            game.get("turns", 0),
            mcts_value_weight_start,
            mcts_value_weight_end,
        )
        target = _value_target(wins, game.get("winner"), n_players, weight)
        if target is None:
            continue
        for board_str, state_str in zip(board_variants, state_variants):
            # Reconstruct full human-readable form, then compact.
            full_hr = board_str + "|" + state_str
            compact = _compact(full_hr)

            for player in range(1, n_players + 1):
                rotated = tokenizer.rotate_player_state(compact, player)
                token_ids = tokenizer.tokenize(rotated)
                rotation = player - 1
                rotated_target = torch.roll(target, shifts=-rotation)
                samples.append((token_ids, rotated_target))


# ── Legacy convenience (used by tests) ───────────────────────────────


def load_samples(
    path: str | Path,
    cfg: GameConfig,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Load and expand all samples from a JSONL file.

    Convenience wrapper: ``load_games(path)`` → ``expand_games(…)``.
    """
    tok = StateTokenizer(cfg)
    return expand_games(load_games(path), cfg, tokenizer=tok)


# ── Dataset ──────────────────────────────────────────────────────────


class SimulationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch dataset backed by a list of game records.

    Expands games into samples on construction and holds them in memory.
    Each item contains token IDs and a float player-value distribution.

    Args:
        games: List of parsed game dicts (from :func:`load_games`).
        cfg: Game configuration matching the exported data.
    """

    def __init__(
        self,
        games: list[dict],
        cfg: GameConfig,
        *,
        mcts_value_weight_start: float = 0.9,
        mcts_value_weight_end: float = 0.1,
        victory_point_sampling_statistic: str = "median",
        victory_point_sampling_upper_percentage: float = 0.10,
    ) -> None:
        tok = StateTokenizer(cfg)
        raw = expand_games(
            games,
            cfg,
            tokenizer=tok,
            mcts_value_weight_start=mcts_value_weight_start,
            mcts_value_weight_end=mcts_value_weight_end,
            victory_point_sampling_statistic=victory_point_sampling_statistic,
            victory_point_sampling_upper_percentage=victory_point_sampling_upper_percentage,
        )
        self._tokens = [t for t, _ in raw]
        self._targets = [target for _, target in raw]

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tokens[idx], self._targets[idx]


# ── Placement phase sample expansion ─────────────────────────────────


def expand_placement_games(
    games: list[dict],
    cfg: GameConfig,
    *,
    tokenizer: PlacementTokenizer | None = None,
    target: str = "winrate",
    advantage: bool = False,
    mcts_value_weight_start: float = 0.9,
    policy_target_temperature: float = 1.0,
) -> list:
    """Emit one state sample per exported state and symmetry permutation.

    ``target`` and ``advantage`` remain accepted for config compatibility.
    ``target="combined"`` emits dense policy targets and legal masks;
    all other supported target values emit value-only samples.
    """
    if tokenizer is None:
        tokenizer = PlacementTokenizer(cfg)
    if target not in ("winrate", "combined"):
        raise ValueError("Placement target must be 'winrate' or 'combined'.")
    if not policy_target_temperature > 0:
        raise ValueError("Placement policy target temperature must be greater than zero.")
    samples: list = []
    for game in games:
        _process_placement_game(
            game,
            cfg.player_count,
            samples,
            tokenizer,
            target,
            mcts_value_weight_start,
            policy_target_temperature,
        )
    return samples


def _process_placement_game(
    game: dict,
    player_count: int,
    samples: list,
    tokenizer: PlacementTokenizer,
    target: str = "winrate",
    mcts_value_weight_start: float = 0.9,
    policy_target_temperature: float = 1.0,
) -> None:
    """Expand one placement game into state-level value/policy samples."""
    for state_entry in game.get("placementStates", game.get("states", [])):
        state_serialized: str = state_entry["serializedState"]
        state_permutations: list[str] = state_entry["permutations"]

        all_variants = [state_serialized, *state_permutations]
        actions = state_entry["actions"]
        if not actions:
            continue
        rollouts = [max(0, int(action.get("rollouts", 0))) for action in actions]
        total_rollouts = sum(rollouts)
        if target == "combined" and total_rollouts == 0:
            continue
        weighted_value = torch.zeros(player_count, dtype=torch.float32)
        value_weight = 0
        for action, rollout in zip(actions, rollouts):
            if rollout <= 0:
                continue
            wins = action.get("wins", action.get("Wins", []))
            action_value = _normalize_wins(wins, player_count)
            if action_value is not None:
                weighted_value += action_value * rollout
                value_weight += rollout
        mcts_wins = (weighted_value / value_weight).tolist() if value_weight > 0 else []
        value_target = _value_target(
            mcts_wins, game.get("winner"), player_count, mcts_value_weight_start
        )
        if value_target is None:
            continue
        value_target = rotate_player_target(value_target, state_entry["playerTurn"])

        for variant_idx, state_str in enumerate(all_variants):
            token_ids = tokenizer.tokenize_state(_compact(state_str))
            if target != "combined":
                samples.append((token_ids, value_target))
                continue

            policy = torch.zeros(tokenizer.policy_size, dtype=torch.float32)
            legal_mask = torch.zeros(tokenizer.policy_size, dtype=torch.bool)
            for action_entry, rollout in zip(actions, rollouts):
                action = (
                    action_entry["action"]
                    if variant_idx == 0
                    else action_entry["permutations"][variant_idx - 1]
                )
                action_idx = tokenizer.vertex_action_index(_composite_export_vertex(action))
                policy[action_idx] += rollout / total_rollouts
                legal_mask[action_idx] = True
            if policy_target_temperature != 1.0:
                positive = policy > 0
                log_weights = policy[positive].double().log()
                weights = ((log_weights - log_weights.max()) / policy_target_temperature).exp()
                policy[positive] = (weights / weights.sum()).float()
            samples.append((token_ids, value_target, policy, legal_mask))


# ── Placement Dataset ────────────────────────────────────────────────


class PlacementDataset(Dataset[tuple[torch.Tensor, ...]]):
    """In-memory placement dataset with one sample per state/permutation."""

    def __init__(
        self,
        games: list[dict],
        cfg: GameConfig,
        *,
        target: str = "winrate",
        advantage: bool = False,
        mcts_value_weight_start: float = 0.9,
        policy_target_temperature: float = 1.0,
    ) -> None:
        tok = PlacementTokenizer(cfg)
        raw = expand_placement_games(
            games,
            cfg,
            tokenizer=tok,
            target=target,
            advantage=advantage,
            mcts_value_weight_start=mcts_value_weight_start,
            policy_target_temperature=policy_target_temperature,
        )
        self._combined = target == "combined"
        self._tokens = [t[0] for t in raw]
        if self._combined:
            self._value_targets = [t[1] for t in raw]
            self._policy_targets = [t[2] for t in raw]
            self._legal_masks = [t[3] for t in raw]
        else:
            self._targets = [t[1] for t in raw]

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        if self._combined:
            return (
                self._tokens[idx],
                self._value_targets[idx],
                self._policy_targets[idx],
                self._legal_masks[idx],
            )
        return self._tokens[idx], self._targets[idx]
