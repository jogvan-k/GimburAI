"""Tests for the data loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gimbur_nn.data_loader import (
    SimulationDataset,
    _prob_to_bucket,
    _win_probability,
    load_samples,
)
from gimbur_nn.game_config import MINI_2P

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_game(
    *,
    board: str,
    board_perms: list[str],
    states: list[dict],
    n_players: int,
    map_name: str = "mini",
) -> dict:
    """Create a minimal JSONL game record."""
    return {
        "version": 1,
        "seed": 42,
        "map": map_name,
        "players": n_players,
        "winner": 1,
        "turns": 10,
        "constraints": {
            "searchTimeMs": 500,
            "maxSimulations": 100,
            "maxRolloutDepth": 50,
            "actionRolloutLimit": 10,
        },
        "board": {
            "serialized": board,
            "permutations": board_perms,
        },
        "states": states,
    }


def _make_state(
    *,
    state_str: str,
    state_perms: list[str],
    best_action_wins: list[float],
    player_turn: int = 1,
) -> dict:
    """Create a minimal state entry within a game."""
    return {
        "playerTurn": player_turn,
        "serializedState": state_str,
        "simulations": 100,
        "elapsedMs": 50,
        "winRate": 0.5,
        "wins": best_action_wins,
        "bestActionWinRate": 0.5,
        "bestActionWins": best_action_wins,
        "bestActionRollouts": 50,
        "permutations": state_perms,
    }


def _write_jsonl(path: Path, games: list[dict]) -> None:
    with path.open("w") as f:
        for game in games:
            f.write(json.dumps(game) + "\n")


# ---------------------------------------------------------------------------
# Mini 2p example strings (from docs/state-serialization.md)
# ---------------------------------------------------------------------------

MINI_BOARD = "w5lb3ls4lW3hd0nW4ho2l|gsgbgw"
MINI_STATE_ONLY = (
    "4|-t-|__|"
    "._._._._._._v-._._._._._._._v+._._._._._._._._._|"
    "_____-_______+________________|"
    "21010/00130|0/0|00000/00000"
)


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestWinProbability:
    def test_player1_wins(self) -> None:
        assert _win_probability([80.0, 20.0], 1) == pytest.approx(0.8)

    def test_player2_wins(self) -> None:
        assert _win_probability([80.0, 20.0], 2) == pytest.approx(0.2)

    def test_three_players(self) -> None:
        assert _win_probability([30.0, 50.0, 20.0], 2) == pytest.approx(0.5)

    def test_zero_total_returns_uniform(self) -> None:
        assert _win_probability([0.0, 0.0], 1) == pytest.approx(0.5)
        assert _win_probability([0.0, 0.0, 0.0], 2) == pytest.approx(1.0 / 3)


class TestProbToBucket:
    def test_zero(self) -> None:
        assert _prob_to_bucket(0.0, 128) == 0

    def test_one(self) -> None:
        # 1.0 * 128 = 128, clamped to 127
        assert _prob_to_bucket(1.0, 128) == 127

    def test_half(self) -> None:
        # 0.5 * 128 = 64
        assert _prob_to_bucket(0.5, 128) == 64

    def test_just_below_boundary(self) -> None:
        # 0.5 / 128 = 0.00390625, bucket 0
        assert _prob_to_bucket(0.003, 128) == 0

    def test_small_bucket_count(self) -> None:
        # 0.6 * 4 = 2.4 -> bucket 2
        assert _prob_to_bucket(0.6, 4) == 2


# ---------------------------------------------------------------------------
# Integration: load_samples
# ---------------------------------------------------------------------------


class TestLoadSamples:
    def test_no_permutations_2_players(self, tmp_path: Path) -> None:
        """Without permutations: 1 state * 1 combo * 2 players = 2 samples."""
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[60.0, 40.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game])
        samples = load_samples(tmp_path / "test.jsonl", MINI_2P)

        assert len(samples) == 2

        # All token tensors should have correct length.
        for token_ids, bucket in samples:
            assert token_ids.shape == (MINI_2P.state_token_size,)
            assert 0 <= bucket < 128

    def test_with_permutations(self, tmp_path: Path) -> None:
        """With 2 permutations: 1 state * 3 combos * 2 players = 6 samples."""
        # Use the identity board/state as "permutations" for simplicity.
        # The loader doesn't validate content, just expansion count.
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[MINI_BOARD, MINI_BOARD],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[MINI_STATE_ONLY, MINI_STATE_ONLY],
                    best_action_wins=[60.0, 40.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game])
        samples = load_samples(tmp_path / "test.jsonl", MINI_2P)

        # (1 + 2 permutations) * 2 players = 6
        assert len(samples) == 6

    def test_multiple_states(self, tmp_path: Path) -> None:
        """2 states, no permutations: 2 * 1 * 2 = 4 samples."""
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[70.0, 30.0],
                ),
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[40.0, 60.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game])
        samples = load_samples(tmp_path / "test.jsonl", MINI_2P)

        assert len(samples) == 4

    def test_multiple_games(self, tmp_path: Path) -> None:
        """2 games with 1 state each, no perms: 2 * 1 * 1 * 2 = 4 samples."""
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[50.0, 50.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game, game])
        samples = load_samples(tmp_path / "test.jsonl", MINI_2P)

        assert len(samples) == 4

    def test_labels_reflect_win_probability(self, tmp_path: Path) -> None:
        """Player with higher wins should get a higher bucket index."""
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[90.0, 10.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game])
        samples = load_samples(tmp_path / "test.jsonl", MINI_2P)

        # samples[0] is player 1 (no rotation), samples[1] is player 2.
        _, bucket_p1 = samples[0]
        _, bucket_p2 = samples[1]

        # Player 1 has 90% win rate -> high bucket.
        # Player 2 has 10% win rate -> low bucket.
        assert bucket_p1 > bucket_p2

    def test_player_rotation_applied(self, tmp_path: Path) -> None:
        """Token IDs should differ between player 1 and player 2 views."""
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[60.0, 40.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game])
        samples = load_samples(tmp_path / "test.jsonl", MINI_2P)

        tokens_p1, _ = samples[0]
        tokens_p2, _ = samples[1]

        # The current player is '-' (P1) in the original; after rotation
        # for P2 it becomes '+'. So the tokens must differ.
        assert not torch.equal(tokens_p1, tokens_p2)

    def test_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "empty.jsonl").write_text("")
        samples = load_samples(tmp_path / "empty.jsonl", MINI_2P)
        assert len(samples) == 0

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[50.0, 50.0],
                ),
            ],
            n_players=2,
        )
        content = "\n" + json.dumps(game) + "\n\n"
        (tmp_path / "test.jsonl").write_text(content)
        samples = load_samples(tmp_path / "test.jsonl", MINI_2P)
        assert len(samples) == 2


# ---------------------------------------------------------------------------
# SimulationDataset
# ---------------------------------------------------------------------------


class TestSimulationDataset:
    def test_len_and_getitem(self, tmp_path: Path) -> None:
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[60.0, 40.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game])
        ds = SimulationDataset(tmp_path / "test.jsonl", MINI_2P)

        assert len(ds) == 2

        token_ids, target = ds[0]
        assert token_ids.shape == (MINI_2P.state_token_size,)
        assert token_ids.dtype == torch.int32
        assert target.dtype == torch.long
        assert 0 <= target.item() < 128

    def test_compatible_with_dataloader(self, tmp_path: Path) -> None:
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    best_action_wins=[50.0, 50.0],
                ),
            ],
            n_players=2,
        )
        _write_jsonl(tmp_path / "test.jsonl", [game])
        ds = SimulationDataset(tmp_path / "test.jsonl", MINI_2P)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)

        batch_tokens, batch_targets = next(iter(loader))
        assert batch_tokens.shape == (2, MINI_2P.state_token_size)
        assert batch_targets.shape == (2,)
