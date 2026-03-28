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
    expand_games,
    load_games,
    load_samples,
    split_games,
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


def _simple_game(wins: list[float] | None = None) -> dict:
    """Shorthand for a 1-state mini 2p game with no permutations."""
    return _make_game(
        board=MINI_BOARD,
        board_perms=[],
        states=[
            _make_state(
                state_str=MINI_STATE_ONLY,
                state_perms=[],
                best_action_wins=wins or [50.0, 50.0],
            ),
        ],
        n_players=2,
    )


# ---------------------------------------------------------------------------
# Mini 2p example strings (from docs/state-action-serialization.md)
# ---------------------------------------------------------------------------

MINI_BOARD = "w5lb3ls4lW3hd0nW4ho2l|gsgbgw"
MINI_STATE_ONLY = (
    "4|-t|__|"
    "._._._._._._v-._._._._._._._v+._._._._._._._._._|"
    "_____-_______+________________|"
    "21010/00130|0/0|00000/00000|0000"
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
# load_games
# ---------------------------------------------------------------------------


class TestLoadGames:
    def test_single_file(self, tmp_path: Path) -> None:
        games = [_simple_game(), _simple_game()]
        _write_jsonl(tmp_path / "a.jsonl", games)
        loaded = load_games(tmp_path / "a.jsonl")
        assert len(loaded) == 2

    def test_directory(self, tmp_path: Path) -> None:
        _write_jsonl(tmp_path / "a.jsonl", [_simple_game()])
        _write_jsonl(tmp_path / "b.jsonl", [_simple_game(), _simple_game()])
        loaded = load_games(tmp_path)
        assert len(loaded) == 3

    def test_empty_file(self, tmp_path: Path) -> None:
        (tmp_path / "empty.jsonl").write_text("")
        loaded = load_games(tmp_path / "empty.jsonl")
        assert len(loaded) == 0

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        content = "\n" + json.dumps(_simple_game()) + "\n\n"
        (tmp_path / "test.jsonl").write_text(content)
        loaded = load_games(tmp_path / "test.jsonl")
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# split_games
# ---------------------------------------------------------------------------


class TestSplitGames:
    def test_default_split(self) -> None:
        games = [_simple_game() for _ in range(100)]
        train, val, test = split_games(games)
        assert len(train) == 90
        assert len(val) == 10
        assert len(test) == 0
        assert len(train) + len(val) + len(test) == 100

    def test_val_and_test(self) -> None:
        games = [_simple_game() for _ in range(100)]
        train, val, test = split_games(games, val=0.1, test=0.1)
        assert len(test) == 10
        assert len(val) == 10
        assert len(train) == 80

    def test_no_split(self) -> None:
        games = [_simple_game() for _ in range(10)]
        train, val, test = split_games(games, val=0.0, test=0.0)
        assert len(train) == 10
        assert len(val) == 0
        assert len(test) == 0

    def test_deterministic(self) -> None:
        games = [_simple_game() for _ in range(50)]
        t1, v1, te1 = split_games(games, val=0.2, test=0.1)
        t2, v2, te2 = split_games(games, val=0.2, test=0.1)
        assert t1 == t2
        assert v1 == v2
        assert te1 == te2

    def test_invalid_fractions_raises(self) -> None:
        with pytest.raises(ValueError):
            split_games([], val=0.6, test=0.5)

    def test_small_dataset_no_val(self) -> None:
        """With only 1 game, int(1*0.1)=0 so all go to train."""
        games = [_simple_game()]
        train, val, test = split_games(games, val=0.1)
        assert len(train) == 1
        assert len(val) == 0


# ---------------------------------------------------------------------------
# Integration: load_samples (convenience wrapper)
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
# expand_games
# ---------------------------------------------------------------------------


class TestExpandGames:
    def test_expand_single_game(self) -> None:
        games = [_simple_game()]
        samples = expand_games(games, MINI_2P)
        assert len(samples) == 2  # 1 state * 1 combo * 2 players

    def test_expand_empty(self) -> None:
        samples = expand_games([], MINI_2P)
        assert len(samples) == 0


# ---------------------------------------------------------------------------
# SimulationDataset (now takes games list)
# ---------------------------------------------------------------------------


class TestSimulationDataset:
    def test_len_and_getitem(self) -> None:
        games = [_simple_game([60.0, 40.0])]
        ds = SimulationDataset(games, MINI_2P)

        assert len(ds) == 2

        token_ids, target = ds[0]
        assert token_ids.shape == (MINI_2P.state_token_size,)
        assert token_ids.dtype == torch.int32
        assert target.dtype == torch.long
        assert 0 <= target.item() < 128

    def test_compatible_with_dataloader(self) -> None:
        games = [_simple_game()]
        ds = SimulationDataset(games, MINI_2P)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)

        batch_tokens, batch_targets = next(iter(loader))
        assert batch_tokens.shape == (2, MINI_2P.state_token_size)
        assert batch_targets.shape == (2,)

    def test_empty_games_list(self) -> None:
        ds = SimulationDataset([], MINI_2P)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Placement phase data loader tests
# ---------------------------------------------------------------------------

# Minimal placement state for mini 2p (4 sections: tiles|ports|vertices|edges).
# Uses the same format from topology-reference.md Part II.
MINI_PLACEMENT_STATE = (
    "w5lb3ls4lW3hd0nW4ho2l|gsgbgw"
    "|._._._._._._._._._._._._._._._._._._._._._._._._"
    "|______________________________"
)

# A second variant (as if produced by a board permutation) — swap first two
# tiles and vertex chars to get a different but still-valid state string.
MINI_PLACEMENT_STATE_PERM = (
    "b3lw5ls4lW3hd0nW4ho2l|gsgbgw"
    "|._._._._._._._._._._._._._._._._._._._._._._._._"
    "|______________________________"
)

# Two valid mini actions (from the placement action vocabulary).
MINI_ACTION_A = "0SE"
MINI_ACTION_B = "5N"

# Permuted action counterparts.
MINI_ACTION_A_PERM = "1SW"
MINI_ACTION_B_PERM = "6N"


def _make_placement_game(
    *,
    states: list[dict],
    n_players: int = 2,
    map_name: str = "mini",
) -> dict:
    """Create a minimal placement-phase JSONL game record."""
    return {
        "version": 1,
        "seed": 42,
        "map": map_name,
        "players": n_players,
        "winner": 1,
        "turns": 1,
        "exportType": "InitialPlacement",
        "board": {
            "serialized": "w5lb3ls4lW3hd0nW4ho2l|gsgbgw",
            "permutations": [],
        },
        "states": states,
    }


def _make_placement_state(
    *,
    state_str: str = MINI_PLACEMENT_STATE,
    state_perms: list[str] | None = None,
    actions: list[dict] | None = None,
    player_turn: int = 1,
) -> dict:
    """Create a minimal placement state entry."""
    return {
        "playerTurn": player_turn,
        "serializedState": state_str,
        "permutations": state_perms or [],
        "actions": actions or [],
    }


def _make_action(
    *,
    action: str,
    win_rate: float,
    permutations: list[str] | None = None,
) -> dict:
    """Create a minimal placement action entry."""
    return {
        "action": action,
        "winRate": win_rate,
        "rollouts": 50,
        "permutations": permutations or [],
    }


class TestExpandPlacementGames:
    """Tests for expand_placement_games."""

    def test_single_state_single_action_no_perms(self) -> None:
        """1 state * 1 variant * 1 action = 1 sample."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[_make_action(action=MINI_ACTION_A, win_rate=0.7)],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        assert len(samples) == 1

    def test_single_state_multiple_actions_no_perms(self) -> None:
        """1 state * 1 variant * 2 actions = 2 samples."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[
                        _make_action(action=MINI_ACTION_A, win_rate=0.7),
                        _make_action(action=MINI_ACTION_B, win_rate=0.3),
                    ],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        assert len(samples) == 2

    def test_with_one_permutation(self) -> None:
        """1 state * 2 variants * 2 actions = 4 samples."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    state_perms=[MINI_PLACEMENT_STATE_PERM],
                    actions=[
                        _make_action(
                            action=MINI_ACTION_A,
                            win_rate=0.7,
                            permutations=[MINI_ACTION_A_PERM],
                        ),
                        _make_action(
                            action=MINI_ACTION_B,
                            win_rate=0.3,
                            permutations=[MINI_ACTION_B_PERM],
                        ),
                    ],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        assert len(samples) == 4

    def test_multiple_states(self) -> None:
        """2 states * 1 variant * 1 action each = 2 samples."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[_make_action(action=MINI_ACTION_A, win_rate=0.6)],
                ),
                _make_placement_state(
                    actions=[_make_action(action=MINI_ACTION_B, win_rate=0.4)],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        assert len(samples) == 2

    def test_multiple_games(self) -> None:
        """2 games, each with 1 state and 1 action = 2 samples."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[_make_action(action=MINI_ACTION_A, win_rate=0.5)],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game, game], MINI_2P)
        assert len(samples) == 2

    def test_empty_games_list(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([], MINI_2P)
        assert len(samples) == 0

    def test_token_shape_and_dtype(self) -> None:
        """Token tensors should be placement_token_size + 1 (state + action)."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[_make_action(action=MINI_ACTION_A, win_rate=0.5)],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        token_ids, bucket = samples[0]
        expected_len = MINI_2P.placement_token_size + 1  # 105 + 1 = 106
        assert token_ids.shape == (expected_len,)
        assert token_ids.dtype == torch.int32
        assert 0 <= bucket < 128

    def test_bucket_reflects_win_rate(self) -> None:
        """Higher win rate should produce a higher bucket index."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[
                        _make_action(action=MINI_ACTION_A, win_rate=0.9),
                        _make_action(action=MINI_ACTION_B, win_rate=0.1),
                    ],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        _, bucket_high = samples[0]  # 0.9 win rate
        _, bucket_low = samples[1]  # 0.1 win rate
        assert bucket_high > bucket_low

    def test_state_not_prefixed_with_board(self) -> None:
        """Regression: serializedState already has all 4 sections.

        The data loader must NOT prepend the board string, otherwise
        the token count would be wrong (doubled tiles+ports section).
        """
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[_make_action(action=MINI_ACTION_A, win_rate=0.5)],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        token_ids, _ = samples[0]
        # If board were incorrectly prepended, length would be
        # 105 + 27 (compact board) + 1 (action) = 133, not 106.
        assert token_ids.shape == (MINI_2P.placement_token_size + 1,)

    def test_permutation_uses_permuted_action(self) -> None:
        """Variant 0 uses action["action"], variant 1+ uses action["permutations"]."""
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    state_perms=[MINI_PLACEMENT_STATE_PERM],
                    actions=[
                        _make_action(
                            action=MINI_ACTION_A,
                            win_rate=0.5,
                            permutations=[MINI_ACTION_A_PERM],
                        ),
                    ],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games
        from gimbur_nn.placement_tokenizer import PlacementTokenizer

        tok = PlacementTokenizer(MINI_2P)
        samples = expand_placement_games([game], MINI_2P, tokenizer=tok)
        assert len(samples) == 2

        # Last token of each sample is the action token.
        action_token_0 = samples[0][0][-1].item()
        action_token_1 = samples[1][0][-1].item()

        # Variant 0 should use MINI_ACTION_A, variant 1 should use MINI_ACTION_A_PERM.
        assert action_token_0 == tok.action_vocab[MINI_ACTION_A]
        assert action_token_1 == tok.action_vocab[MINI_ACTION_A_PERM]

    def test_no_player_rotation(self) -> None:
        """Placement expansion has no player rotation (unlike game-state expansion).

        1 state * 1 action * 1 variant = 1 sample (not multiplied by n_players).
        """
        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[_make_action(action=MINI_ACTION_A, win_rate=0.5)],
                ),
            ],
        )
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([game], MINI_2P)
        # Game state expansion would give 2 samples (2 players).
        # Placement expansion should give only 1.
        assert len(samples) == 1


class TestPlacementDataset:
    """Tests for PlacementDataset."""

    def test_len_and_getitem(self) -> None:
        from gimbur_nn.data_loader import PlacementDataset

        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[
                        _make_action(action=MINI_ACTION_A, win_rate=0.6),
                        _make_action(action=MINI_ACTION_B, win_rate=0.4),
                    ],
                ),
            ],
        )
        ds = PlacementDataset([game], MINI_2P)
        assert len(ds) == 2

        token_ids, target = ds[0]
        assert token_ids.shape == (MINI_2P.placement_token_size + 1,)
        assert token_ids.dtype == torch.int32
        assert target.dtype == torch.long
        assert 0 <= target.item() < 128

    def test_compatible_with_dataloader(self) -> None:
        from gimbur_nn.data_loader import PlacementDataset

        game = _make_placement_game(
            states=[
                _make_placement_state(
                    actions=[
                        _make_action(action=MINI_ACTION_A, win_rate=0.5),
                        _make_action(action=MINI_ACTION_B, win_rate=0.5),
                    ],
                ),
            ],
        )
        ds = PlacementDataset([game], MINI_2P)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)

        batch_tokens, batch_targets = next(iter(loader))
        seq_len = MINI_2P.placement_token_size + 1
        assert batch_tokens.shape == (2, seq_len)
        assert batch_targets.shape == (2,)

    def test_empty_games_list(self) -> None:
        from gimbur_nn.data_loader import PlacementDataset

        ds = PlacementDataset([], MINI_2P)
        assert len(ds) == 0
