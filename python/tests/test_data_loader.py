"""Tests for the data loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gimbur_nn.data_loader import (
    SimulationDataset,
    _normalize_wins,
    _scheduled_mcts_value_weight,
    _select_state_entries,
    _value_target,
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
    winner: int = 1,
) -> dict:
    """Create a minimal JSONL game record."""
    return {
        "version": 1,
        "seed": 42,
        "map": map_name,
        "players": n_players,
        "winner": winner,
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
    best_action_wins: list[float] | None = None,
    wins: list[float] | None = None,
    player_turn: int = 1,
    turn_number: int = 1,
    stage: str = "r",
    scores: list[float] | None = None,
) -> dict:
    """Create a minimal state entry within a game."""
    return {
        "playerTurn": player_turn,
        "turnNumber": turn_number,
        "stage": stage,
        "serializedState": state_str,
        "simulations": 100,
        "elapsedMs": 50,
        "winRate": 0.5,
        "wins": wins if wins is not None else best_action_wins,
        "scores": scores if scores is not None else [2.0, 2.0],
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
    "21010/00130|0/0|00000/00000|00000|0_|72111|_"
)


# ---------------------------------------------------------------------------
# Unit tests: helper functions
# ---------------------------------------------------------------------------


class TestNormalizeWins:
    def test_normalizes_counts(self) -> None:
        torch.testing.assert_close(_normalize_wins([80.0, 20.0], 2), torch.tensor([0.8, 0.2]))

    @pytest.mark.parametrize("wins", [[], [0.0, 0.0], [1.0]])
    def test_invalid_evidence_returns_none(self, wins: list[float]) -> None:
        assert _normalize_wins(wins, 2) is None


class TestValueTarget:
    def test_blend_endpoints(self) -> None:
        torch.testing.assert_close(_value_target([80, 20], 2, 2, 1.0), torch.tensor([0.8, 0.2]))
        torch.testing.assert_close(_value_target([80, 20], 2, 2, 0.0), torch.tensor([0.0, 1.0]))

    def test_blends_mcts_and_terminal_targets(self) -> None:
        torch.testing.assert_close(_value_target([80, 20], 2, 2, 0.5), torch.tensor([0.4, 0.6]))

    def test_uses_only_available_target(self) -> None:
        torch.testing.assert_close(_value_target([80, 20], 0, 2, 0.5), torch.tensor([0.8, 0.2]))
        torch.testing.assert_close(_value_target([], 2, 2, 0.5), torch.tensor([0.0, 1.0]))

    def test_returns_none_without_valid_evidence(self) -> None:
        assert _value_target([], 0, 2, 0.5) is None


class TestScheduledMctsValueWeight:
    def test_exact_endpoints_and_midpoint(self) -> None:
        assert _scheduled_mcts_value_weight(0, 10, 0.9, 0.1) == pytest.approx(0.9)
        assert _scheduled_mcts_value_weight(5, 10, 0.9, 0.1) == pytest.approx(0.5)
        assert _scheduled_mcts_value_weight(10, 10, 0.9, 0.1) == pytest.approx(0.1)

    def test_unfinished_or_invalid_total_turns_clamps_progress(self) -> None:
        assert _scheduled_mcts_value_weight(5, 0, 0.9, 0.1) == pytest.approx(0.1)
        assert _scheduled_mcts_value_weight(15, 10, 0.9, 0.1) == pytest.approx(0.1)


class TestVictoryPointStateSampling:
    @staticmethod
    def _game(seed: int, totals: list[int]) -> dict:
        return {
            "seed": seed,
            "states": [
                {
                    "id": f"{seed}-{index}",
                    "turnNumber": index + 2,
                    "stage": "b",
                    "scores": [total - 1, 1],
                }
                for index, total in enumerate(totals)
            ],
        }

    @staticmethod
    def _flatten(selected: list[list[dict]]) -> list[dict]:
        return [state for states in selected for state in states]

    def test_median_cap_plus_ten_percent_uses_all_games_and_retains_tails(self) -> None:
        games = [self._game(10, [4] * 7 + [5] * 4), self._game(20, [4] * 3 + [9])]

        selected = self._flatten(_select_state_entries(games, "median", 0.10))

        # Bucket sizes are 10, 4, 1: ceil(median 4 * 1.10) = 5.
        assert sum(sum(state["scores"]) == 4 for state in selected) == 5
        assert sum(sum(state["scores"]) == 5 for state in selected) == 4
        assert sum(sum(state["scores"]) == 9 for state in selected) == 1

    def test_average_option_sets_cap_from_mean_bucket_size(self) -> None:
        games = [self._game(10, [4] * 8), self._game(20, [5] * 2 + [9] * 2)]

        selected = self._flatten(_select_state_entries(games, "average", 0.0))

        # Bucket sizes are 8, 2, 2: mean 4.
        assert sum(sum(state["scores"]) == 4 for state in selected) == 4
        assert len(selected) == 8

    def test_oversized_bucket_sampling_is_deterministic_and_preserves_order(self) -> None:
        games = [self._game(10, [4] * 8 + [5] * 2 + [9] * 2)]

        first = _select_state_entries(games, "median", 0.0)
        second = _select_state_entries(games, "median", 0.0)

        assert first == second
        ids = [state["id"] for state in first[0]]
        assert ids == sorted(ids, key=lambda value: int(value.split("-")[1]))
        assert len(ids) == len(set(ids))

    def test_groups_by_total_vp_not_maximum_player_score(self) -> None:
        game = self._game(10, [4, 4, 4, 5, 6])
        game["states"][0]["scores"] = [3, 1]
        game["states"][1]["scores"] = [2, 2]

        selected = self._flatten(_select_state_entries([game], "median", 0.0))

        assert sum(sum(state["scores"]) == 4 for state in selected) == 1

    def test_always_retains_post_placement_and_final_roots_above_cap(self) -> None:
        game = self._game(10, [4] * 8 + [5] + [9] + [4])
        game["states"][0].update(id="post-placement", turnNumber=1, stage="r")
        game["states"][-1]["id"] = "same-bucket-final"

        selected = self._flatten(_select_state_entries([game], "median", 0.0))

        selected_ids = [state["id"] for state in selected]
        assert "post-placement" in selected_ids
        assert "same-bucket-final" in selected_ids
        assert sum(sum(state["scores"]) == 4 for state in selected) == 2

    def test_player_score_permutation_does_not_change_bucket_or_selection(self) -> None:
        games = [self._game(10, [4] * 8 + [5] * 2 + [9] * 2)]
        rotated = [self._game(10, [4] * 8 + [5] * 2 + [9] * 2)]
        for state in rotated[0]["states"]:
            state["scores"].reverse()

        original_ids = [
            state["id"] for state in self._flatten(_select_state_entries(games, "median", 0.0))
        ]
        rotated_ids = [
            state["id"] for state in self._flatten(_select_state_entries(rotated, "median", 0.0))
        ]

        assert original_ids == rotated_ids

    def test_legacy_game_without_scores_is_excluded_from_state_replay(self) -> None:
        legacy = self._game(10, [4, 5])
        for state in legacy["states"]:
            state.pop("scores")
        current = self._game(20, [4, 5])

        selected = _select_state_entries([legacy, current], "median", 0.10)

        assert selected[0] == []
        assert selected[1]


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
        for token_ids, target in samples:
            assert token_ids.shape == (MINI_2P.state_token_size,)
            torch.testing.assert_close(target.sum(), torch.tensor(1.0))

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

    def test_player_rotation_rotates_target_vector(self, tmp_path: Path) -> None:
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
        torch.testing.assert_close(samples[0][1], torch.tensor([0.918, 0.082]))
        torch.testing.assert_close(samples[1][1], torch.tensor([0.082, 0.918]))

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
    def test_resolved_state_uses_exact_distribution_without_terminal_blend(self) -> None:
        game = _simple_game([25.0, 75.0])
        game["winner"] = 1
        game["states"][0]["reachedTerminal"] = True

        samples = expand_games([game], MINI_2P, mcts_value_weight_start=0.1)

        torch.testing.assert_close(samples[0][1], torch.tensor([0.25, 0.75]))

    def test_uses_root_mcts_win_probability(self) -> None:
        game = _make_game(
            board=MINI_BOARD,
            board_perms=[],
            states=[
                _make_state(
                    state_str=MINI_STATE_ONLY,
                    state_perms=[],
                    wins=[90.0, 10.0],
                )
            ],
            n_players=2,
        )

        samples = expand_games([game], MINI_2P)

        torch.testing.assert_close(samples[0][1], torch.tensor([0.918, 0.082]))
        torch.testing.assert_close(samples[1][1], torch.tensor([0.082, 0.918]))

    def test_geometric_symmetry_keeps_player_target(self) -> None:
        game = _simple_game([70.0, 30.0])
        game["board"]["permutations"] = [MINI_BOARD]
        game["states"][0]["permutations"] = [MINI_STATE_ONLY]

        samples = expand_games([game], MINI_2P)

        torch.testing.assert_close(samples[0][1], samples[2][1])
        torch.testing.assert_close(samples[1][1], samples[3][1])

    def test_keeps_mcts_probabilities_from_unfinished_games(self) -> None:
        game = _simple_game()
        game["winner"] = 0

        assert len(expand_games([game], MINI_2P)) == 2

    def test_skips_state_without_mcts_or_terminal_target(self) -> None:
        game = _simple_game()
        game["winner"] = 0
        game["states"][0]["wins"] = []

        assert expand_games([game], MINI_2P) == []

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
        assert target.dtype == torch.float32
        assert target.shape == (MINI_2P.player_count,)

    def test_compatible_with_dataloader(self) -> None:
        games = [_simple_game()]
        ds = SimulationDataset(games, MINI_2P)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)

        batch_tokens, batch_targets = next(iter(loader))
        assert batch_tokens.shape == (2, MINI_2P.state_token_size)
        assert batch_targets.shape == (2, MINI_2P.player_count)

    def test_empty_games_list(self) -> None:
        ds = SimulationDataset([], MINI_2P)
        assert len(ds) == 0


# ---------------------------------------------------------------------------
# Placement phase data loader tests
# ---------------------------------------------------------------------------

MINI_PLACEMENT_STATE = (
    "w5lb3ls4lW3hd0nW4ho2l|gsgbgw|a"
    "|._._._._._._._._._._._._._._._._._._._._._._._._"
    "|______________________________"
)
MINI_PLACEMENT_STATE_PERM = MINI_PLACEMENT_STATE
MINI_ROAD_STATE = MINI_PLACEMENT_STATE.replace("|a|._", "|e|p_", 1)
MINI_ROAD_STATE_PERM = MINI_ROAD_STATE


def _placement_game(
    *, visits: tuple[int, int] = (30, 70), stage: str = "a"
) -> dict:
    road = stage in ("e", "i")
    return {
        "winner": 1,
        "states": [
            {
                "playerTurn": 1,
                "stage": stage,
                "serializedState": MINI_ROAD_STATE if road else MINI_PLACEMENT_STATE,
                "permutations": [MINI_ROAD_STATE_PERM if road else MINI_PLACEMENT_STATE_PERM],
                "actions": [
                    {
                        "policyIndex": 0,
                        "winRate": 0.2,
                        "wins": [18.0, 2.0],
                        "visits": visits[0],
                        "permutations": [1],
                    },
                    {
                        "policyIndex": 5,
                        "winRate": 0.8,
                        "wins": [14.0, 56.0],
                        "visits": visits[1],
                        "permutations": [2 if road else 6],
                    },
                ],
            }
        ],
    }


class TestExpandPlacementGames:
    def test_value_target_rotates_acting_player_first(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        game = _placement_game()
        game["states"][0]["playerTurn"] = 2
        samples = expand_placement_games([game], MINI_2P, mcts_value_weight_start=1.0)

        torch.testing.assert_close(samples[0][1], torch.tensor([0.59, 0.41]))

    def test_placement_value_uses_start_weight(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([_placement_game()], MINI_2P, mcts_value_weight_start=1.0)

        torch.testing.assert_close(samples[0][1], torch.tensor([0.41, 0.59]))

    def test_emits_one_sample_per_state_permutation(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([_placement_game()], MINI_2P)

        assert len(samples) == 2
        assert samples[0][0].shape == (MINI_2P.placement_token_size,)
        torch.testing.assert_close(samples[0][1], torch.tensor([0.469, 0.531]))
        torch.testing.assert_close(samples[0][1], samples[1][1])

    def test_combined_targets_are_dense_and_symmetry_mapped(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games
        from gimbur_nn.placement_tokenizer import PlacementTokenizer

        tok = PlacementTokenizer(MINI_2P)
        samples = expand_placement_games(
            [_placement_game()], MINI_2P, tokenizer=tok, target="combined"
        )
        _, _, identity_policy, identity_mask = samples[0]
        _, _, perm_policy, perm_mask = samples[1]

        assert identity_policy[tok.vertex_action_index(0)] == pytest.approx(0.3)
        assert identity_policy[tok.vertex_action_index(5)] == pytest.approx(0.7)
        assert perm_policy[tok.vertex_action_index(1)] == pytest.approx(0.3)
        assert perm_policy[tok.vertex_action_index(6)] == pytest.approx(0.7)
        assert identity_mask.sum() == perm_mask.sum() == 2

    def test_road_targets_use_direction_indices(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        samples = expand_placement_games([_placement_game(stage="e")], MINI_2P, target="combined")
        identity_policy = samples[0][2]
        perm_policy = samples[1][2]

        assert identity_policy[0] == pytest.approx(0.3)
        assert identity_policy[5] == pytest.approx(0.7)
        assert perm_policy[1] == pytest.approx(0.3)
        assert perm_policy[2] == pytest.approx(0.7)

    def test_greedy_bootstrap_policy_is_one_hot_without_changing_value_target(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        game = _placement_game()
        game["states"][0]["actions"][0]["modelPrior"] = 1
        game["states"][0]["actions"][1]["modelPrior"] = 0
        samples = expand_placement_games(
            [game], MINI_2P, target="combined", mcts_value_weight_start=1.0
        )
        _, value_target, identity_policy, legal_mask = samples[0]
        _, _, perm_policy, _ = samples[1]

        torch.testing.assert_close(value_target, torch.tensor([0.41, 0.59]))
        assert identity_policy[0] == 1
        assert identity_policy[5] == 0
        assert perm_policy[1] == 1
        assert perm_policy[6] == 0
        assert legal_mask.sum() == 2

    def test_soft_model_prior_keeps_mcts_visit_target(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        game = _placement_game()
        game["states"][0]["actions"][0]["modelPrior"] = 0.8
        game["states"][0]["actions"][1]["modelPrior"] = 0.2

        policy = expand_placement_games([game], MINI_2P, target="combined")[0][2]

        assert policy[0] == pytest.approx(0.3)
        assert policy[5] == pytest.approx(0.7)

    def test_policy_temperature_one_preserves_exact_visit_shares(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games
        from gimbur_nn.placement_tokenizer import PlacementTokenizer

        tok = PlacementTokenizer(MINI_2P)
        default_policy = expand_placement_games(
            [_placement_game()], MINI_2P, tokenizer=tok, target="combined"
        )[0][2]
        temperature_one_policy = expand_placement_games(
            [_placement_game()],
            MINI_2P,
            tokenizer=tok,
            target="combined",
            policy_target_temperature=1.0,
        )[0][2]

        assert torch.equal(temperature_one_policy, default_policy)

    def test_policy_temperature_half_squares_and_normalizes(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games
        from gimbur_nn.placement_tokenizer import PlacementTokenizer

        tok = PlacementTokenizer(MINI_2P)
        policy = expand_placement_games(
            [_placement_game()],
            MINI_2P,
            tokenizer=tok,
            target="combined",
            policy_target_temperature=0.5,
        )[0][2]

        assert policy[tok.vertex_action_index(0)] == pytest.approx(0.3**2 / (0.3**2 + 0.7**2))
        assert policy[tok.vertex_action_index(5)] == pytest.approx(0.7**2 / (0.3**2 + 0.7**2))

    def test_policy_temperature_preserves_zero_shares_and_legal_mask(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games
        from gimbur_nn.placement_tokenizer import PlacementTokenizer

        tok = PlacementTokenizer(MINI_2P)
        _, _, policy, legal_mask = expand_placement_games(
            [_placement_game(visits=(0, 70))],
            MINI_2P,
            tokenizer=tok,
            target="combined",
            policy_target_temperature=0.1,
        )[0]

        zero_idx = tok.vertex_action_index(0)
        assert policy[zero_idx] == 0
        assert legal_mask[zero_idx]
        assert legal_mask.sum() == 2

    def test_extremely_small_policy_temperature_is_stable(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games
        from gimbur_nn.placement_tokenizer import PlacementTokenizer

        tok = PlacementTokenizer(MINI_2P)
        policy = expand_placement_games(
            [_placement_game()],
            MINI_2P,
            tokenizer=tok,
            target="combined",
            policy_target_temperature=1e-320,
        )[0][2]

        assert torch.isfinite(policy).all()
        assert policy.sum() == 1
        assert policy[tok.vertex_action_index(5)] == 1

    @pytest.mark.parametrize("temperature", [0.0, -0.5, float("nan")])
    def test_rejects_invalid_policy_temperature(self, temperature: float) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        with pytest.raises(ValueError, match="temperature must be greater than zero"):
            expand_placement_games(
                [_placement_game()],
                MINI_2P,
                target="combined",
                policy_target_temperature=temperature,
            )

    def test_combined_skips_zero_total_rollouts(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        assert (
            expand_placement_games([_placement_game(visits=(0, 0))], MINI_2P, target="combined")
            == []
        )

    def test_value_fallback_uses_terminal_winner(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        game = _placement_game()
        for action in game["states"][0]["actions"]:
            action.pop("wins")
        samples = expand_placement_games([game], MINI_2P)

        torch.testing.assert_close(samples[0][1], torch.tensor([1.0, 0.0]))

    def test_value_only_skips_without_mcts_or_terminal_target(self) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        game = _placement_game()
        game["winner"] = 0
        for action in game["states"][0]["actions"]:
            action.pop("wins")

        assert expand_placement_games([game], MINI_2P) == []

    @pytest.mark.parametrize(
        ("stage", "state", "message"),
        [
            ("x", MINI_PLACEMENT_STATE, "invalid placement stage"),
            ("e", MINI_PLACEMENT_STATE.replace("|a|", "|e|"), "exactly one pending"),
            ("a", MINI_ROAD_STATE.replace("|e|", "|a|"), "must not contain a pending"),
        ],
    )
    def test_rejects_malformed_stage_pending_marker(
        self, stage: str, state: str, message: str
    ) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        game = _placement_game()
        game["states"][0]["stage"] = stage
        game["states"][0]["serializedState"] = state
        with pytest.raises(ValueError, match=message):
            expand_placement_games([game], MINI_2P, target="combined")

    @pytest.mark.parametrize(("stage", "policy_index"), [("a", 24), ("e", 6)])
    def test_rejects_policy_index_outside_stage_vocabulary(
        self, stage: str, policy_index: int
    ) -> None:
        from gimbur_nn.data_loader import expand_placement_games

        game = _placement_game(stage=stage)
        game["states"][0]["actions"][0]["policyIndex"] = policy_index
        with pytest.raises(ValueError, match="policy index"):
            expand_placement_games([game], MINI_2P, target="combined")


class TestPlacementDataset:
    def test_combined_batches_dense_targets(self) -> None:
        from gimbur_nn.data_loader import PlacementDataset

        loader = torch.utils.data.DataLoader(
            PlacementDataset([_placement_game()], MINI_2P, target="combined"), batch_size=2
        )
        tokens, values, policies, masks = next(iter(loader))

        assert tokens.shape == (2, MINI_2P.placement_token_size)
        assert values.shape == (2, MINI_2P.player_count)
        assert policies.shape == masks.shape == (2, MINI_2P.placement_policy_size)


def test_combined_game_feeds_placement_and_state_datasets() -> None:
    combined = _simple_game()
    combined["placementStates"] = _placement_game()["states"]

    state_dataset = SimulationDataset([combined], MINI_2P)
    from gimbur_nn.data_loader import PlacementDataset

    placement_dataset = PlacementDataset([combined], MINI_2P, target="combined")

    assert len(state_dataset) == 2
    assert len(placement_dataset) == 2
