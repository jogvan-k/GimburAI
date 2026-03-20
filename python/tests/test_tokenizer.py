"""Tests for the tokenizer against the examples in docs/state-action-serialization.md."""

from __future__ import annotations

import pytest
import torch

from gimbur_nn.game_config import (
    MINI_2P,
    SMALL_2P,
    SMALL_3P,
    STANDARD_2P,
    STANDARD_3P,
    STANDARD_4P,
)
from gimbur_nn.state_tokenizer import StateTokenizer
from gimbur_nn.placement_tokenizer import PlacementTokenizer

# ---------------------------------------------------------------------------
# Example states copied verbatim from docs/state-action-serialization.md
# ---------------------------------------------------------------------------

MINI_STATE = (
    "w5lb3ls4lW3hd0nW4ho2l"
    "|gsgbgw"
    "|4"
    "|-t"
    "|__"
    "|._._._._._._v-._._._._._._._v+._._._._._._._._._"
    "|_____-_______+________________"
    "|21010/00130"
    "|0/0"
    "|00000/00000"
)

SMALL_STATE = (
    "W2lb3ls4lw3hb2hw1ho5lW4hs5hd0n"
    "|gwWgsb"
    "|9"
    "|+r"
    "|__"
    "|v-v+v+._._._._v-._._._._._._._._._._._._._._._._._._._._._._._._"
    "|-_+_+_-__________________________________"
    "|10010/00100"
    "|0/0"
    "|00000/00000"
)

STANDARD_STATE = (
    "w4lo1lb5lW2lw5hs3hW4ho1hs2hw3lb5hs3hW4ho3ls4lb5lw2lW2hd0n"
    "|ggwgbsWog"
    "|5"
    "|+t"
    "|_-"
    "|._._._._._._._._v-._._._._._._._._._c+._._._._._v*._._._._._._v*._._._v-._._._._._._._._v+._._._._._._._._._"
    "|______-_________-________+______+_**_____-*_____*-_____+_____+__________"
    "|31201/02143/10320"
    "|2/0/1"
    "|10000/01010/00100"
)

# fmt: off

# Expected token IDs for each example, computed by mapping each character
# of the compact form (separators stripped) through the vocabulary dictionary.

MINI_EXPECTED = [
    1, 12, 13, 2, 10, 13, 3, 11, 13, 4, 10, 14, 0, 7, 15, 4, 11, 14, 5, 9, 13,
    6, 3, 6, 2, 6, 1,
    11,
    20, 29,
    19, 19,
    16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 20, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 21, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19,
    19, 19, 19, 19, 19, 20, 19, 19, 19, 19, 19, 19, 19, 21, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    9, 8, 7, 8, 7, 7, 7, 8, 10, 7,
    7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
]

SMALL_EXPECTED = [
    4, 9, 13, 2, 10, 13, 3, 11, 13, 1, 10, 14, 2, 9, 14, 1, 8, 14, 5, 12, 13, 4, 11, 14, 3, 12, 14, 0, 7, 15,
    6, 1, 4, 6, 3, 2,
    33,
    21, 26,
    19, 19,
    17, 20, 17, 21, 17, 21, 16, 19, 16, 19, 16, 19, 16, 19, 17, 20, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19,
    20, 19, 21, 19, 21, 19, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    8, 7, 7, 8, 7, 7, 7, 8, 7, 7,
    7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
]

STANDARD_EXPECTED = [
    1, 11, 13, 5, 8, 13, 2, 12, 13, 4, 9, 13, 1, 12, 14, 3, 10, 14, 4, 11, 14, 5, 8, 14, 3, 9, 14, 1, 10, 13, 2, 12, 14, 3, 10, 14, 4, 11, 14, 5, 10, 13, 3, 11, 13, 2, 12, 13, 1, 9, 13, 4, 9, 14, 0, 7, 15,
    6, 6, 1, 6, 2, 3, 4, 5, 6,
    12,
    21, 30,
    19, 20,
    16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 20, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 18, 21, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 22, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 22, 16, 19, 16, 19, 16, 19, 17, 20, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 21, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19,
    19, 19, 19, 19, 19, 19, 20, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 19, 19, 19, 19, 19, 19, 19, 19, 21, 19, 19, 19, 19, 19, 19, 21, 19, 22, 22, 19, 19, 19, 19, 19, 20, 22, 19, 19, 19, 19, 19, 22, 20, 19, 19, 19, 19, 19, 21, 19, 19, 19, 19, 19, 21, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    10, 8, 9, 7, 8, 7, 9, 8, 11, 10, 8, 7, 10, 9, 7,
    9, 7, 8,
    8, 7, 7, 7, 7, 7, 8, 7, 8, 7, 7, 7, 8, 7, 7,
]

# fmt: on


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class TestVocab:
    def test_game_state_vocab_sizes(self) -> None:
        """Vocab size varies by player count: 2p=44, 3p=45, 4p=46."""
        assert StateTokenizer(MINI_2P).vocab_size == 44
        assert StateTokenizer(SMALL_2P).vocab_size == 44
        assert StateTokenizer(SMALL_3P).vocab_size == 45
        assert StateTokenizer(STANDARD_2P).vocab_size == 44
        assert StateTokenizer(STANDARD_3P).vocab_size == 45
        assert StateTokenizer(STANDARD_4P).vocab_size == 46

    def test_placement_state_vocab_sizes(self) -> None:
        """Placement state vocab (chars only): 2p=21, 3p=22, 4p=23."""
        assert PlacementTokenizer(MINI_2P).state_vocab_size == 21
        assert PlacementTokenizer(SMALL_2P).state_vocab_size == 21
        assert PlacementTokenizer(SMALL_3P).state_vocab_size == 22
        assert PlacementTokenizer(STANDARD_2P).state_vocab_size == 21
        assert PlacementTokenizer(STANDARD_3P).state_vocab_size == 22
        assert PlacementTokenizer(STANDARD_4P).state_vocab_size == 23

    def test_placement_combined_vocab_sizes(self) -> None:
        """Combined vocab (state chars + actions): S + A."""
        assert PlacementTokenizer(MINI_2P).vocab_size == 81
        assert PlacementTokenizer(SMALL_2P).vocab_size == 103
        assert PlacementTokenizer(SMALL_3P).vocab_size == 104
        assert PlacementTokenizer(STANDARD_2P).vocab_size == 165
        assert PlacementTokenizer(STANDARD_3P).vocab_size == 166
        assert PlacementTokenizer(STANDARD_4P).vocab_size == 167

    def test_placement_vocab_matches_config(self) -> None:
        """PlacementTokenizer.vocab_size matches GameConfig.placement_vocab_size."""
        for cfg in (MINI_2P, SMALL_2P, SMALL_3P, STANDARD_2P, STANDARD_3P, STANDARD_4P):
            tok = PlacementTokenizer(cfg)
            assert tok.vocab_size == cfg.placement_vocab_size

    def test_no_duplicate_chars(self) -> None:
        tok = StateTokenizer(MINI_2P)
        assert len(set(tok.vocab_chars)) == len(tok.vocab_chars)

    def test_dict_matches_chars(self) -> None:
        tok = StateTokenizer(MINI_2P)
        assert len(tok.vocab) == len(tok.vocab_chars)
        for idx, ch in enumerate(tok.vocab_chars):
            assert tok.vocab[ch] == idx


# ---------------------------------------------------------------------------
# Full-tensor comparison for each doc example
# ---------------------------------------------------------------------------


class TestMiniMap:
    def test_full_tensor(self) -> None:
        tok = StateTokenizer(MINI_2P)
        t = tok.tokenize(MINI_STATE)
        assert t.shape == (132,)
        assert t.dtype == torch.int32
        assert t.tolist() == MINI_EXPECTED


class TestSmallMap:
    def test_full_tensor(self) -> None:
        tok = StateTokenizer(SMALL_2P)
        t = tok.tokenize(SMALL_STATE)
        assert t.shape == (168,)
        assert t.dtype == torch.int32
        assert t.tolist() == SMALL_EXPECTED


class TestStandardMap:
    def test_full_tensor(self) -> None:
        tok = StateTokenizer(STANDARD_3P)
        t = tok.tokenize(STANDARD_STATE)
        assert t.shape == (284,)
        assert t.dtype == torch.int32
        assert t.tolist() == STANDARD_EXPECTED


# ---------------------------------------------------------------------------
# Compact form (no separators) produces same result
# ---------------------------------------------------------------------------


class TestCompactForm:
    def test_compact_equals_human_readable(self) -> None:
        """Stripping | and / from human-readable form should give same tokens."""
        tok = StateTokenizer(MINI_2P)
        human = tok.tokenize(MINI_STATE)
        compact_str = MINI_STATE.replace("|", "").replace("/", "")
        compact = tok.tokenize(compact_str)
        assert torch.equal(human, compact)


# ---------------------------------------------------------------------------
# Batch tokenization
# ---------------------------------------------------------------------------


class TestBatch:
    def test_same_map_batch(self) -> None:
        tok = StateTokenizer(MINI_2P)
        t = tok.tokenize_batch([MINI_STATE, MINI_STATE])
        assert t.shape == (2, 132)
        assert t.dtype == torch.int32

    def test_batch_matches_single(self) -> None:
        tok = StateTokenizer(SMALL_2P)
        single = tok.tokenize(SMALL_STATE)
        batch = tok.tokenize_batch([SMALL_STATE])
        assert torch.equal(batch[0], single)

    def test_mismatched_lengths_raises(self) -> None:
        tok = StateTokenizer(MINI_2P)
        with pytest.raises((ValueError, KeyError)):
            tok.tokenize_batch([MINI_STATE, SMALL_STATE])

    def test_empty_batch(self) -> None:
        tok = StateTokenizer(MINI_2P)
        t = tok.tokenize_batch([])
        assert t.shape == (0, 0)
        assert t.dtype == torch.int32


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_character_raises(self) -> None:
        tok = StateTokenizer(MINI_2P)
        with pytest.raises(KeyError):
            tok.tokenize("w5l#invalid")


# ---------------------------------------------------------------------------
# Compact-form helpers for rotation tests
# ---------------------------------------------------------------------------

_STRIP = str.maketrans("", "", "|/")

MINI_COMPACT = MINI_STATE.translate(_STRIP)
SMALL_COMPACT = SMALL_STATE.translate(_STRIP)
STANDARD_COMPACT = STANDARD_STATE.translate(_STRIP)


# ---------------------------------------------------------------------------
# Player rotation
# ---------------------------------------------------------------------------


class TestRotatePlayerState:
    """Tests for StateTokenizer.rotate_player_state()."""

    # -- Identity rotation (target=1 returns input unchanged) ---------------

    def test_identity_mini(self) -> None:
        tok = StateTokenizer(MINI_2P)
        result = tok.rotate_player_state(MINI_COMPACT, 1)
        assert result == MINI_COMPACT

    def test_identity_small(self) -> None:
        tok = StateTokenizer(SMALL_2P)
        result = tok.rotate_player_state(SMALL_COMPACT, 1)
        assert result == SMALL_COMPACT

    def test_identity_standard(self) -> None:
        tok = StateTokenizer(STANDARD_3P)
        result = tok.rotate_player_state(STANDARD_COMPACT, 1)
        assert result == STANDARD_COMPACT

    # -- Length preserved ---------------------------------------------------

    def test_length_preserved_mini(self) -> None:
        tok = StateTokenizer(MINI_2P)
        result = tok.rotate_player_state(MINI_COMPACT, 2)
        assert len(result) == len(MINI_COMPACT)

    def test_length_preserved_standard(self) -> None:
        tok = StateTokenizer(STANDARD_3P)
        for target in (2, 3):
            result = tok.rotate_player_state(STANDARD_COMPACT, target)
            assert len(result) == len(STANDARD_COMPACT)

    # -- Board sections unchanged (tiles, ports, robber) --------------------

    def test_board_unchanged_mini(self) -> None:
        """Tiles, ports, robber must be identical after rotation."""
        cfg = MINI_2P
        board_end = 3 * cfg.tile_count + cfg.port_count + 1  # tiles + ports + robber
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(MINI_COMPACT, 2)
        assert result[:board_end] == MINI_COMPACT[:board_end]

    def test_board_unchanged_standard(self) -> None:
        cfg = STANDARD_3P
        board_end = 3 * cfg.tile_count + cfg.port_count + 1
        tok = StateTokenizer(cfg)
        for target in (2, 3):
            result = tok.rotate_player_state(STANDARD_COMPACT, target)
            assert result[:board_end] == STANDARD_COMPACT[:board_end]

    # -- Double rotation is identity (2-player) -----------------------------

    def test_double_rotation_identity_2p(self) -> None:
        """Rotating by player 2 twice should return the original (N=2)."""
        tok = StateTokenizer(MINI_2P)
        once = tok.rotate_player_state(MINI_COMPACT, 2)
        twice = tok.rotate_player_state(once, 2)
        assert twice == MINI_COMPACT

    # -- Full-cycle rotation is identity (3-player) -------------------------

    def test_full_cycle_identity_3p(self) -> None:
        """Rotating p2 then p2 then p2 (3 times) returns original (N=3)."""
        tok = StateTokenizer(STANDARD_3P)
        state = STANDARD_COMPACT
        for _ in range(3):
            state = tok.rotate_player_state(state, 2)
        assert state == STANDARD_COMPACT

    # -- Mini 2p rotation example from docs ---------------------------------

    def test_mini_2p_rotation_matches_doc(self) -> None:
        """Verify the rotation example from docs/state-action-serialization.md."""
        original_hr = (
            "w5lb3ls4lW3hd0nW4ho2l|gsgbgw|4|-t|__|"
            "._._._._._._v-._._._._._._._v+._._._._._._._._._|"
            "_____-_______+________________|"
            "21010/00130|0/0|00000/00000"
        )
        expected_hr = (
            "w5lb3ls4lW3hd0nW4ho2l|gsgbgw|4|+t|__|"
            "._._._._._._v+._._._._._._._v-._._._._._._._._._|"
            "_____+_______-________________|"
            "00130/21010|0/0|00000/00000"
        )
        original = original_hr.translate(_STRIP)
        expected = expected_hr.translate(_STRIP)
        tok = StateTokenizer(MINI_2P)
        result = tok.rotate_player_state(original, 2)
        assert result == expected

    # -- Current player is remapped -----------------------------------------

    def test_current_player_remapped_mini(self) -> None:
        """Mini example: current player '-' (P1) should become '+' (P2)."""
        cfg = MINI_2P
        pos = 3 * cfg.tile_count + cfg.port_count + 1  # offset of currentPlayer
        assert MINI_COMPACT[pos] == "-"
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(MINI_COMPACT, 2)
        assert result[pos] == "+"

    def test_current_player_remapped_standard(self) -> None:
        """Standard example: current player '+' (P2) rotated for P3."""
        cfg = STANDARD_3P
        pos = 3 * cfg.tile_count + cfg.port_count + 1
        assert STANDARD_COMPACT[pos] == "+"
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(STANDARD_COMPACT, 3)
        # P2 with R=2, N=3: ((2-1-2) mod 3)+1 = ((-1) mod 3)+1 = 2+1 = 3 -> '*'
        assert result[pos] == "*"

    # -- Longest road / largest army remapped --------------------------------

    def test_awards_remapped_standard(self) -> None:
        """Standard example: LR=none, LA=player1. Rotate for player 2."""
        cfg = STANDARD_3P
        lr_pos = 3 * cfg.tile_count + cfg.port_count + 3
        la_pos = lr_pos + 1
        assert STANDARD_COMPACT[lr_pos] == "_"  # no longest road
        assert STANDARD_COMPACT[la_pos] == "-"  # player 1
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(STANDARD_COMPACT, 2)
        assert result[lr_pos] == "_"  # still none
        # P1 with R=1, N=3: ((1-1-1) mod 3)+1 = ((-1) mod 3)+1 = 2+1 = 3 -> '*'
        assert result[la_pos] == "*"

    # -- Resources reordered ------------------------------------------------

    def test_resources_reordered_mini(self) -> None:
        """Mini 2p: resources '21010''00130' should swap to '00130''21010'."""
        cfg = MINI_2P
        res_start = 3 * cfg.tile_count + cfg.port_count + 5 + 2 * cfg.vertex_count + cfg.edge_count
        original_res = MINI_COMPACT[res_start : res_start + 10]
        assert original_res == "2101000130"
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(MINI_COMPACT, 2)
        rotated_res = result[res_start : res_start + 10]
        assert rotated_res == "0013021010"

    def test_resources_reordered_standard(self) -> None:
        """Standard 3p: 3 blocks of 5 should rotate for target=2."""
        cfg = STANDARD_3P
        res_start = 3 * cfg.tile_count + cfg.port_count + 5 + 2 * cfg.vertex_count + cfg.edge_count
        original_res = STANDARD_COMPACT[res_start : res_start + 15]
        assert original_res == "312010214310320"
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(STANDARD_COMPACT, 2)
        rotated_res = result[res_start : res_start + 15]
        # Rotation R=1: blocks shift by 1 -> P2,P3,P1
        assert rotated_res == "021431032031201"

    # -- Knights reordered --------------------------------------------------

    def test_knights_reordered_standard(self) -> None:
        """Standard 3p: knights '2','0','1' -> for target=2: '0','1','2'."""
        cfg = STANDARD_3P
        kn_start = (
            3 * cfg.tile_count
            + cfg.port_count
            + 5
            + 2 * cfg.vertex_count
            + cfg.edge_count
            + 5 * cfg.player_count
        )
        original_kn = STANDARD_COMPACT[kn_start : kn_start + 3]
        assert original_kn == "201"
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(STANDARD_COMPACT, 2)
        rotated_kn = result[kn_start : kn_start + 3]
        assert rotated_kn == "012"

    # -- Dev cards reordered ------------------------------------------------

    def test_dev_cards_reordered_standard(self) -> None:
        """Standard 3p: dev blocks reorder for target=3."""
        cfg = STANDARD_3P
        dev_start = (
            3 * cfg.tile_count
            + cfg.port_count
            + 5
            + 2 * cfg.vertex_count
            + cfg.edge_count
            + 5 * cfg.player_count
            + cfg.player_count
        )
        original_dev = STANDARD_COMPACT[dev_start : dev_start + 15]
        assert original_dev == "100000101000100"
        tok = StateTokenizer(cfg)
        result = tok.rotate_player_state(STANDARD_COMPACT, 3)
        rotated_dev = result[dev_start : dev_start + 15]
        # Rotation R=2: blocks shift by 2 -> P3,P1,P2
        assert rotated_dev == "001001000001010"

    # -- Error handling -----------------------------------------------------

    def test_invalid_target_player_raises(self) -> None:
        tok = StateTokenizer(MINI_2P)
        with pytest.raises(ValueError, match="target_player"):
            tok.rotate_player_state(MINI_COMPACT, 0)

    def test_target_too_high_raises(self) -> None:
        tok = StateTokenizer(MINI_2P)
        with pytest.raises(ValueError, match="target_player"):
            tok.rotate_player_state(MINI_COMPACT, 3)

    def test_wrong_length_raises(self) -> None:
        tok = StateTokenizer(MINI_2P)
        with pytest.raises(ValueError, match="Expected"):
            tok.rotate_player_state(MINI_COMPACT + "x", 2)


# ---------------------------------------------------------------------------
# PlacementTokenizer
# ---------------------------------------------------------------------------


class TestPlacementTokenizer:
    """Tests for the unified PlacementTokenizer (state + action)."""

    # -- Mini empty-board placement state (from spec Part II) --
    MINI_PLACEMENT_STATE = (
        "w5lb3ls4lW3hd0nW4ho2l|gsgbgw"
        "|._._._._._._._._._._._._._._._._._._._._._._._._"
        "|______________________________"
    )

    # -- State tokenization ------------------------------------------------

    def test_tokenize_state_mini_empty_board(self) -> None:
        """tokenize_state produces correct shape and dtype."""
        tok = PlacementTokenizer(MINI_2P)
        t = tok.tokenize_state(self.MINI_PLACEMENT_STATE)
        assert t.shape == (MINI_2P.placement_token_size,)
        assert t.dtype == torch.int32

    def test_tokenize_state_strips_separators(self) -> None:
        """Separators | and / are stripped before tokenization."""
        tok = PlacementTokenizer(MINI_2P)
        human = tok.tokenize_state(self.MINI_PLACEMENT_STATE)
        compact_str = self.MINI_PLACEMENT_STATE.replace("|", "").replace("/", "")
        compact = tok.tokenize_state(compact_str)
        assert torch.equal(human, compact)

    def test_tokenize_state_unknown_char_raises(self) -> None:
        tok = PlacementTokenizer(MINI_2P)
        with pytest.raises(KeyError):
            tok.tokenize_state("w5l#bad")

    # -- Placement token sizes from config ---------------------------------

    def test_placement_token_size_mini(self) -> None:
        cfg = MINI_2P
        expected = 3 * cfg.tile_count + cfg.port_count + 2 * cfg.vertex_count + cfg.edge_count
        assert cfg.placement_token_size == expected
        assert cfg.placement_token_size == 105

    def test_placement_token_size_small(self) -> None:
        cfg = SMALL_2P
        expected = 3 * cfg.tile_count + cfg.port_count + 2 * cfg.vertex_count + cfg.edge_count
        assert cfg.placement_token_size == expected
        assert cfg.placement_token_size == 141

    def test_placement_token_size_standard(self) -> None:
        cfg = STANDARD_3P
        expected = 3 * cfg.tile_count + cfg.port_count + 2 * cfg.vertex_count + cfg.edge_count
        assert cfg.placement_token_size == expected
        assert cfg.placement_token_size == 246

    # -- Action tokenization -----------------------------------------------

    def test_tokenize_action_mini(self) -> None:
        """First action maps to state_vocab_size, last to vocab_size - 1."""
        tok = PlacementTokenizer(MINI_2P)
        assert tok.tokenize_action("0SE") == tok.state_vocab_size
        assert tok.tokenize_action("23NW") == tok.vocab_size - 1

    def test_tokenize_action_standard(self) -> None:
        tok = PlacementTokenizer(STANDARD_4P)
        assert tok.tokenize_action("0SE") == tok.state_vocab_size
        assert tok.tokenize_action("53NW") == tok.vocab_size - 1

    def test_tokenize_action_unknown_raises(self) -> None:
        tok = PlacementTokenizer(MINI_2P)
        with pytest.raises(KeyError):
            tok.tokenize_action("99ZZ")

    # -- Decode action -----------------------------------------------------

    def test_decode_action_roundtrip(self) -> None:
        """tokenize_action -> decode_action is identity."""
        tok = PlacementTokenizer(MINI_2P)
        for action in ("0SE", "5N", "23NW"):
            token_id = tok.tokenize_action(action)
            assert tok.decode_action(token_id) == action

    def test_decode_action_all_mini(self) -> None:
        """Every action in mini map round-trips correctly."""
        tok = PlacementTokenizer(MINI_2P)
        for action in tok.actions:
            assert tok.decode_action(tok.tokenize_action(action)) == action

    # -- Combined state + action -------------------------------------------

    def test_tokenize_state_action_shape(self) -> None:
        """tokenize_state_action returns placement_token_size + 1 tokens."""
        tok = PlacementTokenizer(MINI_2P)
        t = tok.tokenize_state_action(self.MINI_PLACEMENT_STATE, "0SE")
        assert t.shape == (MINI_2P.placement_token_size + 1,)
        assert t.dtype == torch.int32

    def test_tokenize_state_action_last_token(self) -> None:
        """Last token in combined sequence is the action token."""
        tok = PlacementTokenizer(MINI_2P)
        t = tok.tokenize_state_action(self.MINI_PLACEMENT_STATE, "5N")
        action_id = tok.tokenize_action("5N")
        assert t[-1].item() == action_id

    def test_tokenize_state_action_prefix_matches_state(self) -> None:
        """All tokens except the last match tokenize_state output."""
        tok = PlacementTokenizer(MINI_2P)
        state_tokens = tok.tokenize_state(self.MINI_PLACEMENT_STATE)
        combined = tok.tokenize_state_action(self.MINI_PLACEMENT_STATE, "0SE")
        assert torch.equal(combined[:-1], state_tokens)

    # -- Batch state + actions ---------------------------------------------

    def test_tokenize_state_actions_shape(self) -> None:
        """tokenize_state_actions returns (N, placement_token_size + 1)."""
        tok = PlacementTokenizer(MINI_2P)
        actions = ["0SE", "5N", "23NW"]
        t = tok.tokenize_state_actions(self.MINI_PLACEMENT_STATE, actions)
        assert t.shape == (3, MINI_2P.placement_token_size + 1)
        assert t.dtype == torch.int32

    def test_tokenize_state_actions_matches_single(self) -> None:
        """Each row matches calling tokenize_state_action individually."""
        tok = PlacementTokenizer(MINI_2P)
        actions = ["0SE", "5N", "23NW"]
        batch = tok.tokenize_state_actions(self.MINI_PLACEMENT_STATE, actions)
        for i, action in enumerate(actions):
            single = tok.tokenize_state_action(self.MINI_PLACEMENT_STATE, action)
            assert torch.equal(batch[i], single)

    def test_tokenize_state_actions_empty(self) -> None:
        """Empty action list returns shape (0, placement_token_size + 1)."""
        tok = PlacementTokenizer(MINI_2P)
        t = tok.tokenize_state_actions(self.MINI_PLACEMENT_STATE, [])
        assert t.shape == (0, MINI_2P.placement_token_size + 1)

    # -- Action vocab sizes ------------------------------------------------

    def test_action_vocab_size_mini(self) -> None:
        assert PlacementTokenizer(MINI_2P).action_vocab_size == 60

    def test_action_vocab_size_small(self) -> None:
        assert PlacementTokenizer(SMALL_2P).action_vocab_size == 82

    def test_action_vocab_size_standard(self) -> None:
        assert PlacementTokenizer(STANDARD_2P).action_vocab_size == 144

    def test_action_vocab_matches_config(self) -> None:
        """action_vocab_size matches GameConfig.action_vocab_size."""
        for cfg in (MINI_2P, SMALL_2P, SMALL_3P, STANDARD_2P, STANDARD_3P, STANDARD_4P):
            tok = PlacementTokenizer(cfg)
            assert tok.action_vocab_size == cfg.action_vocab_size
