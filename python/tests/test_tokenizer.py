"""Tests for the tokenizer against the examples in docs/state-serialization.md."""

from __future__ import annotations

import pytest
import torch

from gimbur_nn.tokenizer import VOCAB, VOCAB_CHARS, VOCAB_SIZE, tokenize, tokenize_batch

# ---------------------------------------------------------------------------
# Example states copied verbatim from docs/state-serialization.md
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
    "|gwWgsgb"
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
# of the compact form (separators stripped) through the VOCAB dictionary.

MINI_EXPECTED = [
    1, 12, 13, 2, 10, 13, 3, 11, 13, 4, 10, 14, 0, 7, 15, 4, 11, 14, 5, 9, 13,
    6, 3, 6, 2, 6, 1,
    11,
    20, 31,
    19, 19,
    16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 20, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 17, 21, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19, 16, 19,
    19, 19, 19, 19, 19, 20, 19, 19, 19, 19, 19, 19, 19, 21, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
    9, 8, 7, 8, 7, 7, 7, 8, 10, 7,
    7, 7,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
]

SMALL_EXPECTED = [
    4, 9, 13, 2, 10, 13, 3, 11, 13, 1, 10, 14, 2, 9, 14, 1, 8, 14, 5, 12, 13, 4, 11, 14, 3, 12, 14, 0, 7, 15,
    6, 1, 4, 6, 3, 6, 2,
    35,
    21, 28,
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
    21, 31,
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
    def test_size_is_46(self) -> None:
        assert VOCAB_SIZE == 46

    def test_no_duplicate_chars(self) -> None:
        assert len(set(VOCAB_CHARS)) == len(VOCAB_CHARS)

    def test_dict_matches_chars(self) -> None:
        assert len(VOCAB) == len(VOCAB_CHARS)
        for idx, ch in enumerate(VOCAB_CHARS):
            assert VOCAB[ch] == idx


# ---------------------------------------------------------------------------
# Full-tensor comparison for each doc example
# ---------------------------------------------------------------------------


class TestMiniMap:
    def test_full_tensor(self) -> None:
        t = tokenize(MINI_STATE)
        assert t.shape == (132,)
        assert t.dtype == torch.int32
        assert t.tolist() == MINI_EXPECTED


class TestSmallMap:
    def test_full_tensor(self) -> None:
        t = tokenize(SMALL_STATE)
        assert t.shape == (169,)
        assert t.dtype == torch.int32
        assert t.tolist() == SMALL_EXPECTED


class TestStandardMap:
    def test_full_tensor(self) -> None:
        t = tokenize(STANDARD_STATE)
        assert t.shape == (284,)
        assert t.dtype == torch.int32
        assert t.tolist() == STANDARD_EXPECTED


# ---------------------------------------------------------------------------
# Compact form (no separators) produces same result
# ---------------------------------------------------------------------------


class TestCompactForm:
    def test_compact_equals_human_readable(self) -> None:
        """Stripping | and / from human-readable form should give same tokens."""
        human = tokenize(MINI_STATE)
        compact_str = MINI_STATE.replace("|", "").replace("/", "")
        compact = tokenize(compact_str)
        assert torch.equal(human, compact)


# ---------------------------------------------------------------------------
# Batch tokenization
# ---------------------------------------------------------------------------


class TestBatch:
    def test_same_map_batch(self) -> None:
        t = tokenize_batch([MINI_STATE, MINI_STATE])
        assert t.shape == (2, 132)
        assert t.dtype == torch.int32

    def test_batch_matches_single(self) -> None:
        single = tokenize(SMALL_STATE)
        batch = tokenize_batch([SMALL_STATE])
        assert torch.equal(batch[0], single)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            tokenize_batch([MINI_STATE, SMALL_STATE])

    def test_empty_batch(self) -> None:
        t = tokenize_batch([])
        assert t.shape == (0, 0)
        assert t.dtype == torch.int32


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_character_raises(self) -> None:
        with pytest.raises(KeyError):
            tokenize("w5l#invalid")
