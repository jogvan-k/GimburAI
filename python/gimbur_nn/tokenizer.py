"""
Tokenizer for converting serialized Catan game states into tensors.

The tokenizer strips '|' and '/' separators so inputs can be in either
human-readable or compact form. See ``docs/state-serialization.md`` for
the full serialization spec.

The vocabulary maps each token character to a unique integer, ordered by
the Token Alphabets table in the spec: resource type, port generic, pip
count, side, building type, player ID, turn stage, then the remaining
count characters (digits 0-5 are shared with pip count and only listed
once).
"""

from __future__ import annotations

import torch

# Vocabulary: characters listed in the order they appear in the Token
# Alphabets table of docs/state-serialization.md.  Shared characters
# (digits 0-5 used by both pip count and count) appear once at their
# first occurrence.
#
# Resource type:  d w b s W o        (6)
# Port generic:   g                  (1)
# Pip count:      0 1 2 3 4 5        (6)
# Side:           l h n              (3)
# Building type:  . v c              (3)
# Player ID:      _ - + * ^          (5)
# Turn stage:     a e f i r x y t    (8)
# Count (rest):   6 7 8 9 A B C D E F G H J K  (14)
# Total: 46 unique characters.

VOCAB_CHARS: str = "dwbsWog012345lhn.vc_-+*^aefirxyt6789ABCDEFGHJK"

VOCAB: dict[str, int] = {ch: idx for idx, ch in enumerate(VOCAB_CHARS)}
"""Maps each token character to its integer id (0-based)."""

VOCAB_SIZE: int = len(VOCAB)
"""Number of unique token characters (46)."""

_STRIP = str.maketrans("", "", "|/")


def tokenize(state_str: str) -> torch.Tensor:
    """Convert one or more serialized states into an int tensor.

    Separators ``|`` and ``/`` are stripped before tokenization.

    Args:
        state_str: A single serialized state string (human-readable or
            compact form).

    Returns:
        A 1-D ``torch.int`` tensor of token ids, one per character.

    Raises:
        KeyError: If the string contains a character not in the vocabulary.
    """
    compact = state_str.translate(_STRIP)
    return torch.tensor([VOCAB[ch] for ch in compact], dtype=torch.int)


def tokenize_batch(state_strs: list[str]) -> torch.Tensor:
    """Tokenize multiple states into a 2-D tensor.

    All states must have the same length after stripping separators
    (i.e. same map and player count).

    Args:
        state_strs: List of serialized state strings.

    Returns:
        A 2-D ``torch.int`` tensor of shape ``(n, seq_len)``.

    Raises:
        ValueError: If the states have different lengths after stripping.
        KeyError: If any string contains a character not in the vocabulary.
    """
    tensors = [tokenize(s) for s in state_strs]
    if len(tensors) == 0:
        return torch.empty(0, 0, dtype=torch.int)
    seq_len = tensors[0].shape[0]
    for i, t in enumerate(tensors):
        if t.shape[0] != seq_len:
            msg = f"State {i} has length {t.shape[0]}, expected {seq_len}"
            raise ValueError(msg)
    return torch.stack(tensors)
