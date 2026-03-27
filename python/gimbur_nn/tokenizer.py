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

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .game_config import GameConfig

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
# Count (rest):   6 7 8 9 A B C D E F G H J K M N P Q R S T V X Y Z  (25)
# Total: 57 unique characters (W shared with resource type).

VOCAB_CHARS: str = "dwbsWog012345lhn.vc_-+*^aefirxyt6789ABCDEFGHJKMNPQRSTVXYZ"

VOCAB: dict[str, int] = {ch: idx for idx, ch in enumerate(VOCAB_CHARS)}
"""Maps each token character to its integer id (0-based)."""

VOCAB_SIZE: int = len(VOCAB)
"""Number of unique token characters (57)."""

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


# ── Player rotation ──────────────────────────────────────────────────

# Player ID characters in order: none, player 1..4.
_PLAYER_CHARS = "_-+*^"


def _rotate_player_char(ch: str, rotation: int, n_players: int) -> str:
    """Remap a single player-ID character by *rotation* positions.

    ``_`` (none) is unchanged.  For a player with 1-based index *p*,
    the new index is ``((p - 1 - rotation) % n_players) + 1``.
    """
    idx = _PLAYER_CHARS.index(ch)
    if idx == 0:  # '_' — no player
        return ch
    new_idx = ((idx - 1 - rotation) % n_players) + 1
    return _PLAYER_CHARS[new_idx]


def rotate_player_state(
    compact: str,
    target_player: int,
    cfg: GameConfig,
) -> str:
    """Rotate a compact state string so that *target_player* becomes player 1.

    The model always predicts **player 1's** win probability.  To obtain
    the win probability for an arbitrary player, rotate the serialized
    state so that the target player occupies the player-1 slot, then run
    inference as usual.

    Rotation affects two kinds of data:

    1. **Player-ID tokens** (current turn, longest road, largest army,
       vertex owners, edge occupancy) are remapped via a cyclic shift.
    2. **Per-player data blocks** (resources, knights, dev cards) are
       reordered so the target player's block comes first.

    Section 11 (new dev cards this turn) is always relative to the
    current player and is left unchanged by rotation.

    Args:
        compact: Compact-form state string (no ``|`` / ``/`` separators).
        target_player: 1-based player number that should become player 1.
        cfg: Game configuration describing the board topology sizes.

    Returns:
        A new compact-form string with the rotation applied.

    Raises:
        ValueError: If *target_player* is out of range or *compact*
            has the wrong length.
    """
    n = cfg.player_count
    if not 1 <= target_player <= n:
        msg = f"target_player must be 1..{n}, got {target_player}"
        raise ValueError(msg)
    if target_player == 1:
        return compact  # No rotation needed.

    expected_len = cfg.state_token_size
    if len(compact) != expected_len:
        msg = f"Expected {expected_len} chars, got {len(compact)}"
        raise ValueError(msg)

    rotation = target_player - 1
    t = cfg.tile_count
    v = cfg.vertex_count
    e = cfg.edge_count
    p = cfg.port_count

    chars = list(compact)

    # ── 1. Remap individual player-ID tokens ─────────────────────

    # Section offsets (compact form, see docs/state-serialization.md):
    #   Tiles:        0            .. 3*T
    #   Ports:        3*T          .. 3*T + P
    #   Robber:       3*T + P      (1 char)
    #   Current turn: 3*T + P + 1  (2 chars: player, stage)
    #   LR/LA:        3*T + P + 3  (2 chars)
    #   Vertices:     3*T + P + 5  (2*V chars, pairs of building+owner)
    #   Edges:        3*T + P + 5 + 2*V  (E chars)

    base = 3 * t + p

    # Current player (1st char of current-turn section).
    pos_current_player = base + 1
    chars[pos_current_player] = _rotate_player_char(
        chars[pos_current_player],
        rotation,
        n,
    )

    # Longest road owner.
    pos_lr = base + 3
    chars[pos_lr] = _rotate_player_char(chars[pos_lr], rotation, n)

    # Largest army owner.
    pos_la = base + 4
    chars[pos_la] = _rotate_player_char(chars[pos_la], rotation, n)

    # Vertex owners (2nd char of each vertex pair).
    vertex_start = base + 5
    for vi in range(v):
        owner_pos = vertex_start + 2 * vi + 1
        chars[owner_pos] = _rotate_player_char(
            chars[owner_pos],
            rotation,
            n,
        )

    # Edge occupancy (each char is a player ID).
    edge_start = vertex_start + 2 * v
    for ei in range(e):
        pos = edge_start + ei
        chars[pos] = _rotate_player_char(chars[pos], rotation, n)

    # ── 2. Reorder per-player data blocks ────────────────────────

    per_player_start = edge_start + e  # Start of resources section.

    # Resources: N blocks of 5 tokens.
    res_start = per_player_start
    res_blocks = [chars[res_start + i * 5 : res_start + (i + 1) * 5] for i in range(n)]
    rotated_res = res_blocks[rotation:] + res_blocks[:rotation]
    for i, block in enumerate(rotated_res):
        chars[res_start + i * 5 : res_start + (i + 1) * 5] = block

    # Knights: N blocks of 1 token.
    kn_start = res_start + 5 * n
    kn_blocks = [chars[kn_start + i] for i in range(n)]
    rotated_kn = kn_blocks[rotation:] + kn_blocks[:rotation]
    for i, ch in enumerate(rotated_kn):
        chars[kn_start + i] = ch

    # Dev cards: N blocks of 5 tokens.
    dev_start = kn_start + n
    dev_blocks = [chars[dev_start + i * 5 : dev_start + (i + 1) * 5] for i in range(n)]
    rotated_dev = dev_blocks[rotation:] + dev_blocks[:rotation]
    for i, block in enumerate(rotated_dev):
        chars[dev_start + i * 5 : dev_start + (i + 1) * 5] = block

    return "".join(chars)
