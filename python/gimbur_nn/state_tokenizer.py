"""
Tokenizer for converting serialized Catan game states into tensors.

Provides :class:`StateTokenizer` for game state strings (Part I of the spec).
Supports player rotation for perspective-invariant training.

Builds a **minimal vocabulary** from the game configuration
(map size + player count).  Characters are drawn from the Token
Alphabets table in ``docs/state-action-serialization.md``.
Separators ``|`` and ``/`` are stripped before tokenization.

See also :mod:`placement_tokenizer` for the placement phase tokenizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .game_config import GameConfig

# ── Alphabet building blocks ─────────────────────────────────────────

_RESOURCE_TYPE = "dwbsWo"
_PORT_GENERIC = "g"
_PIP_COUNT = "012345"
_PIP_SIDE = "lhn"
_BUILDING_TYPE = ".vc"
_PLAYER_ID_ALL = "_-+*^"
_TURN_STAGE = "aefirxyt"
_COUNT_REST = "6789ABCDEFGHJKMNPQRSTVWXYZ"

_STRIP = str.maketrans("", "", "|/")


def _player_id_chars(player_count: int) -> str:
    """Return the player ID characters needed for *player_count* players.

    Always includes ``_`` (none), plus one char per player.
    """
    # _=none, -=P1, +=P2, *=P3, ^=P4
    return _PLAYER_ID_ALL[: player_count + 1]


def _build_game_state_vocab(player_count: int) -> str:
    """Build the vocabulary string for the game state tokenizer.

    Order: resource type, port generic, pip count, side, building type,
    player id (sized for player_count), turn stage, count rest.
    Shared characters (pip digits 0-5) appear once at their first
    occurrence (in pip count); they are *not* duplicated in count rest.
    """
    chars: list[str] = []
    seen: set[str] = set()

    for group in (
        _RESOURCE_TYPE,
        _PORT_GENERIC,
        _PIP_COUNT,
        _PIP_SIDE,
        _BUILDING_TYPE,
        _player_id_chars(player_count),
        _TURN_STAGE,
        _COUNT_REST,
    ):
        for ch in group:
            if ch not in seen:
                chars.append(ch)
                seen.add(ch)
    return "".join(chars)



def _rotate_player_char(ch: str, rotation: int, n_players: int) -> str:
    """Remap a player-ID character by *rotation* positions."""
    if ch == "_":
        return "_"
    player_chars = _PLAYER_ID_ALL[1:]  # "-+*^"
    idx = player_chars.index(ch)
    new_idx = (idx - rotation) % n_players
    return player_chars[new_idx]


# ── Tokenizer classes ────────────────────────────────────────────────


class StateTokenizer:
    """Tokenizer for game state strings (Part I of the spec).

    Builds a minimal vocabulary from the game configuration.  Supports
    ``tokenize``, ``tokenize_batch``, and ``rotate_player_state``.

    Args:
        cfg: Game configuration describing topology sizes and player count.
    """

    def __init__(self, cfg: GameConfig) -> None:
        self._cfg = cfg
        self.vocab_chars: str = _build_game_state_vocab(cfg.player_count)
        """Ordered vocabulary characters."""

        self.vocab: dict[str, int] = {ch: i for i, ch in enumerate(self.vocab_chars)}
        """Maps each token character to its integer id (0-based)."""

        self.vocab_size: int = len(self.vocab_chars)
        """Number of unique token characters in this vocabulary."""

    def tokenize(self, state_str: str) -> torch.Tensor:
        """Convert a serialized game state into an int tensor.

        Separators ``|`` and ``/`` are stripped before tokenization.

        Args:
            state_str: A single serialized state string (human-readable
                or compact form).

        Returns:
            A 1-D ``torch.int`` tensor of token ids, one per character.

        Raises:
            KeyError: If the string contains a character not in the
                vocabulary.
        """
        compact = state_str.translate(_STRIP)
        return torch.tensor([self.vocab[ch] for ch in compact], dtype=torch.int)

    def tokenize_batch(self, state_strs: list[str]) -> torch.Tensor:
        """Tokenize multiple states into a 2-D tensor.

        All states must have the same length after stripping separators
        (i.e. same map and player count).

        Args:
            state_strs: List of serialized state strings.

        Returns:
            A 2-D ``torch.int`` tensor of shape ``(n, seq_len)``.

        Raises:
            ValueError: If the states have different lengths after
                stripping.
            KeyError: If any string contains a character not in the
                vocabulary.
        """
        tensors = [self.tokenize(s) for s in state_strs]
        if len(tensors) == 0:
            return torch.empty(0, 0, dtype=torch.int)
        seq_len = tensors[0].shape[0]
        for i, t in enumerate(tensors):
            if t.shape[0] != seq_len:
                msg = f"State {i} has length {t.shape[0]}, expected {seq_len}"
                raise ValueError(msg)
        return torch.stack(tensors)

    def rotate_player_state(
        self,
        compact: str,
        target_player: int,
    ) -> str:
        """Rotate a compact state string so *target_player* becomes player 1.

        The model always predicts **player 1's** win probability.  To
        obtain the win probability for an arbitrary player, rotate the
        serialized state so that the target player occupies the player-1
        slot, then run inference as usual.

        Rotation affects two kinds of data:

        1. **Player-ID tokens** (current turn, longest road, largest army,
           vertex owners, edge occupancy) are remapped via a cyclic shift.
        2. **Per-player data blocks** (resources, knights, dev cards) are
           reordered so the target player's block comes first.

        Section 11 (new dev cards this turn) is always relative to the
        current player and is left unchanged by rotation.

        Args:
            compact: Compact-form state string (no ``|``/``/`` separators).
            target_player: 1-based player number that should become
                player 1.

        Returns:
            A new compact-form string with the rotation applied.

        Raises:
            ValueError: If *target_player* is out of range or *compact*
                has the wrong length.
        """
        cfg = self._cfg
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

        # ── 1. Remap individual player-ID tokens ─────────────
        #
        # Section offsets (compact form, see docs/state-action-serialization.md):
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
            chars[pos_current_player], rotation, n,
        )

        # Turn stage (2nd char of current-turn section) is NOT a player ID.

        # Longest road owner.
        pos_lr = base + 3
        chars[pos_lr] = _rotate_player_char(chars[pos_lr], rotation, n)

        # Largest army owner.
        pos_la = base + 4
        chars[pos_la] = _rotate_player_char(chars[pos_la], rotation, n)

        # Vertex owners (every 2nd char starting at base+5).
        vertex_start = base + 5
        for vi in range(v):
            owner_pos = vertex_start + 2 * vi + 1
            chars[owner_pos] = _rotate_player_char(
                chars[owner_pos], rotation, n,
            )

        # Edge occupancy.
        edge_start = vertex_start + 2 * v
        for ei in range(e):
            pos = edge_start + ei
            chars[pos] = _rotate_player_char(chars[pos], rotation, n)

        # -- 2. Reorder per-player data blocks ----------------------------
        per_player_start = edge_start + e

        def _rotate_blocks(start: int, block_size: int) -> int:
            """Rotate N blocks of *block_size* chars starting at *start*.

            Returns the index just past the last block.
            """
            total = block_size * n
            section = chars[start : start + total]
            blocks = [
                section[i * block_size : (i + 1) * block_size]
                for i in range(n)
            ]
            rotated = blocks[rotation:] + blocks[:rotation]
            flat = [ch for blk in rotated for ch in blk]
            chars[start : start + total] = flat
            return start + total

        # Resources: N blocks of 5.
        next_start = _rotate_blocks(per_player_start, 5)
        # Knights: N blocks of 1.
        next_start = _rotate_blocks(next_start, 1)
        # Dev cards: N blocks of 5.
        next_start = _rotate_blocks(next_start, 5)

        # New dev cards (5), resolution state (2), and remaining deck (5)
        # are relative/global rather than absolute player blocks.
        winner_pos = next_start + 5 + 2 + 5
        chars[winner_pos] = _rotate_player_char(chars[winner_pos], rotation, n)

        return "".join(chars)
