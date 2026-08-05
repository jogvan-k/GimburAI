"""Tokenizer for canonical five-section placement-phase states."""

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
_PLACEMENT_NUMBER = ".abp"
_PLACEMENT_STAGE = "aefi"
_PLAYER_ID_ALL = "_-+*^"

_STRIP = str.maketrans("", "", "|/")

DIRECTION_ORDER: tuple[str, ...] = ("N", "NE", "SE", "S", "SW", "NW")
"""Clockwise road-direction policy order, matching C#."""


def _player_id_chars(player_count: int) -> str:
    """Return the player ID characters needed for *player_count* players."""
    return _PLAYER_ID_ALL[: player_count + 1]


def _build_placement_state_vocab(player_count: int) -> str:
    """Build the state vocabulary string for the placement phase.

    Order: resource type, port generic, pip count, side, placement number,
    placement stage, player id (sized for player_count). Serialized owner IDs
    are already canonicalized by C# so the acting player uses player 1.
    """
    chars: list[str] = []
    seen: set[str] = set()

    for group in (
        _RESOURCE_TYPE,
        _PORT_GENERIC,
        _PIP_COUNT,
        _PIP_SIDE,
        _PLACEMENT_NUMBER,
        _PLACEMENT_STAGE,
        _player_id_chars(player_count),
    ):
        for ch in group:
            if ch not in seen:
                chars.append(ch)
                seen.add(ch)
    return "".join(chars)


# ── Tokenizer class ──────────────────────────────────────────────────


class PlacementTokenizer:
    """Tokenize placement states and expose stage-policy output indices.

    Args:
        cfg: Game configuration (uses map_name, player_count, topology sizes).
    """

    def __init__(self, cfg: GameConfig) -> None:
        self._cfg = cfg

        # -- State vocabulary (character-level) --
        self.state_vocab_chars: str = _build_placement_state_vocab(cfg.player_count)
        """Ordered state vocabulary characters."""

        self.state_vocab: dict[str, int] = {
            ch: i for i, ch in enumerate(self.state_vocab_chars)
        }
        """Maps each state character to its integer id (0-based)."""

        self.state_vocab_size: int = len(self.state_vocab_chars)
        """Number of unique state vocabulary characters (S)."""

        self.policy_size: int = max(cfg.vertex_count, len(DIRECTION_ORDER))
        """Fixed policy width; settlement uses V logits and road uses six."""

        self.vocab_size: int = self.state_vocab_size
        """State input vocabulary size for the embedding table."""

    # -- State tokenization --

    def tokenize_state(self, state_str: str) -> torch.Tensor:
        """Convert a placement state string into an int tensor.

        Separators ``|`` and ``/`` are stripped before tokenization.

        Returns:
            A 1-D ``torch.int`` tensor of state token ids.

        Raises:
            KeyError: If the string contains a character not in the
                state vocabulary.
        """
        compact = state_str.translate(_STRIP)
        return torch.tensor(
            [self.state_vocab[ch] for ch in compact], dtype=torch.int,
        )

    def tokenize_batch(self, states: list[str]) -> torch.Tensor:
        """Tokenize a batch of equally sized placement states."""
        return torch.stack([self.tokenize_state(state) for state in states])

    # -- Stage-policy output indexing --

    def vertex_action_index(self, vertex: int) -> int:
        """Return the settlement-policy index for a vertex."""
        if not 0 <= vertex < self._cfg.vertex_count:
            raise ValueError(f"vertex must be in [0, {self._cfg.vertex_count})")
        return vertex

    @staticmethod
    def direction_action_index(direction: str) -> int:
        """Return the road-policy index for a canonical direction."""
        try:
            return DIRECTION_ORDER.index(direction)
        except ValueError as exc:
            raise ValueError(f"unknown road direction: {direction}") from exc
