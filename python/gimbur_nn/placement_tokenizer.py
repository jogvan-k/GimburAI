"""
Tokenizer for the placement phase (state + action).

Provides :class:`PlacementTokenizer` which assigns every state
character and every action string a unique token ID in a single
embedding space.  State character tokens occupy IDs 0..S-1 and
action tokens occupy IDs S..S+A-1, where S is the number of state
vocabulary characters and A is the number of valid placement actions
for the map.

The combined vocabulary allows the transformer to attend across
both state and action tokens in a single sequence.

See also :mod:`state_tokenizer` for the game state tokenizer.
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
_PLACEMENT_NUMBER = ".ab"
_PLAYER_ID_ALL = "_-+*^"

_STRIP = str.maketrans("", "", "|/")


def _player_id_chars(player_count: int) -> str:
    """Return the player ID characters needed for *player_count* players."""
    return _PLAYER_ID_ALL[: player_count + 1]


def _build_placement_state_vocab(player_count: int) -> str:
    """Build the state vocabulary string for the placement phase.

    Order: resource type, port generic, pip count, side, placement number,
    player id (sized for player_count).
    No turn stage, no count-rest chars, no building type ``v``/``c``.
    """
    chars: list[str] = []
    seen: set[str] = set()

    for group in (
        _RESOURCE_TYPE,
        _PORT_GENERIC,
        _PIP_COUNT,
        _PIP_SIDE,
        _PLACEMENT_NUMBER,
        _player_id_chars(player_count),
    ):
        for ch in group:
            if ch not in seen:
                chars.append(ch)
                seen.add(ch)
    return "".join(chars)


# ── Action tables (from topology-reference.md) ───────────────────────
#
# Each tuple lists every valid placement action string for the map.
# Sorted by vertex index, then direction (alphabetical).
# Token index offset == state_vocab_size + position in list.

_MINI_ACTIONS: tuple[str, ...] = (
    '0SE', '0SW', '1SE', '1SW', '2NE', '2S', '3NE', '3NW',
    '3S', '4NW', '4S', '5N', '5SE', '5SW', '6N', '6SE',
    '6SW', '7N', '7SE', '7SW', '8NE', '8S', '9NE', '9NW',
    '9S', '10NE', '10NW', '10S', '11NW', '11S', '12N', '12SE',
    '13N', '13SW', '14N', '14SE', '14SW', '15N', '15SE', '15SW',
    '16NE', '16NW', '16S', '17NE', '17NW', '17S', '18NE', '18NW',
    '18S', '19N', '19SE', '20N', '20SE', '20SW', '21N', '21SW',
    '22NE', '22NW', '23NE', '23NW',
)

_SMALL_ACTIONS: tuple[str, ...] = (
    '0SE', '0SW', '1SE', '1SW', '2SE', '2SW', '3NE', '3S',
    '4NE', '4NW', '4S', '5NE', '5NW', '5S', '6NW', '6S',
    '7N', '7SE', '7SW', '8N', '8SE', '8SW', '9N', '9SE',
    '9SW', '10N', '10SE', '10SW', '11NE', '11S', '12NE', '12NW',
    '12S', '13NE', '13NW', '13S', '14NE', '14NW', '14S', '15NW',
    '15S', '16N', '16SE', '17N', '17SE', '17SW', '18N', '18SE',
    '18SW', '19N', '19SE', '19SW', '20N', '20SW', '21NE', '21NW',
    '21S', '22NE', '22NW', '22S', '23NE', '23NW', '23S', '24NE',
    '24NW', '24S', '25N', '25SE', '26N', '26SE', '26SW', '27N',
    '27SE', '27SW', '28N', '28SW', '29NE', '29NW', '30NE', '30NW',
    '31NE', '31NW',
)

_STANDARD_ACTIONS: tuple[str, ...] = (
    '0SE', '0SW', '1SE', '1SW', '2SE', '2SW', '3NE', '3S',
    '4NE', '4NW', '4S', '5NE', '5NW', '5S', '6NW', '6S',
    '7N', '7SE', '7SW', '8N', '8SE', '8SW', '9N', '9SE',
    '9SW', '10N', '10SE', '10SW', '11NE', '11S', '12NE', '12NW',
    '12S', '13NE', '13NW', '13S', '14NE', '14NW', '14S', '15NW',
    '15S', '16N', '16SE', '16SW', '17N', '17SE', '17SW', '18N',
    '18SE', '18SW', '19N', '19SE', '19SW', '20N', '20SE', '20SW',
    '21NE', '21S', '22NE', '22NW', '22S', '23NE', '23NW', '23S',
    '24NE', '24NW', '24S', '25NE', '25NW', '25S', '26NW', '26S',
    '27N', '27SE', '28N', '28SW', '29N', '29SE', '29SW', '30N',
    '30SE', '30SW', '31N', '31SE', '31SW', '32N', '32SE', '32SW',
    '33NE', '33NW', '33S', '34NE', '34NW', '34S', '35NE', '35NW',
    '35S', '36NE', '36NW', '36S', '37NE', '37NW', '37S', '38N',
    '38SE', '39N', '39SE', '39SW', '40N', '40SE', '40SW', '41N',
    '41SE', '41SW', '42N', '42SW', '43NE', '43NW', '43S', '44NE',
    '44NW', '44S', '45NE', '45NW', '45S', '46NE', '46NW', '46S',
    '47N', '47SE', '48N', '48SE', '48SW', '49N', '49SE', '49SW',
    '50N', '50SW', '51NE', '51NW', '52NE', '52NW', '53NE', '53NW',
)

_ACTIONS_BY_MAP: dict[str, tuple[str, ...]] = {
    "mini": _MINI_ACTIONS,
    "small": _SMALL_ACTIONS,
    "standard": _STANDARD_ACTIONS,
}


# ── Tokenizer class ──────────────────────────────────────────────────


class PlacementTokenizer:
    """Unified tokenizer for placement phase state + action.

    State characters and action strings share a single ID space
    so the transformer can attend across both in one sequence.

    State character tokens occupy IDs ``0 .. state_vocab_size - 1``.
    Action tokens occupy IDs ``state_vocab_size .. vocab_size - 1``.

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

        # -- Action vocabulary (string-level, offset by S) --
        actions = _ACTIONS_BY_MAP.get(cfg.map_name)
        if actions is None:
            msg = f"Unknown map: {cfg.map_name}"
            raise ValueError(msg)
        self.actions: tuple[str, ...] = actions
        """Ordered action strings (position + state_vocab_size = token id)."""

        offset = self.state_vocab_size
        self.action_vocab: dict[str, int] = {
            a: offset + i for i, a in enumerate(actions)
        }
        """Maps each action string to its global integer token id."""

        self.action_vocab_size: int = len(actions)
        """Number of unique placement actions (A)."""

        self.vocab_size: int = self.state_vocab_size + self.action_vocab_size
        """Total vocabulary size S + A (for embedding table)."""

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

    # -- Action tokenization --

    def tokenize_action(self, action: str) -> int:
        """Convert a single action string to its global token id.

        Returns:
            The integer token id (in range ``[state_vocab_size, vocab_size)``).

        Raises:
            KeyError: If the action is not in the vocabulary.
        """
        return self.action_vocab[action]

    def decode_action(self, token_id: int) -> str:
        """Convert a global token id back to the action string.

        Raises:
            IndexError: If *token_id* is out of range.
        """
        return self.actions[token_id - self.state_vocab_size]

    # -- Combined state + action tokenization --

    def tokenize_state_action(self, state_str: str, action: str) -> torch.Tensor:
        """Tokenize a state string and append a single action token.

        Returns:
            A 1-D ``torch.int`` tensor of length ``placement_token_size + 1``.
        """
        state_ids = self.tokenize_state(state_str)
        action_id = torch.tensor([self.action_vocab[action]], dtype=torch.int)
        return torch.cat([state_ids, action_id])

    def tokenize_state_actions(
        self, state_str: str, actions: list[str],
    ) -> torch.Tensor:
        """Tokenize a state string paired with each action in a list.

        Each row is the same state token sequence with a different
        action token appended.

        Args:
            state_str: A single placement state string.
            actions: List of action strings.

        Returns:
            A 2-D ``torch.int`` tensor of shape
            ``(len(actions), placement_token_size + 1)``.
        """
        state_ids = self.tokenize_state(state_str)
        rows: list[torch.Tensor] = []
        for act in actions:
            action_id = torch.tensor([self.action_vocab[act]], dtype=torch.int)
            rows.append(torch.cat([state_ids, action_id]))
        if len(rows) == 0:
            seq_len = state_ids.shape[0] + 1
            return torch.empty(0, seq_len, dtype=torch.int)
        return torch.stack(rows)
