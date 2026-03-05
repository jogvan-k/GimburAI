"""
Tokenizer for converting serialized Catan game states into tensors.

The tokenizer strips '|' and '/' characters, states can therefore
be in either human readable or copact state. See ../../docs/state-serialization.md
for the expected state.
"""

from __future__ import annotations

import torch


def tokenize(state_str: str) -> torch.Tensor:
    """Parse a serialized state into a tensor.

    Args:
        state_str: Serialized board

    Returns:
        A 1-D int tensor representing the state.
    """
    raise NotImplementedError
