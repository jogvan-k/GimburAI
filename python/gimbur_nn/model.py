"""
PyTorch models for Catan state evaluation.

Takes tokenized state tensors and predicts the win chance of player 1.
To predict the win chance of other players, the state has to be rearranged
beforehand in order to have the given player appear as player 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GimburTransformer(nn.Module):
    """Neural network for evaluating Catan game states.

    Input:  tokenized state tensor (from tokenizer.tokenize_game).
    Output: player 1 win probability
    """

    def __init__(self, input_dim: int, player_count: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.player_count = player_count
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of tokenized game states, shape (batch, input_dim).

        Returns:
            Win probabilities, shape (batch, win_chance_buckets).
        """
        raise NotImplementedError
