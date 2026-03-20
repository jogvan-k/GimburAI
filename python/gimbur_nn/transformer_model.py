"""
PyTorch models for Catan state evaluation.

Takes tokenized state tensors and predicts the win chance of player 1.
To predict the win chance of other players, the state has to be rearranged
beforehand in order to have the given player appear as player 1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game_config import GameConfig
from .state_tokenizer import StateTokenizer


class GimburTransformerConfig:
    d_model: int  # Embedding dimension
    n_heads: int  # Number of heads per layer
    n_layers: int  # Number of multi-head transformer layers
    n_buckets: int  # Number of output value buckets
    ffn_hidden_mult: int  # Dimension multiplier in feed forward network in hidden layers
    dropout: float = 0.0


def _make_model_config(
    *,
    d_model: int,
    n_heads: int,
    n_layers: int,
    n_buckets: int,
    ffn_hidden_mult: int,
    dropout: float,
) -> GimburTransformerConfig:
    cfg = GimburTransformerConfig()
    cfg.d_model = d_model
    cfg.n_heads = n_heads
    cfg.n_layers = n_layers
    cfg.n_buckets = n_buckets
    cfg.ffn_hidden_mult = ffn_hidden_mult
    cfg.dropout = dropout
    return cfg


MODEL_SMALL = _make_model_config(
    d_model=256,
    n_heads=8,
    n_layers=8,
    n_buckets=128,
    ffn_hidden_mult=1,
    dropout=0.05,
)

MODEL_MEDIUM = _make_model_config(
    d_model=1024,
    n_heads=8,
    n_layers=8,
    n_buckets=128,
    ffn_hidden_mult=1,
    dropout=0.05,
)

MODEL_LARGE = _make_model_config(
    d_model=1024,
    n_heads=8,
    n_layers=16,
    n_buckets=128,
    ffn_hidden_mult=1,
    dropout=0.05,
)

MODEL_CONFIGS_BY_NAME: dict[str, GimburTransformerConfig] = {
    "small": MODEL_SMALL,
    "medium": MODEL_MEDIUM,
    "large": MODEL_LARGE,
}
"""Lookup table for predefined model configs."""


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(gated)


class TransformerBlock(nn.Module):
    """
    Decoder-style block, but without causal masking.

    Input:  x of shape [B, T, D]
    Output: x of shape [B, T, D]
    """

    def __init__(self, cfg: GimburTransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0

        hidden_dim = int(cfg.d_model * cfg.ffn_hidden_mult)

        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model,
            num_heads=cfg.n_heads,
            dropout=cfg.dropout,
            batch_first=True,
            bias=True,
        )
        self.dropout1 = nn.Dropout(cfg.dropout)

        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model, hidden_dim)
        self.dropout2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention over the full sequence, no causal mask.
        attn_in = self.ln1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + self.dropout1(attn_out)

        mlp_in = self.ln2(x)
        mlp_out = self.mlp(mlp_in)
        x = x + self.dropout2(mlp_out)

        return x


class GimburTransformer(nn.Module):
    """Neural network for evaluating Catan game states.

    Input:  tokenized state tensor (from tokenizer.tokenize_game).
    Output: player 1 win probability
    """

    def __init__(self, game_cfg: GameConfig, model_cfg: GimburTransformerConfig) -> None:
        super().__init__()
        self.game_config = game_cfg
        self.model_config = model_cfg
        assert model_cfg.d_model % model_cfg.n_heads == 0

        tok = StateTokenizer(game_cfg)
        self.tok_embeddings = nn.Embedding(tok.vocab_size, model_cfg.d_model)
        self.pos_embeddings = nn.Embedding(game_cfg.state_token_size, model_cfg.d_model)
        self.embed_dropout = nn.Dropout(model_cfg.dropout)

        # Transformer
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(model_cfg) for _ in range(model_cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(model_cfg.d_model)

        # Final bucketized output head
        self.bucket_head = nn.Linear(model_cfg.d_model, model_cfg.n_buckets, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Batch of tokenized game states, shape (batch, input_dim).

        Returns:
            Win probabilities, shape (batch, win_chance_buckets).
        """
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # [1, T]

        # [batch_size, token_length, emb_dimension]
        x = self.tok_embeddings(token_ids) + self.pos_embeddings(positions)
        x = self.embed_dropout(x)
        x = self.trf_blocks(x)
        x = self.final_ln(x)

        # Only keep the last token for each batch
        # [batch_size, emb_dimension]
        # last_token = x[:, -1, :]

        bucket_logits = self.bucket_head(x)
        return bucket_logits
