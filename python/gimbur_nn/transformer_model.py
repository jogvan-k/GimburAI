"""
PyTorch models for Catan state evaluation.

Takes tokenized state tensors and predicts a win distribution over players.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .game_config import GameConfig
from .placement_tokenizer import PlacementTokenizer
from .state_tokenizer import StateTokenizer


class OutputMode(Enum):
    """Controls the output head topology of the model.

    * ``VALUE`` — per-player value head.
    * ``POLICY`` — policy-only head.
    * ``COMBINED`` — per-player value and policy heads.
    """

    VALUE = "value"
    """Single head predicting a player win distribution."""

    POLICY = "policy"
    """Single policy head predicting visit-share distribution."""

    COMBINED = "combined"
    """Dual heads: player value head and dense policy head."""


OUTPUT_MODES = tuple(m.value for m in OutputMode)
"""Valid output mode strings for CLI / config validation."""


class GimburTransformerConfig:
    d_model: int  # Embedding dimension
    n_heads: int  # Number of heads per layer
    n_layers: int  # Number of multi-head transformer layers
    ffn_hidden_mult: int  # Dimension multiplier in feed forward network in hidden layers
    dropout: float = 0.0
    output_mode: str = "value"  # "value" | "policy" | "combined"


def _make_model_config(
    *,
    d_model: int,
    n_heads: int,
    n_layers: int,
    ffn_hidden_mult: int,
    dropout: float,
    output_mode: str = "value",
) -> GimburTransformerConfig:
    cfg = GimburTransformerConfig()
    cfg.d_model = d_model
    cfg.n_heads = n_heads
    cfg.n_layers = n_layers
    cfg.ffn_hidden_mult = ffn_hidden_mult
    cfg.dropout = dropout
    cfg.output_mode = output_mode
    return cfg


MODEL_SMALL = _make_model_config(
    d_model=256,
    n_heads=8,
    n_layers=8,
    ffn_hidden_mult=1,
    dropout=0.05,
)

MODEL_MEDIUM = _make_model_config(
    d_model=1024,
    n_heads=8,
    n_layers=8,
    ffn_hidden_mult=1,
    dropout=0.05,
)

MODEL_LARGE = _make_model_config(
    d_model=1024,
    n_heads=8,
    n_layers=16,
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
    Output: raw per-player value logits with shape ``(batch, player_count)``.
    """

    def __init__(self, game_cfg: GameConfig, model_cfg: GimburTransformerConfig) -> None:
        super().__init__()
        self.game_config = game_cfg
        self.model_config = model_cfg
        self.output_mode = getattr(model_cfg, "output_mode", "value")
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

        if self.output_mode != "value":
            raise ValueError("State models support only 'value' output mode.")
        self.value_head = nn.Linear(model_cfg.d_model, game_cfg.player_count, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: Batch of tokenized game states, shape (batch, seq_len).

        Returns:
            Raw player logits with shape ``(batch, player_count)``.
        """
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)  # [1, T]

        # [batch_size, token_length, emb_dimension]
        x = self.tok_embeddings(token_ids) + self.pos_embeddings(positions)
        x = self.embed_dropout(x)
        x = self.trf_blocks(x)
        x = self.final_ln(x)

        return self.value_head(x.mean(dim=1))


class GimburPlacementTransformer(nn.Module):
    """Placement model with pooled player-value and stage-policy heads."""

    def __init__(self, game_cfg: GameConfig, model_cfg: GimburTransformerConfig) -> None:
        super().__init__()
        self.game_config = game_cfg
        self.model_config = model_cfg
        self.output_mode = getattr(model_cfg, "output_mode", "value")
        if self.output_mode not in ("value", "combined"):
            raise ValueError("Placement models support only 'value' and 'combined' output modes.")
        assert model_cfg.d_model % model_cfg.n_heads == 0

        tok = PlacementTokenizer(game_cfg)
        self.tok_embeddings = nn.Embedding(tok.vocab_size, model_cfg.d_model)
        self.pos_embeddings = nn.Embedding(game_cfg.placement_token_size, model_cfg.d_model)
        self.embed_dropout = nn.Dropout(model_cfg.dropout)

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(model_cfg) for _ in range(model_cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(model_cfg.d_model)

        self.value_head = nn.Linear(model_cfg.d_model, game_cfg.player_count, bias=False)
        self.policy_head = nn.Linear(model_cfg.d_model, tok.policy_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.tok_embeddings(token_ids) + self.pos_embeddings(positions)
        x = self.embed_dropout(x)
        x = self.trf_blocks(x)
        x = self.final_ln(x)
        pooled = x.mean(dim=1)

        return {
            "value": self.value_head(pooled),
            "policy": self.policy_head(pooled),
        }
