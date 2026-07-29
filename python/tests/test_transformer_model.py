"""Placement architecture shape tests."""

from __future__ import annotations

import pytest
import torch

from gimbur_nn.game_config import MINI_2P
from gimbur_nn.transformer_model import GimburPlacementTransformer, _make_model_config


def _config(output_mode: str):
    return _make_model_config(
        d_model=16,
        n_heads=4,
        n_layers=1,
        n_buckets=8,
        ffn_hidden_mult=1,
        dropout=0.0,
        output_mode=output_mode,
    )


def test_placement_value_shape_is_pooled() -> None:
    model = GimburPlacementTransformer(MINI_2P, _config("value"))
    output = model(torch.zeros(3, MINI_2P.placement_token_size, dtype=torch.long))

    assert output.shape == (3, 8)


def test_placement_combined_shapes_are_dense() -> None:
    model = GimburPlacementTransformer(MINI_2P, _config("combined"))
    output = model(torch.zeros(3, MINI_2P.placement_token_size, dtype=torch.long))

    assert output["value"].shape == (3, 8)
    assert output["policy"].shape == (3, MINI_2P.action_vocab_size)


def test_placement_rejects_policy_only() -> None:
    with pytest.raises(ValueError, match="only 'value' and 'combined'"):
        GimburPlacementTransformer(MINI_2P, _config("policy"))
