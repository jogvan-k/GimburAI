"""Player-value model architecture shape tests."""

from __future__ import annotations

import pytest
import torch

from gimbur_nn.game_config import ALL_CONFIGS
from gimbur_nn.transformer_model import GimburTransformer, _make_model_config


def _config():
    return _make_model_config(
        d_model=16,
        n_heads=4,
        n_layers=1,
        ffn_hidden_mult=1,
        dropout=0.0,
    )


@pytest.mark.parametrize("game_cfg", ALL_CONFIGS, ids=lambda cfg: cfg.name)
def test_state_combined_shapes_match_contract(game_cfg) -> None:
    model = GimburTransformer(game_cfg, _config())
    output = model(torch.zeros(2, game_cfg.state_token_size, dtype=torch.long))

    assert output["value"].shape == (2, game_cfg.player_count)
    assert output["policy"].shape == (2, game_cfg.policy_size)
