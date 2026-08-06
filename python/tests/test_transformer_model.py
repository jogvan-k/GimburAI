"""Player-value model architecture shape tests."""

from __future__ import annotations

import pytest
import torch

from gimbur_nn.game_config import ALL_CONFIGS, MINI_2P
from gimbur_nn.transformer_model import (
    GimburPlacementTransformer,
    GimburTransformer,
    _make_model_config,
)


def _config(output_mode: str):
    return _make_model_config(
        d_model=16,
        n_heads=4,
        n_layers=1,
        ffn_hidden_mult=1,
        dropout=0.0,
        output_mode=output_mode,
    )


@pytest.mark.parametrize("game_cfg", ALL_CONFIGS, ids=lambda cfg: cfg.name)
def test_state_combined_shapes_match_contract(game_cfg) -> None:
    model = GimburTransformer(game_cfg, _config("combined"))
    output = model(torch.zeros(2, game_cfg.state_token_size, dtype=torch.long))

    assert output["value"].shape == (2, game_cfg.player_count)
    assert output["policy"].shape == (2, game_cfg.policy_size)


def test_state_rejects_value_only() -> None:
    with pytest.raises(ValueError, match="only 'combined'"):
        GimburTransformer(MINI_2P, _config("value"))


@pytest.mark.parametrize("game_cfg", ALL_CONFIGS, ids=lambda cfg: cfg.name)
@pytest.mark.parametrize("output_mode", ["value", "combined"])
def test_placement_value_shape_matches_player_count(game_cfg, output_mode: str) -> None:
    model = GimburPlacementTransformer(game_cfg, _config(output_mode))
    output = model(torch.zeros(2, game_cfg.placement_token_size, dtype=torch.long))
    assert output["value"].shape == (2, game_cfg.player_count)
    assert output["policy"].shape == (2, game_cfg.placement_policy_size)
    assert game_cfg.placement_policy_size == game_cfg.vertex_count


def test_placement_rejects_policy_only() -> None:
    with pytest.raises(ValueError, match="only 'value' and 'combined'"):
        GimburPlacementTransformer(MINI_2P, _config("policy"))
