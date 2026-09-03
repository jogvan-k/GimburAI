from __future__ import annotations

import pytest

from gimbur_nn.train import _limit_games


def test_limit_games_ignores_excess_games() -> None:
    games = [{"seed": seed} for seed in range(200)]

    selected = _limit_games(games, 150)

    assert len(selected) == 150
    assert selected[-1]["seed"] == 149


def test_limit_games_rejects_insufficient_games() -> None:
    with pytest.raises(ValueError, match="requested exactly 150 games"):
        _limit_games([{}] * 149, 150)
