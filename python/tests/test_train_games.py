from __future__ import annotations

import json

import pytest

from gimbur_nn.train import _limit_games, _load_training_games


def test_limit_games_ignores_excess_games() -> None:
    games = [{"seed": seed} for seed in range(200)]

    selected = _limit_games(games, 150)

    assert len(selected) == 150
    assert selected[-1]["seed"] == 149


def test_limit_games_rejects_insufficient_games() -> None:
    with pytest.raises(ValueError, match="requested exactly 150 games"):
        _limit_games([{}] * 149, 150)


def test_load_training_games_balances_generation_inputs(tmp_path) -> None:
    newest = tmp_path / "gen2.jsonl"
    replay = tmp_path / "gen1.jsonl"
    newest.write_text("".join(json.dumps({"seed": seed}) + "\n" for seed in range(10)))
    replay.write_text("".join(json.dumps({"seed": seed}) + "\n" for seed in range(10, 15)))

    games = _load_training_games([newest, replay], 4)

    assert [game["seed"] for game in games] == [0, 1, 2, 3, 10, 11, 12, 13]
