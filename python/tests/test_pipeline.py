from __future__ import annotations

import json

from gimbur_nn.pipeline import (
    PipelineConfig,
    SimulateConfig,
    TrainConfig,
    _load_section,
    _step_train,
)


def test_train_config_loads_replay_window() -> None:
    config = _load_section(TrainConfig, {"replayGenerations": 5})

    assert config.replay_generations == 5


def test_train_config_loads_value_blending_and_state_sampling() -> None:
    config = _load_section(
        TrainConfig,
        {
            "mctsValueWeight": 0.75,
            "earlyGameTurnLimit": 8,
            "maxLateGameStatesPerGame": 12,
        },
    )

    assert config.mcts_value_weight == 0.75
    assert config.early_game_turn_limit == 8
    assert config.max_late_game_states_per_game == 12


def test_simulate_config_loads_parallelism() -> None:
    config = _load_section(SimulateConfig, {"parallelism": 8})

    assert config.parallelism == 8


def test_step_train_passes_value_blending_and_state_sampling(tmp_path, monkeypatch) -> None:
    cfg = PipelineConfig(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        train=TrainConfig(
            mcts_value_weight=0.75,
            early_game_turn_limit=8,
            max_late_game_states_per_game=12,
        ),
    )
    monkeypatch.setattr("gimbur_nn.pipeline._run", lambda *args, **kwargs: None)

    _step_train(cfg, 0, tmp_path)

    config = json.loads((tmp_path / "models/.configs/train_gen0.json").read_text())
    assert config["mctsValueWeight"] == 0.75
    assert config["earlyGameTurnLimit"] == 8
    assert config["maxLateGameStatesPerGame"] == 12
