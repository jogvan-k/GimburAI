from __future__ import annotations

import json

from gimbur_nn.pipeline import (
    PipelineConfig,
    SimulateConfig,
    TrainConfig,
    _count_json_files,
    _discard_stop_reason,
    _load_section,
    _step_simulate,
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
    config = _load_section(
        SimulateConfig,
        {
            "parallelism": 8,
            "maxPendingEvaluations": 16,
            "leafEvaluationTimeoutMs": 250,
            "drainTimeoutMs": 750,
            "maxErrorsPerGame": 3,
            "maxDiscardRate": 0.1,
        },
    )

    assert config.parallelism == 8
    assert config.max_pending_evaluations == 16
    assert config.leaf_evaluation_timeout_ms == 250
    assert config.drain_timeout_ms == 750
    assert config.max_errors_per_game == 3
    assert config.max_discard_rate == 0.1


def test_accepted_count_excludes_discarded_files(tmp_path) -> None:
    (tmp_path / "accepted.json").write_text("{}")
    discarded = tmp_path / "discarded"
    discarded.mkdir()
    (discarded / "bad.json").write_text("{}")

    assert _count_json_files(tmp_path) == 1


def test_discard_stop_thresholds(tmp_path) -> None:
    discarded = tmp_path / "discarded"
    discarded.mkdir()
    for index in range(3):
        (discarded / f"{index}.json").write_text("{}")

    sim = SimulateConfig(max_discarded_games=2)
    assert _discard_stop_reason(tmp_path, sim).startswith("discarded games")

    sim = SimulateConfig(max_discarded_games=10, max_consecutive_discards=2)
    assert _discard_stop_reason(tmp_path, sim).startswith("consecutive discards")


def test_step_simulate_writes_discard_policy_config(tmp_path, monkeypatch) -> None:
    cfg = PipelineConfig(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        simulate=SimulateConfig(
            games=1,
            max_errors_per_game=3,
            discard_games_with_fallbacks=True,
            max_discard_rate=0.1,
        ),
    )
    monkeypatch.setattr(
        "gimbur_nn.pipeline.subprocess.run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0})(),
    )

    _step_simulate(cfg, 0, tmp_path, None)

    config = json.loads((tmp_path / "models/.configs/simulate_gen0.json").read_text())
    assert config["maxErrorsPerGame"] == 3
    assert config["discardGamesWithFallbacks"] is True
    assert config["maxDiscardRate"] == 0.1


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
