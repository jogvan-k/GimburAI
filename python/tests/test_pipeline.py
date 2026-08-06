from __future__ import annotations

import json

import pytest

from gimbur_nn.pipeline import (
    BenchmarkConfig,
    PipelineConfig,
    PromotionGateConfig,
    SimulateConfig,
    TrainConfig,
    _benchmark_score,
    _count_json_files,
    _discard_stop_reason,
    _evaluate_promotion_gate,
    _generation_complete,
    _load_config,
    _load_section,
    _run_placement_and_state_generation,
    _save_progress_chart,
    _save_summary,
    _step_simulate,
    _step_train,
)


def test_train_config_loads_replay_window() -> None:
    config = _load_section(TrainConfig, {"replayGenerations": 5})

    assert config.replay_generations == 5


def test_train_config_loads_enabled() -> None:
    config = _load_section(TrainConfig, {"enabled": False})

    assert not config.enabled


def test_pipeline_config_loads_per_model_training_enabled(tmp_path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text(
        json.dumps(
            {
                "trainingMode": "complete",
                "placementTrain": {"enabled": False},
                "stateTrain": {"enabled": True},
            }
        )
    )

    config = _load_config(path)

    assert config.placement_train is not None
    assert not config.placement_train.enabled
    assert config.state_train is not None
    assert config.state_train.enabled


def test_pipeline_config_loads_nested_promotion_defaults_and_overrides(tmp_path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text(
        json.dumps(
            {
                "trainingMode": "complete",
                "promotion": {
                    "enabled": True,
                    "additionalTrainingGames": 250,
                    "direct": {"minimumImprovementVsGreedy": 0.02},
                }
            }
        )
    )

    config = _load_config(path)

    assert config.promotion.enabled
    assert config.promotion.additional_training_games == 250
    assert config.promotion.direct.games == 10_000
    assert config.promotion.direct.minimum_improvement_vs_greedy == 0.02
    assert config.promotion.hybrid.games == 1_000


def test_promotion_gate_requires_both_thresholds() -> None:
    gate = PromotionGateConfig(
        minimum_improvement_vs_greedy=0.02,
        minimum_improvement_vs_champion=0.01,
    )

    assert _evaluate_promotion_gate(gate, 0.52, 0.51)["passed"]
    assert not _evaluate_promotion_gate(gate, 0.519, 0.60)["passed"]
    assert not _evaluate_promotion_gate(gate, 0.60, 0.509)["passed"]


def test_promotion_benchmark_score_counts_draws_and_requires_exact_games() -> None:
    result = {
        "totalGames": 10,
        "draws": 2,
        "winRates": [{"label": "challenger", "wins": 5, "rate": 0.5}],
    }

    assert _benchmark_score(result, "challenger", 10) == 0.6
    with pytest.raises(RuntimeError, match="10/11"):
        _benchmark_score(result, "challenger", 11)


def test_promotion_generation_completion_uses_terminal_decision(tmp_path) -> None:
    cfg = PipelineConfig(
        training_mode="complete",
        model_dir=str(tmp_path / "models"),
        results_dir=str(tmp_path / "results"),
    )
    cfg.promotion.enabled = True

    assert not _generation_complete(cfg, 2)
    decision = tmp_path / "results/promotion/gen2/generation.json"
    decision.parent.mkdir(parents=True)
    decision.write_text('{"status":"rejected"}')
    assert _generation_complete(cfg, 2)


def test_pipeline_defaults_to_complete_promotion_mode(tmp_path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps({"promotion": {"enabled": True}}))

    assert _load_config(path).training_mode == "complete"


def test_train_config_loads_policy_target_temperature() -> None:
    config = _load_section(TrainConfig, {"policyTargetTemperature": 0.5})

    assert config.policy_target_temperature == 0.5


def test_train_config_loads_value_blending_and_state_sampling() -> None:
    config = _load_section(
        TrainConfig,
        {
            "mctsValueWeightStart": 0.75,
            "mctsValueWeightEnd": 0.25,
            "victoryPointSamplingStatistic": "average",
            "victoryPointSamplingUpperPercentage": 0.25,
        },
    )

    assert config.mcts_value_weight_start == 0.75
    assert config.mcts_value_weight_end == 0.25
    assert config.victory_point_sampling_statistic == "average"
    assert config.victory_point_sampling_upper_percentage == 0.25


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


def test_simulate_config_has_combined_budgets() -> None:
    config = _load_section(
        SimulateConfig,
        {"placementSearchTimeMs": 16000, "mainGameSearchTimeMs": 8000},
    )

    assert config.placement_search_time_ms == 16000
    assert config.main_game_search_time_ms == 8000


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
    assert config["greedyPrior"] is True


def test_step_train_passes_value_blending_and_state_sampling(tmp_path, monkeypatch) -> None:
    cfg = PipelineConfig(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        train=TrainConfig(
            mcts_value_weight_start=0.75,
            mcts_value_weight_end=0.25,
            victory_point_sampling_statistic="average",
            victory_point_sampling_upper_percentage=0.25,
        ),
    )
    monkeypatch.setattr("gimbur_nn.pipeline._run", lambda *args, **kwargs: None)

    _step_train(cfg, 0, tmp_path)

    config = json.loads((tmp_path / "models/.configs/train_gen0.json").read_text())
    assert config["mctsValueWeightStart"] == 0.75
    assert config["mctsValueWeightEnd"] == 0.25
    assert config["victoryPointSamplingStatistic"] == "average"
    assert config["victoryPointSamplingUpperPercentage"] == 0.25
    assert "policyTargetTemperature" not in config


def test_complete_mode_generates_shared_sim_and_one_combined_train_config(
    tmp_path, monkeypatch
) -> None:
    cfg = PipelineConfig(
        training_mode="complete",
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        state_model_config="state-small",
        simulate=SimulateConfig(games=1),
        train=TrainConfig(batch_size=22, mcts_value_weight_start=0.7, output_mode="combined"),
    )
    monkeypatch.setattr(
        "gimbur_nn.pipeline.subprocess.run",
        lambda *args, **kwargs: type("R", (), {"returncode": 0})(),
    )
    monkeypatch.setattr("gimbur_nn.pipeline._run", lambda *args, **kwargs: None)

    _step_simulate(cfg, 0, tmp_path, None)
    _step_train(cfg, 0, tmp_path, model_type="state")

    sim = json.loads((tmp_path / "models/.configs/simulate_gen0.json").read_text())
    state = json.loads((tmp_path / "models/.configs/train_gen0_state.json").read_text())
    assert sim["exportType"] == "PlacementAndState"
    assert sim["placementSearchTimeMs"] == 16000
    assert sim["mainGameSearchTimeMs"] == 8000
    assert state["data"] == [str(tmp_path / "data/gen0")]
    assert state["modelConfig"] == "state-small"
    assert state["batchSize"] == 22
    assert state["mctsValueWeightStart"] == 0.7
    assert state["outputMode"] == "combined"


def test_complete_mode_resume_requires_shared_data_and_model(tmp_path) -> None:
    cfg = PipelineConfig(
        training_mode="complete",
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        results_dir=str(tmp_path / "results"),
        simulate=SimulateConfig(games=1),
        benchmarks=[],
    )
    data = tmp_path / "data/gen0"
    data.mkdir(parents=True)
    (data / "game.json").write_text("{}")
    (tmp_path / "models/complete").mkdir(parents=True)

    assert not _generation_complete(cfg, 0)
    (tmp_path / "models/complete/gen0.pt").write_text("")
    assert _generation_complete(cfg, 0)


class _UnusedServer:
    def start(self, **kwargs) -> None:
        raise AssertionError("server should not start")

    def stop(self) -> None:
        raise AssertionError("server should not stop")


def test_placement_and_state_disabled_model_copies_previous_checkpoint(
    tmp_path, monkeypatch
) -> None:
    cfg = PipelineConfig(
        training_mode="placement-and-state",
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        benchmarks=[],
        simulate=SimulateConfig(games=1),
        placement_train=TrainConfig(enabled=False),
        state_train=TrainConfig(),
    )
    data = tmp_path / "data/gen1"
    data.mkdir(parents=True)
    (data / "game.json").write_text("{}")
    previous = tmp_path / "models/placement/gen0.pt"
    previous.parent.mkdir(parents=True)
    previous.write_bytes(b"frozen placement")
    trained: list[str] = []

    def fake_train(cfg, gen, project_root, model_type=None, sim_override=None) -> None:
        trained.append(model_type)
        destination = tmp_path / f"models/{model_type}/gen{gen}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"trained")

    monkeypatch.setattr("gimbur_nn.pipeline._step_train", fake_train)

    _run_placement_and_state_generation(cfg, 1, tmp_path, _UnusedServer(), "unused", {})

    assert (tmp_path / "models/placement/gen1.pt").read_bytes() == b"frozen placement"
    assert trained == ["state"]
    assert not (tmp_path / "models/.configs").exists()
    assert not (tmp_path / "models/placement/gen1_checkpoints").exists()


def test_placement_and_state_disabled_gen0_without_checkpoint_errors(tmp_path, monkeypatch) -> None:
    cfg = PipelineConfig(
        training_mode="placement-and-state",
        model_dir=str(tmp_path / "models"),
        placement_train=TrainConfig(enabled=False),
        benchmarks=[],
    )
    monkeypatch.setattr(
        "gimbur_nn.pipeline._step_simulate",
        lambda *args, **kwargs: pytest.fail("simulation should not run"),
    )

    with pytest.raises(FileNotFoundError, match="generation 0.*Seed that path"):
        _run_placement_and_state_generation(cfg, 0, tmp_path, _UnusedServer(), "unused", {})


def test_placement_and_state_enabled_models_still_train(tmp_path, monkeypatch) -> None:
    cfg = PipelineConfig(
        training_mode="placement-and-state",
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        benchmarks=[],
        simulate=SimulateConfig(games=1),
    )
    data = tmp_path / "data/gen0"
    data.mkdir(parents=True)
    (data / "game.json").write_text("{}")
    trained: list[str] = []

    def fake_train(cfg, gen, project_root, model_type=None, sim_override=None) -> None:
        trained.append(model_type)
        destination = tmp_path / f"models/{model_type}/gen{gen}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"trained")

    monkeypatch.setattr("gimbur_nn.pipeline._step_train", fake_train)

    _run_placement_and_state_generation(cfg, 0, tmp_path, _UnusedServer(), "unused", {})

    assert trained == ["placement", "state"]


def test_summary_preserves_confidence_metadata(tmp_path) -> None:
    cfg = PipelineConfig(results_dir=str(tmp_path))
    results = {
        0: {
            "hybrid": {
                "totalGames": 10000,
                "draws": 0,
                "winRates": [
                    {
                        "ai": "nn-placement-state",
                        "wins": 5100,
                        "rate": 0.51,
                        "confidence95Margin": 0.009796,
                        "worstCaseConfidence95Margin": 0.0098,
                    }
                ],
            }
        }
    }

    _save_summary(cfg, results)

    benchmark = json.loads((tmp_path / "summary.json").read_text())[0]["benchmarks"]["hybrid"]
    assert benchmark["confidence95Margin"]["nn-placement-state"] == 0.009796
    assert benchmark["worstCaseConfidence95Margin"]["nn-placement-state"] == 0.0098


def test_benchmark_config_loads_parallelism() -> None:
    config = _load_section(BenchmarkConfig, {"parallelism": 12})

    assert config.parallelism == 12


def test_progress_chart_accepts_confidence_error_bars(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    cfg = PipelineConfig(
        results_dir=str(tmp_path),
        benchmarks=[BenchmarkConfig(name="hybrid", ai=["nn-placement-state", "greedy"])],
    )
    results = {
        0: {
            "hybrid": {
                "winRates": {"nn-placement-state": 0.51, "greedy": 0.49},
                "confidence95Margin": {"nn-placement-state": 0.0098},
            }
        }
    }

    assert _save_progress_chart(cfg, results)
    assert (tmp_path / "progress.png").is_file()
