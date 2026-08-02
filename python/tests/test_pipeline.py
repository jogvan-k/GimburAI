from __future__ import annotations

from gimbur_nn.pipeline import SimulateConfig, TrainConfig, _load_section


def test_train_config_loads_replay_window() -> None:
    config = _load_section(TrainConfig, {"replayGenerations": 5})

    assert config.replay_generations == 5


def test_simulate_config_loads_parallelism() -> None:
    config = _load_section(SimulateConfig, {"parallelism": 8})

    assert config.parallelism == 8
