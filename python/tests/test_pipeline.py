from __future__ import annotations

from gimbur_nn.pipeline import TrainConfig, _load_section


def test_train_config_loads_replay_window() -> None:
    config = _load_section(TrainConfig, {"replayGenerations": 5})

    assert config.replay_generations == 5
