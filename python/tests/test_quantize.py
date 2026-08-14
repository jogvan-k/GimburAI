from __future__ import annotations

from pathlib import Path

import torch

from gimbur_nn.quantize import export_fp16_checkpoint


def test_export_fp16_checkpoint_preserves_source_and_metadata(tmp_path: Path) -> None:
    source = tmp_path / "model.pt"
    destination = tmp_path / "model.fp16.pt"
    original = {
        "architecture": "catan_policy_value_v1",
        "checkpoint_version": 5,
        "model_config": "small",
        "game_config": "mini_2p",
        "model_state_dict": {
            "weight": torch.tensor([1.25], dtype=torch.float32),
            "counter": torch.tensor([3], dtype=torch.int64),
        },
    }
    torch.save(original, source)

    export_fp16_checkpoint(source, destination)

    source_checkpoint = torch.load(source, weights_only=False)
    converted = torch.load(destination, weights_only=False)
    assert source_checkpoint["model_state_dict"]["weight"].dtype == torch.float32
    assert converted["model_state_dict"]["weight"].dtype == torch.float16
    assert converted["model_state_dict"]["counter"].dtype == torch.int64
    assert converted["inference_precision"] == "fp16"
    assert converted["source_checkpoint"] == "model.pt"
