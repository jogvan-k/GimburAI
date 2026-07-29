"""Tests for placement serving contracts and checkpoint metadata."""

from __future__ import annotations

from pathlib import Path

import torch

from gimbur_nn.serve import (
    PlacementPriorRequest,
    PredictPlacementRequest,
    PredictPlacementResponse,
    _load_checkpoint,
)


def test_placement_requests_are_state_only() -> None:
    predict = PredictPlacementRequest(states=["state"])
    prior = PlacementPriorRequest(id="node", state="state", priority=2)

    assert predict.states == ["state"]
    assert prior.state == "state"
    assert set(prior.model_dump()) == {"id", "state", "priority"}


def test_placement_response_uses_value_probabilities_contract() -> None:
    response = PredictPlacementResponse(
        value_probabilities=[[0.25, 0.75]],
        policy_probabilities=[[0.4, 0.6]],
    )

    assert set(response.model_dump()) == {"value_probabilities", "policy_probabilities"}


def test_load_checkpoint_preserves_v2_metadata(tmp_path: Path) -> None:
    path = tmp_path / "placement.pt"
    torch.save(
        {
            "model_state_dict": {},
            "checkpoint_version": 2,
            "architecture": "placement_state_v2",
            "output_mode": "combined",
        },
        path,
    )

    checkpoint = _load_checkpoint(path, torch.device("cpu"))

    assert checkpoint["checkpoint_version"] == 2
    assert checkpoint["architecture"] == "placement_state_v2"
    assert checkpoint["output_mode"] == "combined"
