"""Tests for serving contracts and checkpoint metadata."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
from test_data_loader import MINI_BOARD, MINI_STATE_ONLY

from gimbur_nn.game_config import MINI_2P
from gimbur_nn.serve import (
    LeafEnqueueRequest,
    LeafRequest,
    PlacementPriorRequest,
    PredictPlacementRequest,
    PredictPlacementResponse,
    PredictPlayerRequest,
    PredictResponse,
    PriorResponseItem,
    _load_checkpoint,
    create_app,
)
from gimbur_nn.state_tokenizer import StateTokenizer


def test_placement_requests_are_state_only() -> None:
    predict = PredictPlacementRequest(states=["state"])
    prior = PlacementPriorRequest(id="node", state="state", priority=2)

    assert predict.states == ["state"]
    assert set(prior.model_dump()) == {"id", "state", "priority"}


def test_prediction_responses_use_player_probabilities() -> None:
    state = PredictResponse(player_win_probabilities=[[0.25, 0.75]])
    placement = PredictPlacementResponse(
        player_win_probabilities=[[0.25, 0.75]],
        policy_probabilities=[[0.4, 0.6]],
    )

    assert set(state.model_dump()) == {"player_win_probabilities"}
    assert set(placement.model_dump()) == {
        "player_win_probabilities",
        "policy_probabilities",
    }


def test_prior_response_can_carry_full_player_distribution() -> None:
    response = PriorResponseItem(
        id="node",
        priors=[0.6],
        value_estimate=0.6,
        player_win_probabilities=[0.6, 0.4],
    )

    assert response.player_win_probabilities == [0.6, 0.4]


def test_predict_player_gathers_from_one_unrotated_vector_inference(monkeypatch) -> None:
    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            assert tokens.shape[0] == 2
            return torch.tensor([[2.0, 0.0], [2.0, 0.0]])

    model = FixedModel()
    monkeypatch.setattr(
        StateTokenizer,
        "rotate_player_state",
        lambda *args: pytest.fail("predict-player must not rotate states"),
    )
    app = create_app(
        state_model=model,
        state_device=torch.device("cpu"),
        state_game_cfg=MINI_2P,
    )
    state = MINI_BOARD + "|" + MINI_STATE_ONLY

    endpoint = next(route.endpoint for route in app.routes if route.path == "/state/predict-player")
    response = asyncio.run(endpoint(PredictPlayerRequest(states=[state, state], players=[1, 2])))

    assert model.calls == 1
    assert response.win_probabilities == pytest.approx([0.880797, 0.119203])


def test_leaf_queue_batches_all_outcomes_into_one_model_invocation() -> None:
    class FixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(tokens.shape[0])
            return torch.tensor([[2.0, 0.0]] * tokens.shape[0])

    model = FixedModel()
    app = create_app(
        state_model=model,
        state_device=torch.device("cpu"),
        state_game_cfg=MINI_2P,
    )
    state = MINI_BOARD + "|" + MINI_STATE_ONLY
    enqueue = next(route.endpoint for route in app.routes if route.path == "/state/leaf-enqueue")
    collect = next(route.endpoint for route in app.routes if route.path == "/state/leaf-collect")

    async def run() -> object:
        async with app.router.lifespan_context(app):
            response = await enqueue(
                LeafEnqueueRequest(
                    requests=[LeafRequest(id="chance", states=[state, state, state], priority=1)]
                )
            )
            assert response.status_code == 202
            for _ in range(100):
                result = await collect()
                if result.responses:
                    return result
                await asyncio.sleep(0.005)
            pytest.fail("leaf response did not arrive")

    result = asyncio.run(run())
    assert model.batch_sizes == [3]
    assert result.responses[0].id == "chance"
    assert len(result.responses[0].values) == 3
    assert result.responses[0].values[0] == pytest.approx([0.880797, 0.119203])


def test_leaf_enqueue_reports_accepted_request_ids() -> None:
    app = create_app(
        state_model=torch.nn.Identity(),
        state_device=torch.device("cpu"),
        state_game_cfg=MINI_2P,
    )
    enqueue = next(route.endpoint for route in app.routes if route.path == "/state/leaf-enqueue")

    response = asyncio.run(
        enqueue(
            LeafEnqueueRequest(
                requests=[
                    LeafRequest(id="first", states=["state"], priority=1),
                    LeafRequest(id="second", states=["state"], priority=2),
                ]
            )
        )
    )

    assert json.loads(response.body) == {
        "accepted": 2,
        "dropped": 0,
        "accepted_ids": ["first", "second"],
        "dropped_ids": [],
    }


def test_leaf_enqueue_reports_dropped_request_id() -> None:
    app = create_app(
        state_model=torch.nn.Identity(),
        state_device=torch.device("cpu"),
        state_game_cfg=MINI_2P,
    )
    enqueue = next(route.endpoint for route in app.routes if route.path == "/state/leaf-enqueue")

    async def fill_and_drop() -> object:
        await enqueue(
            LeafEnqueueRequest(
                requests=[
                    LeafRequest(id=f"queued-{i}", states=["state"], priority=10)
                    for i in range(4096)
                ]
            )
        )
        return await enqueue(
            LeafEnqueueRequest(requests=[LeafRequest(id="urgent", states=["state"], priority=0)])
        )

    response = asyncio.run(fill_and_drop())

    assert json.loads(response.body) == {
        "accepted": 0,
        "dropped": 1,
        "accepted_ids": [],
        "dropped_ids": ["urgent"],
    }


@pytest.mark.parametrize("architecture", ["state_player_value_v1", "placement_state_v3"])
def test_load_checkpoint_accepts_v3_metadata(tmp_path: Path, architecture: str) -> None:
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {},
            "checkpoint_version": 3,
            "architecture": architecture,
            "output_mode": "combined",
        },
        path,
    )

    checkpoint = _load_checkpoint(path, torch.device("cpu"), architecture)

    assert checkpoint["checkpoint_version"] == 3
    assert checkpoint["architecture"] == architecture


@pytest.mark.parametrize(
    ("version", "architecture"),
    [(2, "placement_state_v2"), (2, "state"), (3, "placement_state_v3")],
)
def test_load_checkpoint_rejects_old_or_wrong_architecture(
    tmp_path: Path, version: int, architecture: str
) -> None:
    path = tmp_path / "old.pt"
    torch.save(
        {
            "model_state_dict": {},
            "checkpoint_version": version,
            "architecture": architecture,
        },
        path,
    )

    with pytest.raises(ValueError, match="incompatible checkpoint"):
        _load_checkpoint(path, torch.device("cpu"), "state_player_value_v1")
