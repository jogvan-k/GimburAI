"""Tests for serving contracts and checkpoint metadata."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import torch
from test_data_loader import MINI_BOARD, MINI_STATE_ONLY

from gimbur_nn.game_config import MINI_2P
from gimbur_nn.placement_tokenizer import PlacementTokenizer
from gimbur_nn.serve import (
    LeafCancelRequest,
    LeafEnqueueRequest,
    LeafRequest,
    LeafResponseItem,
    PlacementPriorRequest,
    PredictPlacementRequest,
    PredictPlacementResponse,
    PredictPlayerRequest,
    PredictResponse,
    PriorQueue,
    PriorRequest,
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
    state = PredictResponse(
        player_win_probabilities=[[0.25, 0.75]],
        policy_probabilities=[[0.4, 0.6]],
    )
    placement = PredictPlacementResponse(
        player_win_probabilities=[[0.25, 0.75]],
        policy_probabilities=[[0.4, 0.6]],
    )

    assert set(state.model_dump()) == {"player_win_probabilities", "policy_probabilities"}
    assert set(placement.model_dump()) == {
        "player_win_probabilities",
        "policy_probabilities",
    }


def test_placement_predict_returns_fixed_policy_width() -> None:
    class FixedPlacementModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
            batch_size = tokens.shape[0]
            return {
                "value": torch.zeros(batch_size, MINI_2P.player_count),
                "policy": torch.zeros(batch_size, MINI_2P.placement_policy_size),
            }

    state = (
        "w5lb3ls4lW3hd0nW4ho2l|gsgbgw|a|"
        "._._._._._._._._._._._._._._._._._._._._._._._._|"
        "______________________________"
    )
    app = create_app(
        placement_model=FixedPlacementModel(),
        placement_device=torch.device("cpu"),
        placement_game_cfg=MINI_2P,
    )
    predict = next(route.endpoint for route in app.routes if route.path == "/placement/predict")

    response = asyncio.run(predict(PredictPlacementRequest(states=[state])))

    assert len(response.policy_probabilities[0]) == PlacementTokenizer(MINI_2P).policy_size


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


def test_state_predict_returns_complete_policy_and_canonical_value() -> None:
    class FixedModel(torch.nn.Module):
        def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "value": torch.tensor([[2.0, 0.0]] * tokens.shape[0]),
                "policy": torch.zeros(tokens.shape[0], MINI_2P.policy_size),
            }

    app = create_app(
        state_model=FixedModel(),
        state_device=torch.device("cpu"),
        state_game_cfg=MINI_2P,
        state_output_mode="combined",
    )
    predict = next(route.endpoint for route in app.routes if route.path == "/state/predict")

    request = type("Request", (), {"states": [MINI_BOARD + "|" + MINI_STATE_ONLY]})()
    response = asyncio.run(predict(request))

    assert response.player_win_probabilities[0] == pytest.approx([0.880797, 0.119203])
    assert len(response.policy_probabilities[0]) == MINI_2P.policy_size
    assert sum(response.policy_probabilities[0]) == pytest.approx(1.0)


def test_leaf_predict_batches_requests_and_restores_boundaries() -> None:
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
    predict = next(route.endpoint for route in app.routes if route.path == "/state/leaf-predict")

    result = asyncio.run(
        predict(
            LeafEnqueueRequest(
                requests=[
                    LeafRequest(id="first", states=[state], priority=1),
                    LeafRequest(id="chance", states=[state, state], priority=2),
                ]
            )
        )
    )
    assert model.batch_sizes == [3]
    assert [response.id for response in result.responses] == ["first", "chance"]
    assert [len(response.values) for response in result.responses] == [1, 2]
    assert result.responses[0].values[0] == pytest.approx([0.880797, 0.119203])


def test_leaf_predict_returns_empty_only_for_malformed_request() -> None:
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
    predict = next(route.endpoint for route in app.routes if route.path == "/state/leaf-predict")

    result = asyncio.run(
        predict(
            LeafEnqueueRequest(
                requests=[
                    LeafRequest(id="bad", states=["invalid"], priority=1),
                    LeafRequest(id="valid", states=[state], priority=1),
                ]
            )
        )
    )

    assert model.batch_sizes == [1]
    assert result.responses[0].values == []
    assert result.responses[1].values[0] == pytest.approx([0.880797, 0.119203])


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


def test_leaf_cancel_endpoint_reports_removed_counts() -> None:
    app = create_app(
        state_model=torch.nn.Identity(),
        state_device=torch.device("cpu"),
        state_game_cfg=MINI_2P,
    )
    enqueue = next(route.endpoint for route in app.routes if route.path == "/state/leaf-enqueue")
    cancel = next(route.endpoint for route in app.routes if route.path == "/state/leaf-cancel")
    asyncio.run(
        enqueue(
            LeafEnqueueRequest(requests=[LeafRequest(id="cancel", states=["state"], priority=1)])
        )
    )

    response = asyncio.run(cancel(LeafCancelRequest(ids=["cancel", "unknown"])))

    assert response.removed_queued == 1
    assert response.removed_results == 0


def test_leaf_queue_cancellation_removes_queued_request() -> None:
    queue: PriorQueue[LeafRequest] = PriorQueue()
    queue.enqueue(LeafRequest(id="keep", states=["state"], priority=2))
    queue.enqueue(LeafRequest(id="cancel", states=["state"], priority=1))

    assert queue.cancel({"cancel"}) == (1, 0)
    assert [request.id for request in queue.dequeue_batch(2)] == ["keep"]


def test_leaf_queue_cancellation_removes_completed_result() -> None:
    queue: PriorQueue[LeafRequest] = PriorQueue()
    queue.add_results([LeafResponseItem(id="cancel", values=[[0.5, 0.5]])])

    assert queue.cancel({"cancel"}) == (0, 1)
    assert queue.collect_results() == []


def test_leaf_queue_cancellation_suppresses_in_flight_result() -> None:
    queue: PriorQueue[LeafRequest] = PriorQueue()
    queue.enqueue(LeafRequest(id="cancel", states=["state"], priority=1))
    assert [request.id for request in queue.dequeue_batch(1)] == ["cancel"]

    assert queue.cancel({"cancel"}) == (0, 0)
    queue.add_results([LeafResponseItem(id="cancel", values=[[0.5, 0.5]])])

    assert queue.collect_results() == []


def test_full_state_prior_request_is_parent_state_only() -> None:
    request = PriorRequest(id="node", parent_state="state", priority=1)

    assert set(request.model_dump()) == {"id", "parent_state", "priority"}


@pytest.mark.parametrize("architecture", ["catan_policy_value_v1", "placement_stage_policy"])
def test_load_checkpoint_accepts_current_architecture(tmp_path: Path, architecture: str) -> None:
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {},
            **({"checkpoint_version": 5} if architecture == "catan_policy_value_v1" else {}),
            "architecture": architecture,
            "output_mode": "combined",
        },
        path,
    )

    checkpoint = _load_checkpoint(path, torch.device("cpu"), architecture)

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
        _load_checkpoint(path, torch.device("cpu"), "catan_policy_value_v1")
