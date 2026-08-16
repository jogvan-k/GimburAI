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
    _PRIOR_QUEUE_CAPACITY,
    InferenceDiagnostics,
    LeafCancelRequest,
    LeafEnqueueRequest,
    LeafRequest,
    LeafResponseItem,
    PredictPlayerRequest,
    PredictResponse,
    PriorQueue,
    PriorRequest,
    PriorResponseItem,
    _checkpoint_precision,
    _dequeue_with_window,
    _legal_policy_softmax,
    _load_checkpoint,
    create_app,
)
from gimbur_nn.state_tokenizer import StateTokenizer


def test_prediction_responses_use_player_probabilities() -> None:
    state = PredictResponse(
        player_win_probabilities=[[0.25, 0.75]],
        policy_probabilities=[[0.4, 0.6]],
    )
    assert set(state.model_dump()) == {"player_win_probabilities", "policy_probabilities"}


def test_checkpoint_precision_defaults_to_fp32_and_accepts_fp16() -> None:
    assert _checkpoint_precision({}) == "fp32"
    assert _checkpoint_precision({"inference_precision": "fp16"}) == "fp16"
    with pytest.raises(ValueError, match="unsupported inference precision"):
        _checkpoint_precision({"inference_precision": "int8"})


def test_prior_response_can_carry_full_player_distribution() -> None:
    response = PriorResponseItem(
        id="node",
        priors=[0.6],
        value_estimate=0.6,
        player_win_probabilities=[0.6, 0.4],
    )

    assert response.player_win_probabilities == [0.6, 0.4]


def test_batch_window_collects_request_arriving_after_first_dequeue() -> None:
    queue: PriorQueue[PriorRequest] = PriorQueue()
    queue.enqueue(
        PriorRequest(id="first", parent_state="state", legal_policy_indices=[0], priority=0)
    )

    async def collect() -> list[PriorRequest]:
        async def enqueue_second() -> None:
            await asyncio.sleep(0.0005)
            queue.enqueue(
                PriorRequest(id="second", parent_state="state", legal_policy_indices=[0], priority=0)
            )

        producer = asyncio.create_task(enqueue_second())
        batch = await _dequeue_with_window(queue, batch_size=32, batch_window_ms=1.5)
        await producer
        return batch

    assert [request.id for request in asyncio.run(collect())] == ["first", "second"]


def test_inference_diagnostics_reports_batch_and_stage_averages() -> None:
    diagnostics = InferenceDiagnostics()
    diagnostics.record(
        states=4,
        queue_wait_ms=8.0,
        tokenize_ms=2.0,
        transfer_ms=1.0,
        forward_ms=5.0,
    )
    diagnostics.record(
        states=2,
        queue_wait_ms=4.0,
        tokenize_ms=1.0,
        transfer_ms=0.5,
        forward_ms=3.0,
    )

    snapshot = diagnostics.snapshot()
    assert snapshot["batches"] == 2
    assert snapshot["states"] == 6
    assert snapshot["average_batch_size"] == 3
    assert snapshot["average_queue_wait_ms"] == 2
    assert snapshot["average_forward_ms"] == 4


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
    )
    predict = next(route.endpoint for route in app.routes if route.path == "/state/predict")

    request = type("Request", (), {"states": [MINI_BOARD + "|" + MINI_STATE_ONLY]})()
    response = asyncio.run(predict(request))

    assert response.player_win_probabilities[0] == pytest.approx([0.880797, 0.119203])
    assert len(response.policy_probabilities[0]) == MINI_2P.policy_size
    assert sum(response.policy_probabilities[0]) == pytest.approx(1.0)


def test_prior_softmax_excludes_illegal_logits_and_preserves_legal_order() -> None:
    logits = torch.tensor([1000.0, 1.0, 0.0])

    probabilities = _legal_policy_softmax(logits, [2, 1])

    assert probabilities.tolist() == pytest.approx([0.268941, 0.731059], abs=1e-6)


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
                        for i in range(_PRIOR_QUEUE_CAPACITY)
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


def test_full_state_prior_request_includes_legal_policy_indices() -> None:
    request = PriorRequest(
        id="node", parent_state="state", legal_policy_indices=[2, 4], priority=1
    )

    assert set(request.model_dump()) == {
        "id",
        "parent_state",
        "legal_policy_indices",
        "priority",
    }


def test_load_checkpoint_accepts_current_architecture(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": {},
            "checkpoint_version": 5,
            "architecture": "catan_policy_value_v1",
        },
        path,
    )

    checkpoint = _load_checkpoint(path, torch.device("cpu"))

    assert checkpoint["architecture"] == "catan_policy_value_v1"


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
        _load_checkpoint(path, torch.device("cpu"))
