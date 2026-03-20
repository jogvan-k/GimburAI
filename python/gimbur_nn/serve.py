"""
Inference server for GimburNet.

Loads a trained model checkpoint and exposes an HTTP endpoint that
accepts serialized board+state strings and returns win probability
predictions.  The Kjarni MCTS engine can call this endpoint as a
learned leaf evaluator.

Usage::

    python -m gimbur_nn.serve \
        --model model.pt \
        --game-config mini_2p \
        --model-config small \
        --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import logging
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .game_config import CONFIGS_BY_NAME, GameConfig
from .state_tokenizer import StateTokenizer
from .transformer_model import (
    MODEL_CONFIGS_BY_NAME,
    GimburTransformer,
)

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint."""

    states: list[str]
    """List of serialized game state strings (compact or human-readable)."""


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""

    probabilities: list[list[float]]
    """Per-state bucket probabilities, shape (n_states, n_buckets)."""


class PredictPlayerRequest(BaseModel):
    """Request body for the /predict-player endpoint."""

    states: list[str]
    """Compact serialized game state strings."""

    players: list[int]
    """1-based target player for each state.  The state is rotated so
    that this player becomes player 1 before inference."""


class PredictPlayerResponse(BaseModel):
    """Response body for the /predict-player endpoint."""

    win_probabilities: list[float]
    """Scalar expected win probability for each target player."""


# ── Prior queue models ────────────────────────────────────────────────────────


class PriorRequest(BaseModel):
    """A single prior request for one tree node."""

    id: str
    """Opaque ID to correlate response back to the MCTSState."""

    states: list[str]
    """Serialized result states for each action (deterministic: 1 state,
    stochastic: 1 per outcome)."""

    player: int
    """1-based acting player for server-side rotation."""

    priority: int
    """Depth from root; lower = more important."""


class PriorEnqueueRequest(BaseModel):
    """Request body for /prior-enqueue."""

    requests: list[PriorRequest]


class PriorResponseItem(BaseModel):
    """A completed prior inference result for one tree node."""

    id: str
    win_probabilities: list[float]


class PriorCollectResponse(BaseModel):
    """Response body for /prior-collect."""

    responses: list[PriorResponseItem]


# ── Priority queue for async prior inference ──────────────────────────────────

_PRIOR_QUEUE_CAPACITY = 4096


class PriorQueue:
    """Thread-safe priority queue for prior requests, with bounded capacity.

    Requests are ordered by priority (depth from root, lower = higher
    priority).  When the queue is full, a new request with lower priority
    than the worst queued item is silently dropped; one with higher
    priority evicts the lowest-priority entry.
    """

    def __init__(self, capacity: int = _PRIOR_QUEUE_CAPACITY) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        # Min-heap of (priority, sequence_no, PriorRequest).
        # sequence_no breaks ties to maintain FIFO within the same priority.
        self._heap: list[tuple[int, int, PriorRequest]] = []
        self._seq = 0
        # Completed results waiting to be collected.
        self._results: list[PriorResponseItem] = []

    def enqueue(self, req: PriorRequest) -> bool:
        """Add a request.  Returns True if accepted, False if dropped."""
        with self._lock:
            if len(self._heap) < self._capacity:
                heapq.heappush(self._heap, (req.priority, self._seq, req))
                self._seq += 1
                return True
            # Queue full — check if the new request is higher priority
            # than the worst (highest priority value) in the heap.
            # Since this is a min-heap, the worst is *not* at index 0.
            # We maintain a min-heap by priority, so the *largest* priority
            # is the one we'd want to evict.  Using nlargest(1) is O(n) but
            # the queue is bounded so this is fine.
            worst_priority = max(item[0] for item in self._heap)
            if req.priority < worst_priority:
                # Evict the worst entry.
                # Find and remove the worst (largest priority, latest seq).
                worst_idx = max(
                    range(len(self._heap)), key=lambda i: (self._heap[i][0], self._heap[i][1])
                )
                self._heap[worst_idx] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                heapq.heappush(self._heap, (req.priority, self._seq, req))
                self._seq += 1
                return True
            # New request is lower priority — drop it.
            return False

    def dequeue_batch(self, batch_size: int) -> list[PriorRequest]:
        """Remove and return up to *batch_size* highest-priority requests."""
        with self._lock:
            batch: list[PriorRequest] = []
            for _ in range(min(batch_size, len(self._heap))):
                _, _, req = heapq.heappop(self._heap)
                batch.append(req)
            return batch

    def add_results(self, results: list[PriorResponseItem]) -> None:
        """Add completed inference results to the collection buffer."""
        with self._lock:
            self._results.extend(results)

    def collect_results(self) -> list[PriorResponseItem]:
        """Drain and return all completed results."""
        with self._lock:
            results = self._results
            self._results = []
            return results

    def flush(self) -> None:
        """Clear the queue and all pending results."""
        with self._lock:
            self._heap.clear()
            self._results.clear()
            self._seq = 0

    def pending_count(self) -> int:
        with self._lock:
            return len(self._heap)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GimburNet for inference.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--game-config",
        type=str,
        required=True,
        choices=sorted(CONFIGS_BY_NAME),
        help="Game configuration preset (must match the checkpoint).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        choices=sorted(MODEL_CONFIGS_BY_NAME),
        help="Model size preset (must match the checkpoint).",
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level. Use 'warning' to suppress HTTP 200/202 access logs.",
    )
    return parser.parse_args()


def create_app(
    model: GimburTransformer,
    device: torch.device,
    game_cfg: GameConfig,
) -> FastAPI:
    """Build the FastAPI application with the model captured in closure."""
    tokenizer = StateTokenizer(game_cfg)
    app = FastAPI(title="GimburNet Inference")
    prior_queue = PriorQueue()

    def _expected_win_prob(probs: torch.Tensor) -> float:
        """Compute expected win probability from a 1-D bucket distribution."""
        n = probs.shape[0]
        centres = torch.arange(n, dtype=probs.dtype, device=probs.device)
        centres = (centres + 0.5) / n
        return float((probs * centres).sum())

    def _infer_prior_batch(batch: list[PriorRequest]) -> list[PriorResponseItem]:
        """Run inference on a batch of prior requests and return results."""
        results: list[PriorResponseItem] = []
        for req in batch:
            if not req.states:
                results.append(PriorResponseItem(id=req.id, win_probabilities=[]))
                continue
            try:
                rotated = [tokenizer.rotate_player_state(s, req.player) for s in req.states]
                token_ids = tokenizer.tokenize_batch(rotated).to(device)
            except (KeyError, ValueError):
                # Bad state — return zeros so the MCTS falls back to uniform.
                results.append(
                    PriorResponseItem(
                        id=req.id,
                        win_probabilities=[0.0] * len(req.states),
                    )
                )
                continue

            with torch.no_grad():
                logits = model(token_ids)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)

            win_probs = [_expected_win_prob(probs[i]) for i in range(probs.shape[0])]
            results.append(PriorResponseItem(id=req.id, win_probabilities=win_probs))

        return results

    async def _prior_worker() -> None:
        """Background task that continuously processes the prior queue."""
        batch_size = 8
        while True:
            batch = prior_queue.dequeue_batch(batch_size)
            if batch:
                # Run inference in a thread pool to avoid blocking the event loop.
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(None, _infer_prior_batch, batch)
                prior_queue.add_results(results)
            else:
                # No work — sleep briefly to avoid busy-waiting.
                await asyncio.sleep(0.005)

    # Store background task reference to prevent garbage collection.
    _background_tasks: set[asyncio.Task[None]] = set()

    @app.on_event("startup")
    async def start_prior_worker() -> None:
        task = asyncio.create_task(_prior_worker())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        if not request.states:
            raise HTTPException(status_code=400, detail="states list must not be empty")

        try:
            token_ids = tokenizer.tokenize_batch(request.states).to(device)
        except (KeyError, ValueError) as exc:
            logger.error(
                "Tokenization failed in /predict: %s\n  First state (truncated): %.200s",
                exc,
                request.states[0] if request.states else "<empty>",
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with torch.no_grad():
            logits = model(token_ids)  # (batch, seq_len, n_buckets)
            last_logits = logits[:, -1, :]  # (batch, n_buckets)
            probs = F.softmax(last_logits, dim=-1)

        return PredictResponse(probabilities=probs.cpu().tolist())

    @app.post("/predict-player", response_model=PredictPlayerResponse)
    async def predict_player(
        request: PredictPlayerRequest,
    ) -> PredictPlayerResponse:
        """Predict win probability for specific target players.

        Each state is rotated so that the corresponding target player
        becomes player 1, then the model's player-1 win probability is
        returned as a scalar expected value.
        """
        if not request.states:
            raise HTTPException(
                status_code=400,
                detail="states list must not be empty",
            )
        if len(request.states) != len(request.players):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"states ({len(request.states)}) and players "
                    f"({len(request.players)}) must have the same length"
                ),
            )

        try:
            rotated = [
                tokenizer.rotate_player_state(s, p) for s, p in zip(request.states, request.players)
            ]
            token_ids = tokenizer.tokenize_batch(rotated).to(device)
        except (KeyError, ValueError) as exc:
            logger.error(
                "Tokenization failed in /predict-player: %s\n"
                "  Players: %s\n  First state (truncated): %.200s",
                exc,
                request.players[:5],
                request.states[0] if request.states else "<empty>",
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        with torch.no_grad():
            logits = model(token_ids)
            last_logits = logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)

        win_probs = [_expected_win_prob(probs[i]) for i in range(probs.shape[0])]
        return PredictPlayerResponse(win_probabilities=win_probs)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ── Prior endpoints ───────────────────────────────────────────────────────

    @app.post("/prior-enqueue", status_code=202)
    async def prior_enqueue(request: PriorEnqueueRequest) -> JSONResponse:
        """Accept a batch of prior requests into the priority queue."""
        accepted = 0
        dropped = 0
        for req in request.requests:
            if prior_queue.enqueue(req):
                accepted += 1
            else:
                dropped += 1
        return JSONResponse(
            status_code=202,
            content={"accepted": accepted, "dropped": dropped},
        )

    @app.post("/prior-collect", response_model=PriorCollectResponse)
    async def prior_collect() -> PriorCollectResponse:
        """Return all completed prior inference results."""
        results = prior_queue.collect_results()
        return PriorCollectResponse(responses=results)

    @app.post("/prior-flush")
    async def prior_flush() -> dict[str, str]:
        """Clear the priority queue and discard pending results."""
        prior_queue.flush()
        return {"status": "flushed"}

    return app


def main() -> None:
    args = parse_args()

    game_cfg = CONFIGS_BY_NAME[args.game_config]
    model_cfg = MODEL_CONFIGS_BY_NAME[args.model_config]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GimburTransformer(game_cfg, model_cfg)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(
        f"Loaded {args.model_config} model for {args.game_config} "
        f"({param_count:,} parameters) on {device}"
    )

    app = create_app(model, device, game_cfg)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.log_level in ("debug", "info"),
    )


if __name__ == "__main__":
    main()
