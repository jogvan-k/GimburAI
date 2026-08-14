"""
Inference server for GimburNet.

Loads a complete Catan policy/value checkpoint and exposes state parent
policy/value and leaf-value endpoints under ``/state/...``.

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
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
import torch.nn.functional as F
from pydantic import BaseModel, PrivateAttr

if TYPE_CHECKING:
    from fastapi import FastAPI

from .game_config import CONFIGS_BY_NAME, GameConfig
from .state_tokenizer import StateTokenizer
from .transformer_model import MODEL_CONFIGS_BY_NAME, GimburTransformer

logger = logging.getLogger(__name__)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load a checkpoint for the current architecture or reject it."""
    raw = torch.load(path, map_location=device, weights_only=False)
    if (
        not isinstance(raw, dict)
        or "model_state_dict" not in raw
        or raw.get("architecture") != "catan_policy_value_v1"
        or raw.get("checkpoint_version") != 5
    ):
        raise ValueError("incompatible checkpoint; expected architecture='catan_policy_value_v1'")
    return raw


def _checkpoint_precision(checkpoint: dict) -> str:
    precision = checkpoint.get("inference_precision", "fp32")
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"unsupported inference precision {precision!r}")
    return precision


def _extract_logits(
    output: torch.Tensor | dict[str, torch.Tensor], head: str = "value"
) -> torch.Tensor:
    """Select a head from model output containing pooled logits."""
    if isinstance(output, dict):
        return output[head]
    return output


class PredictRequest(BaseModel):
    """Request body for the /state/predict endpoint."""

    states: list[str]
    """List of serialized game state strings (compact or human-readable)."""


class PredictResponse(BaseModel):
    """Response body for the /state/predict endpoint."""

    player_win_probabilities: list[list[float]]
    """Per-state player win distributions."""

    policy_probabilities: list[list[float]]
    """Per-state complete fixed-width policy distributions."""


class PredictPlayerRequest(BaseModel):
    """Request body for the /state/predict-player endpoint."""

    states: list[str]
    """Compact serialized game state strings."""

    players: list[int]
    """1-based target player for each state."""


class PredictPlayerResponse(BaseModel):
    """Response body for the /state/predict-player endpoint."""

    win_probabilities: list[float]
    """Scalar expected win probability for each target player."""


# ── Prior queue models ────────────────────────────────────────────────────────


class PriorRequest(BaseModel):
    """A single prior request for one tree node."""

    id: str
    """Opaque ID to correlate response back to the MCTSState."""

    parent_state: str
    """Serialized parent decision state."""

    priority: int
    """Depth from root; lower = more important."""

    _queued_at_ns: int = PrivateAttr(default=0)


class PriorEnqueueRequest(BaseModel):
    """Request body for /state/prior-enqueue."""

    requests: list[PriorRequest]


class PriorResponseItem(BaseModel):
    """A completed prior inference result for one tree node."""

    id: str
    priors: list[float]
    """Per-action prior weights in the legal-action order supplied by the client."""

    value_estimate: float | None = None
    """Canonical acting-player value estimate."""

    player_win_probabilities: list[float] | None = None
    """Full player value distribution when a value head is available."""


class PriorCollectResponse(BaseModel):
    """Response body for /state/prior-collect."""

    responses: list[PriorResponseItem]


class LeafRequest(BaseModel):
    """One value request; all states are evaluated in one model batch."""

    id: str
    states: list[str]
    priority: int

    _queued_at_ns: int = PrivateAttr(default=0)


class LeafEnqueueRequest(BaseModel):
    requests: list[LeafRequest]


class LeafResponseItem(BaseModel):
    id: str
    values: list[list[float]]


class LeafCollectResponse(BaseModel):
    responses: list[LeafResponseItem]


class LeafCancelRequest(BaseModel):
    ids: list[str]


class LeafCancelResponse(BaseModel):
    removed_queued: int
    removed_results: int


# ── Priority queue for async prior inference ──────────────────────────────────

_PRIOR_QUEUE_CAPACITY = 16384 # 4096

T = TypeVar("T")


class PriorQueue(Generic[T]):
    """Thread-safe priority queue for prior requests, with bounded capacity.

    Requests are ordered by priority (depth from root, lower = higher
    priority).  When the queue is full, a new request with lower priority
    than the worst queued item is silently dropped; one with higher
    priority evicts the lowest-priority entry.

    Generic over request types with ``id`` and ``priority`` attributes.
    """

    def __init__(self, capacity: int = _PRIOR_QUEUE_CAPACITY) -> None:
        self._capacity = capacity
        self._lock = threading.Lock()
        # Min-heap of (priority, sequence_no, request).
        # sequence_no breaks ties to maintain FIFO within the same priority.
        self._heap: list[tuple[int, int, T]] = []
        self._seq = 0
        # Completed results waiting to be collected.
        self._results: list[PriorResponseItem] = []
        self._in_flight: set[str] = set()
        self._cancelled: set[str] = set()

    def enqueue(self, req: T) -> bool:
        """Add a request.  Returns True if accepted, False if dropped."""
        accepted, _ = self.enqueue_with_evicted(req)
        return accepted

    def enqueue_with_evicted(self, req: T) -> tuple[bool, T | None]:
        """Add a request and return any lower-priority request it evicted."""
        priority: int = req.priority  # type: ignore[union-attr]
        req._queued_at_ns = time.perf_counter_ns()  # type: ignore[union-attr]
        with self._lock:
            if len(self._heap) < self._capacity:
                heapq.heappush(self._heap, (priority, self._seq, req))
                self._seq += 1
                return True, None
            # Queue full — check if the new request is higher priority
            # than the worst (highest priority value) in the heap.
            # Since this is a min-heap, the worst is *not* at index 0.
            # We maintain a min-heap by priority, so the *largest* priority
            # is the one we'd want to evict.  Using max() is O(n) but
            # the queue is bounded so this is fine.
            worst_priority = max(item[0] for item in self._heap)
            if priority < worst_priority:
                # Evict the worst entry.
                # Find and remove the worst (largest priority, latest seq).
                worst_idx = max(
                    range(len(self._heap)),
                    key=lambda i: (self._heap[i][0], self._heap[i][1]),
                )
                evicted = self._heap[worst_idx][2]
                self._heap[worst_idx] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                heapq.heappush(self._heap, (priority, self._seq, req))
                self._seq += 1
                return True, evicted
            # New request is lower priority — drop it.
            return False, None

    def enqueue_without_eviction(self, req: T) -> bool:
        """Add a request only when capacity is available."""
        priority: int = req.priority  # type: ignore[union-attr]
        req._queued_at_ns = time.perf_counter_ns()  # type: ignore[union-attr]
        with self._lock:
            if len(self._heap) >= self._capacity:
                return False
            heapq.heappush(self._heap, (priority, self._seq, req))
            self._seq += 1
            return True

    def dequeue_batch(self, batch_size: int) -> list[T]:
        """Remove and return up to *batch_size* highest-priority requests."""
        with self._lock:
            batch: list[T] = []
            for _ in range(min(batch_size, len(self._heap))):
                _, _, req = heapq.heappop(self._heap)
                batch.append(req)
                self._in_flight.add(req.id)  # type: ignore[union-attr]
            return batch

    def add_results(self, results: list[PriorResponseItem]) -> None:
        """Add completed inference results to the collection buffer."""
        with self._lock:
            for result in results:
                self._in_flight.discard(result.id)
                if result.id in self._cancelled:
                    self._cancelled.remove(result.id)
                else:
                    self._results.append(result)

    def cancel(self, ids: set[str]) -> tuple[int, int]:
        """Remove queued/completed IDs and suppress results already in flight."""
        with self._lock:
            queued_before = len(self._heap)
            self._heap = [
                item
                for item in self._heap
                if item[2].id not in ids  # type: ignore[union-attr]
            ]
            removed_queued = queued_before - len(self._heap)
            if removed_queued:
                heapq.heapify(self._heap)

            results_before = len(self._results)
            self._results = [result for result in self._results if result.id not in ids]
            removed_results = results_before - len(self._results)
            self._cancelled.update(ids & self._in_flight)
            return removed_queued, removed_results

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
            self._in_flight.clear()
            self._cancelled.clear()
            self._seq = 0

    def pending_count(self) -> int:
        with self._lock:
            return len(self._heap)


class InferenceDiagnostics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.batches = 0
        self.states = 0
        self.queue_wait_ms = 0.0
        self.tokenize_ms = 0.0
        self.transfer_ms = 0.0
        self.forward_ms = 0.0

    def record(
        self,
        *,
        states: int,
        queue_wait_ms: float,
        tokenize_ms: float,
        transfer_ms: float,
        forward_ms: float,
    ) -> None:
        with self._lock:
            self.batches += 1
            self.states += states
            self.queue_wait_ms += queue_wait_ms
            self.tokenize_ms += tokenize_ms
            self.transfer_ms += transfer_ms
            self.forward_ms += forward_ms

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            batches = self.batches
            states = self.states
            return {
                "batches": batches,
                "states": states,
                "average_batch_size": states / batches if batches else 0.0,
                "average_queue_wait_ms": self.queue_wait_ms / states if states else 0.0,
                "average_tokenize_ms": self.tokenize_ms / batches if batches else 0.0,
                "average_transfer_ms": self.transfer_ms / batches if batches else 0.0,
                "average_forward_ms": self.forward_ms / batches if batches else 0.0,
                "total_queue_wait_ms": self.queue_wait_ms,
                "total_tokenize_ms": self.tokenize_ms,
                "total_transfer_ms": self.transfer_ms,
                "total_forward_ms": self.forward_ms,
            }


async def _dequeue_with_window(
    queue: PriorQueue[T], batch_size: int, batch_window_ms: float
) -> list[T]:
    batch = queue.dequeue_batch(batch_size)
    if not batch or len(batch) >= batch_size or batch_window_ms <= 0:
        return batch
    await asyncio.sleep(batch_window_ms / 1000.0)
    batch.extend(queue.dequeue_batch(batch_size - len(batch)))
    return batch


# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_json_comments(text: str) -> str:
    """Remove single-line // comments from JSON text (outside strings)."""
    lines = []
    for line in text.splitlines():
        in_string = False
        escape = False
        for i, ch in enumerate(line):
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
            elif ch == "/" and not in_string and i + 1 < len(line) and line[i + 1] == "/":
                line = line[:i].rstrip()
                break
        lines.append(line)
    return "\n".join(lines)


_CONFIG_KEY_MAP: dict[str, str] = {
    "model": "model",
    "gameConfig": "game_config",
    "modelConfig": "model_config",
    "port": "port",
    "host": "host",
    "logLevel": "log_level",
    "batchWindowMs": "batch_window_ms",
    "compileModel": "compile_model",
}
"""Maps camelCase JSON config keys to argparse dest names."""


_ARG_DEFAULTS: dict[str, object] = {
    "model": None,
    "game_config": None,
    "model_config": None,
    "port": 8000,
    "host": "127.0.0.1",
    "log_level": "info",
    "batch_window_ms": 0.0,
    "compile_model": False,
}
"""Default values for argparse arguments, used to detect explicit CLI overrides."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GimburNet for inference.")

    parser.add_argument(
        "--model",
        type=Path,
        required=False,
        default=None,
        help="Path to trained state model checkpoint.",
    )
    parser.add_argument(
        "--game-config",
        type=str,
        required=False,
        default=None,
        choices=sorted(CONFIGS_BY_NAME),
        help="Game configuration preset for the state model.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        required=False,
        default=None,
        choices=sorted(MODEL_CONFIGS_BY_NAME),
        help="Model size preset for the state model.",
    )

    # ── Server args ───────────────────────────────────────────────────────
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Uvicorn log level. Use 'warning' to suppress HTTP 200/202 access logs.",
    )
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=0.0,
        help="Milliseconds to collect additional async requests after the first arrives.",
    )
    parser.add_argument(
        "--compile-model",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model with torch.compile before serving.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file with camelCase keys. CLI args override config values.",
    )
    return parser.parse_args()


def create_app(
    state_model: GimburTransformer,
    state_device: torch.device,
    state_game_cfg: GameConfig,
    batch_window_ms: float = 0.0,
) -> FastAPI:
    """Build the full-state inference application."""

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError("create_app requires the optional 'serve' dependencies.") from exc

    # Collect async worker coroutines to be started by the lifespan handler.
    _worker_coros: list[object] = []

    tokenizer = StateTokenizer(state_game_cfg)
    prior_queue: PriorQueue[PriorRequest] = PriorQueue()
    leaf_queue: PriorQueue[LeafRequest] = PriorQueue()
    prior_diagnostics = InferenceDiagnostics()
    leaf_diagnostics = InferenceDiagnostics()
    if batch_window_ms < 0:
        raise ValueError("batch_window_ms must be non-negative")

    if state_model is not None:

        def _infer_prior_batch(batch: list[PriorRequest]) -> list[PriorResponseItem]:
            """Return complete parent-state policies and canonical values."""
            started_ns = time.perf_counter_ns()
            queue_wait_ms = sum(
                max(0, started_ns - req._queued_at_ns) / 1_000_000 for req in batch
            )
            valid: list[tuple[int, PriorRequest]] = []
            results: list[PriorResponseItem | None] = [None] * len(batch)
            tokens: list[torch.Tensor] = []
            for index, req in enumerate(batch):
                try:
                    tokens.append(tokenizer.tokenize(tokenizer.canonicalize(req.parent_state)))
                    valid.append((index, req))
                except (KeyError, ValueError):
                    results[index] = PriorResponseItem(id=req.id, priors=[])

            tokenized_ns = time.perf_counter_ns()
            transfer_ms = 0.0
            forward_ms = 0.0

            if tokens:
                transfer_start = time.perf_counter_ns()
                token_ids = torch.stack(tokens).to(state_device)
                if state_device.type == "cuda":
                    torch.cuda.synchronize(state_device)
                transferred_ns = time.perf_counter_ns()
                with torch.no_grad():
                    output = state_model(token_ids)
                    values = F.softmax(_extract_logits(output, "value").float(), dim=-1)
                    policies = F.softmax(_extract_logits(output, "policy").float(), dim=-1)
                if state_device.type == "cuda":
                    torch.cuda.synchronize(state_device)
                forwarded_ns = time.perf_counter_ns()
                transfer_ms = (transferred_ns - transfer_start) / 1_000_000
                forward_ms = (forwarded_ns - transferred_ns) / 1_000_000
                for batch_index, (result_index, req) in enumerate(valid):
                    player_values = values[batch_index].cpu().tolist()
                    results[result_index] = PriorResponseItem(
                        id=req.id,
                        priors=policies[batch_index].cpu().tolist(),
                        value_estimate=player_values[0],
                        player_win_probabilities=player_values,
                    )

            prior_diagnostics.record(
                states=len(valid),
                queue_wait_ms=queue_wait_ms,
                tokenize_ms=(tokenized_ns - started_ns) / 1_000_000,
                transfer_ms=transfer_ms,
                forward_ms=forward_ms,
            )

            return [result for result in results if result is not None]

        async def _prior_worker() -> None:
            """Background task that continuously processes the prior queue."""
            batch_size = 512 #32
            while True:
                batch = await _dequeue_with_window(prior_queue, batch_size, batch_window_ms)
                if batch:
                    # Run inference in a thread pool to avoid blocking the event loop.
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, _infer_prior_batch, batch)
                    prior_queue.add_results(results)
                else:
                    # No work — sleep briefly to avoid busy-waiting.
                    await asyncio.sleep(0.005)

        _worker_coros.append(_prior_worker)

        def _infer_leaf_batch(batch: list[LeafRequest]) -> list[LeafResponseItem]:
            """Flatten requests so one model call evaluates every stochastic outcome."""
            started_ns = time.perf_counter_ns()
            queue_wait_ms = sum(
                max(0, started_ns - request._queued_at_ns) / 1_000_000
                for request in batch
                for _ in request.states
            )
            valid: list[tuple[int, LeafRequest]] = []
            results: list[LeafResponseItem | None] = [None] * len(batch)
            token_batches: list[torch.Tensor] = []
            for index, request in enumerate(batch):
                if not request.states:
                    results[index] = LeafResponseItem(id=request.id, values=[])
                    continue
                try:
                    canonical = [tokenizer.canonicalize(state) for state in request.states]
                    tokens = tokenizer.tokenize_batch(canonical)
                    if tokens.shape[1] != state_game_cfg.state_token_size:
                        raise ValueError("state has the wrong token length")
                    token_batches.append(tokens)
                    valid.append((index, request))
                except (KeyError, ValueError):
                    results[index] = LeafResponseItem(id=request.id, values=[])

            tokenized_ns = time.perf_counter_ns()
            transfer_ms = 0.0
            forward_ms = 0.0

            if not token_batches:
                leaf_diagnostics.record(
                    states=0,
                    queue_wait_ms=queue_wait_ms,
                    tokenize_ms=(tokenized_ns - started_ns) / 1_000_000,
                    transfer_ms=0.0,
                    forward_ms=0.0,
                )
                return [result for result in results if result is not None]
            transfer_start = time.perf_counter_ns()
            token_ids = torch.cat(token_batches).to(state_device)
            if state_device.type == "cuda":
                torch.cuda.synchronize(state_device)
            transferred_ns = time.perf_counter_ns()
            with torch.no_grad():
                probabilities = F.softmax(
                    _extract_logits(state_model(token_ids), "value").float(), dim=-1
                )
            if state_device.type == "cuda":
                torch.cuda.synchronize(state_device)
            forwarded_ns = time.perf_counter_ns()
            transfer_ms = (transferred_ns - transfer_start) / 1_000_000
            forward_ms = (forwarded_ns - transferred_ns) / 1_000_000
            values = probabilities.cpu().tolist()
            offset = 0
            for result_index, request in valid:
                count = len(request.states)
                results[result_index] = LeafResponseItem(
                    id=request.id, values=values[offset : offset + count]
                )
                offset += count
            leaf_diagnostics.record(
                states=token_ids.shape[0],
                queue_wait_ms=queue_wait_ms,
                tokenize_ms=(tokenized_ns - started_ns) / 1_000_000,
                transfer_ms=transfer_ms,
                forward_ms=forward_ms,
            )
            return [result for result in results if result is not None]

        async def _leaf_worker() -> None:
            while True:
                batch = await _dequeue_with_window(leaf_queue, 32, batch_window_ms)
                if batch:
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, _infer_leaf_batch, batch)
                    leaf_queue.add_results(results)  # type: ignore[arg-type]
                else:
                    await asyncio.sleep(0.005)

        _worker_coros.append(_leaf_worker)

    # ── Lifespan context manager ──────────────────────────────────────────

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        """Start background workers on startup; cancel them on shutdown."""
        tasks: set[asyncio.Task[None]] = set()
        for coro_fn in _worker_coros:
            task = asyncio.create_task(coro_fn())
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(title="GimburNet Inference", lifespan=lifespan)

    # ── State-model route registration ────────────────────────────────────

    if state_model is not None:

        @app.post("/state/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest) -> PredictResponse:
            if not request.states:
                raise HTTPException(status_code=400, detail="states list must not be empty")

            try:
                canonical = [tokenizer.canonicalize(state) for state in request.states]
                token_ids = tokenizer.tokenize_batch(canonical).to(state_device)
            except (KeyError, ValueError) as exc:
                logger.error(
                    "Tokenization failed in /state/predict: %s\n  First state (truncated): %.200s",
                    exc,
                    request.states[0] if request.states else "<empty>",
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            with torch.no_grad():
                output = state_model(token_ids)
                probs = F.softmax(_extract_logits(output, "value").float(), dim=-1)
                policy_probs = F.softmax(_extract_logits(output, "policy").float(), dim=-1)

            return PredictResponse(
                player_win_probabilities=probs.cpu().tolist(),
                policy_probabilities=policy_probs.cpu().tolist(),
            )

        @app.post("/state/predict-player", response_model=PredictPlayerResponse)
        async def predict_player(
            request: PredictPlayerRequest,
        ) -> PredictPlayerResponse:
            """Predict win probability for specific target players.

            Runs one full-vector inference per state and gathers each requested player.
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
                if any(
                    not 1 <= player <= state_game_cfg.player_count for player in request.players
                ):
                    raise ValueError("player is outside the configured player range")
                token_ids = tokenizer.tokenize_batch(request.states).to(state_device)
            except (KeyError, ValueError) as exc:
                logger.error(
                    "Tokenization failed in /state/predict-player: %s\n"
                    "  Players: %s\n  First state (truncated): %.200s",
                    exc,
                    request.players[:5],
                    request.states[0] if request.states else "<empty>",
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            with torch.no_grad():
                output = state_model(token_ids)
                probs = F.softmax(_extract_logits(output).float(), dim=-1)

            win_probs = [float(probs[i, player - 1]) for i, player in enumerate(request.players)]
            return PredictPlayerResponse(win_probabilities=win_probs)

        # ── Prior endpoints ───────────────────────────────────────────────────

        @app.post("/state/prior-enqueue", status_code=202)
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

        @app.post("/state/prior-collect", response_model=PriorCollectResponse)
        async def prior_collect() -> PriorCollectResponse:
            """Return all completed prior inference results."""
            results = prior_queue.collect_results()
            return PriorCollectResponse(responses=results)

        @app.post("/state/leaf-enqueue", status_code=202)
        async def leaf_enqueue(request: LeafEnqueueRequest) -> JSONResponse:
            accepted_ids: list[str] = []
            dropped_ids: list[str] = []
            for item in request.requests:
                target = accepted_ids if leaf_queue.enqueue_without_eviction(item) else dropped_ids
                target.append(item.id)
            return JSONResponse(
                status_code=202,
                content={
                    "accepted": len(accepted_ids),
                    "dropped": len(dropped_ids),
                    "accepted_ids": accepted_ids,
                    "dropped_ids": dropped_ids,
                },
            )

        @app.post("/state/leaf-predict", response_model=LeafCollectResponse)
        async def leaf_predict(request: LeafEnqueueRequest) -> LeafCollectResponse:
            """Evaluate all requested leaves in one flattened model batch."""
            loop = asyncio.get_running_loop()
            responses = await loop.run_in_executor(None, _infer_leaf_batch, request.requests)
            return LeafCollectResponse(responses=responses)

        @app.post("/state/leaf-collect", response_model=LeafCollectResponse)
        async def leaf_collect() -> LeafCollectResponse:
            return LeafCollectResponse(responses=leaf_queue.collect_results())  # type: ignore[arg-type]

        @app.post("/state/leaf-cancel", response_model=LeafCancelResponse)
        async def leaf_cancel(request: LeafCancelRequest) -> LeafCancelResponse:
            removed_queued, removed_results = leaf_queue.cancel(set(request.ids))
            return LeafCancelResponse(
                removed_queued=removed_queued,
                removed_results=removed_results,
            )

        @app.post("/state/prior-flush")
        async def prior_flush() -> dict[str, str]:
            """Clear the priority queue and discard pending results."""
            prior_queue.flush()
            return {"status": "flushed"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/diagnostics")
    async def diagnostics() -> dict[str, object]:
        return {
            "prior": prior_diagnostics.snapshot(),
            "leaf": leaf_diagnostics.snapshot(),
            "queues": {
                "prior_pending": prior_queue.pending_count(),
                "leaf_pending": leaf_queue.pending_count(),
            },
        }

    return app


def main() -> None:
    args = parse_args()

    # ── Load config file and apply defaults ───────────────────────────────
    if args.config is not None:
        text = _strip_json_comments(args.config.read_text())
        raw = json.loads(text)
        for json_key, attr in _CONFIG_KEY_MAP.items():
            if json_key not in raw:
                continue
            current = getattr(args, attr, None)
            default = _ARG_DEFAULTS.get(attr)
            # If the CLI value differs from the default, the user explicitly
            # provided it — keep the CLI value.
            if current != default:
                continue
            value = raw[json_key]
            if attr == "model":
                value = Path(value)
            setattr(args, attr, value)

    if args.model is None:
        raise SystemExit("Error: --model is required.")
    if args.game_config is None:
        raise SystemExit("Error: --game-config is required.")
    if args.model_config is None:
        raise SystemExit("Error: --model-config is required.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state_game_cfg = CONFIGS_BY_NAME[args.game_config]
    state_model_cfg = MODEL_CONFIGS_BY_NAME[args.model_config]
    try:
        state_ckpt = _load_checkpoint(args.model, device)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}") from exc
    precision = _checkpoint_precision(state_ckpt)
    if precision == "fp16" and device.type != "cuda":
        raise SystemExit("Error: FP16 inference checkpoints require CUDA.")
    if state_ckpt.get("game_config") not in (None, args.game_config):
        raise SystemExit("Error: checkpoint game_config does not match --game-config.")
    if state_ckpt.get("model_config") not in (None, args.model_config):
        raise SystemExit("Error: checkpoint model_config does not match --model-config.")
    loaded_state_model = GimburTransformer(state_game_cfg, state_model_cfg)
    if precision == "fp16":
        loaded_state_model.half()
    loaded_state_model.load_state_dict(state_ckpt["model_state_dict"])
    loaded_state_model.to(device)
    loaded_state_model.eval()
    param_count = sum(p.numel() for p in loaded_state_model.parameters())
    if args.compile_model:
        if device.type != "cuda":
            raise SystemExit("Error: compiled inference requires CUDA.")
        loaded_state_model = torch.compile(loaded_state_model, dynamic=True)
        warmup = torch.zeros(
            (1, state_game_cfg.state_token_size), dtype=torch.long, device=device
        )
        with torch.no_grad():
            loaded_state_model(warmup)
        torch.cuda.synchronize()
    print(
        f"Loaded model ({args.model_config}) for {args.game_config} "
        f"({param_count:,} parameters, {precision}, "
        f"compiled={args.compile_model}) on {device}"
    )

    app = create_app(
        state_model=loaded_state_model,
        state_device=device,
        state_game_cfg=state_game_cfg,
        batch_window_ms=args.batch_window_ms,
    )
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Error: serving requires the optional 'serve' dependencies.") from exc

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=args.log_level in ("debug", "info"),
        limit_concurrency=200,
    )


if __name__ == "__main__":
    main()
