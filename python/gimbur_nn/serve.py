"""
Inference server for GimburNet.

Loads trained model checkpoints and exposes HTTP endpoints for both state
and placement models.  A single server can serve one or both model types.

State-model endpoints live under ``/state/...`` and placement-model
endpoints under ``/placement/...``.  The server registers whichever set
of endpoints corresponds to the models that were provided.

Usage (state model only)::

    python -m gimbur_nn.serve \
        --model model.pt \
        --game-config mini_2p \
        --model-config small \
        --port 8000

Usage (placement model only)::

    python -m gimbur_nn.serve \
        --placement-model placement.pt \
        --game-config mini_2p \
        --placement-model-config small \
        --port 8000

Usage (both models on a single server)::

    python -m gimbur_nn.serve \
        --model model.pt \
        --game-config mini_2p \
        --model-config small \
        --placement-model placement.pt \
        --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
import torch.nn.functional as F
from pydantic import BaseModel

if TYPE_CHECKING:
    from fastapi import FastAPI

from .game_config import CONFIGS_BY_NAME, GameConfig
from .placement_tokenizer import PlacementTokenizer
from .state_tokenizer import StateTokenizer
from .transformer_model import (
    MODEL_CONFIGS_BY_NAME,
    GimburPlacementTransformer,
    GimburTransformer,
    _make_model_config,
)

logger = logging.getLogger(__name__)


def _load_checkpoint(path: Path, device: torch.device, architecture: str) -> dict:
    """Load a version-3 player-value checkpoint or reject it."""
    raw = torch.load(path, map_location=device, weights_only=False)
    if (
        not isinstance(raw, dict)
        or "model_state_dict" not in raw
        or raw.get("checkpoint_version") != 3
        or raw.get("architecture") != architecture
    ):
        raise ValueError(
            f"incompatible checkpoint; expected checkpoint_version=3 and "
            f"architecture={architecture!r}"
        )
    return raw


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


# ── Placement endpoint models ─────────────────────────────────────────────────


class PredictPlacementRequest(BaseModel):
    """Request body for the /placement/predict endpoint."""

    states: list[str]
    """Serialized placement phase state strings."""


class PredictPlacementResponse(BaseModel):
    """Response body for the /placement/predict endpoint."""

    player_win_probabilities: list[list[float]]
    """Per-state player win distributions."""

    policy_probabilities: list[list[float]] | None = None
    """Full canonical action probabilities for combined models."""


# ── Prior queue models ────────────────────────────────────────────────────────


class PriorRequest(BaseModel):
    """A single prior request for one tree node."""

    id: str
    """Opaque ID to correlate response back to the MCTSState."""

    parent_state: str | None = None
    """Serialized state at the node, used by a value head when available."""

    states: list[str]
    """Serialized result states for each action (deterministic: 1 state,
    stochastic: 1 per outcome)."""

    player: int
    """1-based acting player for server-side rotation."""

    priority: int
    """Depth from root; lower = more important."""


class PriorEnqueueRequest(BaseModel):
    """Request body for /state/prior-enqueue."""

    requests: list[PriorRequest]


class PriorResponseItem(BaseModel):
    """A completed prior inference result for one tree node."""

    id: str
    priors: list[float]
    """Per-action prior weights in the legal-action order supplied by the client."""

    value_estimate: float | None = None
    """Scalar value estimate for the node's state. None if not available."""

    player_win_probabilities: list[float] | None = None
    """Full player value distribution when a value head is available."""


class PriorCollectResponse(BaseModel):
    """Response body for /state/prior-collect and /placement/prior-collect."""

    responses: list[PriorResponseItem]


# ── Placement prior queue models ──────────────────────────────────────────────


class PlacementPriorRequest(BaseModel):
    """A single placement prior request for one tree node."""

    id: str
    """Opaque ID to correlate response back to the MCTSState."""

    state: str
    """Serialized placement phase state string."""

    priority: int
    """Depth from root; lower = more important."""


class PlacementPriorEnqueueRequest(BaseModel):
    """Request body for /placement/prior-enqueue."""

    requests: list[PlacementPriorRequest]


# ── Priority queue for async prior inference ──────────────────────────────────

_PRIOR_QUEUE_CAPACITY = 4096

T = TypeVar("T")


class PriorQueue(Generic[T]):
    """Thread-safe priority queue for prior requests, with bounded capacity.

    Requests are ordered by priority (depth from root, lower = higher
    priority).  When the queue is full, a new request with lower priority
    than the worst queued item is silently dropped; one with higher
    priority evicts the lowest-priority entry.

    Generic over the request type *T* which must have a ``priority: int``
    attribute.  Works with both ``PriorRequest`` and
    ``PlacementPriorRequest``.
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

    def enqueue(self, req: T) -> bool:
        """Add a request.  Returns True if accepted, False if dropped."""
        priority: int = req.priority  # type: ignore[union-attr]
        with self._lock:
            if len(self._heap) < self._capacity:
                heapq.heappush(self._heap, (priority, self._seq, req))
                self._seq += 1
                return True
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
                self._heap[worst_idx] = self._heap[-1]
                self._heap.pop()
                heapq.heapify(self._heap)
                heapq.heappush(self._heap, (priority, self._seq, req))
                self._seq += 1
                return True
            # New request is lower priority — drop it.
            return False

    def dequeue_batch(self, batch_size: int) -> list[T]:
        """Remove and return up to *batch_size* highest-priority requests."""
        with self._lock:
            batch: list[T] = []
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
    "placementModel": "placement_model",
    "placementModelConfig": "placement_model_config",
    "port": "port",
    "host": "host",
    "logLevel": "log_level",
}
"""Maps camelCase JSON config keys to argparse dest names."""


_ARG_DEFAULTS: dict[str, object] = {
    "model": None,
    "game_config": None,
    "model_config": None,
    "placement_model": None,
    "placement_model_config": None,
    "port": 8000,
    "host": "127.0.0.1",
    "log_level": "info",
}
"""Default values for argparse arguments, used to detect explicit CLI overrides."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve GimburNet for inference.")

    # ── State model args ──────────────────────────────────────────────────
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

    # ── Placement model args ──────────────────────────────────────────────
    parser.add_argument(
        "--placement-model",
        type=Path,
        required=False,
        default=None,
        help="Path to trained placement model checkpoint.",
    )
    parser.add_argument(
        "--placement-model-config",
        type=str,
        required=False,
        default=None,
        choices=sorted(MODEL_CONFIGS_BY_NAME),
        help="Model size preset for the placement model.  Defaults to --model-config if omitted.",
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
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file with camelCase keys. CLI args override config values.",
    )
    return parser.parse_args()


def create_app(
    state_model: GimburTransformer | None = None,
    state_device: torch.device | None = None,
    state_game_cfg: GameConfig | None = None,
    state_output_mode: str = "value",
    placement_model: GimburPlacementTransformer | None = None,
    placement_device: torch.device | None = None,
    placement_game_cfg: GameConfig | None = None,
    placement_target: str = "winrate",
    placement_output_mode: str = "value",
) -> FastAPI:
    """Build the FastAPI application.

    Registers ``/state/...`` endpoints when a state model is provided
    and ``/placement/...`` endpoints when a placement model is provided.
    Both can be active simultaneously.

    ``placement_target`` is retained only for caller compatibility.
    """

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError("create_app requires the optional 'serve' dependencies.") from exc

    # Collect async worker coroutines to be started by the lifespan handler.
    _worker_coros: list[object] = []

    # ── State-model endpoints ─────────────────────────────────────────────

    if state_model is not None:
        assert state_device is not None
        assert state_game_cfg is not None
        tokenizer = StateTokenizer(state_game_cfg)
        prior_queue: PriorQueue[PriorRequest] = PriorQueue()

        def _infer_prior_batch(batch: list[PriorRequest]) -> list[PriorResponseItem]:
            """Run inference on a batch of prior requests and return results."""
            results: list[PriorResponseItem] = []
            for req in batch:
                if not req.states:
                    results.append(PriorResponseItem(id=req.id, priors=[]))
                    continue
                try:
                    inference_states = req.states
                    has_parent = req.parent_state is not None
                    if has_parent:
                        inference_states = [req.parent_state, *inference_states]
                    if not 1 <= req.player <= state_game_cfg.player_count:
                        raise ValueError("player is outside the configured player range")
                    token_ids = tokenizer.tokenize_batch(inference_states).to(state_device)
                except (KeyError, ValueError):
                    # Bad state — return zeros so the MCTS falls back to uniform.
                    results.append(
                        PriorResponseItem(
                            id=req.id,
                            priors=[0.0] * len(req.states),
                        )
                    )
                    continue

                with torch.no_grad():
                    output = state_model(token_ids)
                    prior_logits = _extract_logits(output)
                    prior_probs = F.softmax(prior_logits, dim=-1)

                prior_offset = 1 if has_parent else 0
                player_index = req.player - 1
                priors = prior_probs[prior_offset:, player_index].cpu().tolist()
                value_estimate: float | None = None
                player_probs: list[float] | None = None
                if has_parent:
                    player_probs = prior_probs[0].cpu().tolist()
                    value_estimate = player_probs[player_index]
                results.append(
                    PriorResponseItem(
                        id=req.id,
                        priors=priors,
                        value_estimate=value_estimate,
                        player_win_probabilities=player_probs,
                    )
                )

            return results

        async def _prior_worker() -> None:
            """Background task that continuously processes the prior queue."""
            batch_size = 32
            while True:
                batch = prior_queue.dequeue_batch(batch_size)
                if batch:
                    # Run inference in a thread pool to avoid blocking the event loop.
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, _infer_prior_batch, batch)
                    prior_queue.add_results(results)
                else:
                    # No work — sleep briefly to avoid busy-waiting.
                    await asyncio.sleep(0.005)

        _worker_coros.append(_prior_worker)

    if placement_model is not None:
        assert placement_device is not None
        assert placement_game_cfg is not None
        placement_tokenizer = PlacementTokenizer(placement_game_cfg)

        placement_prior_queue: PriorQueue[PlacementPriorRequest] = PriorQueue()

        def _infer_placement_prior_batch(
            batch: list[PlacementPriorRequest],
        ) -> list[PriorResponseItem]:
            """Return full-vocabulary priors and state values for placement states."""
            valid: list[tuple[int, PlacementPriorRequest]] = []
            results: list[PriorResponseItem | None] = [None] * len(batch)
            tokens: list[torch.Tensor] = []
            for index, req in enumerate(batch):
                try:
                    tokens.append(placement_tokenizer.tokenize_state(req.state))
                    valid.append((index, req))
                except (KeyError, ValueError):
                    results[index] = PriorResponseItem(id=req.id, priors=[])

            if tokens:
                token_batch = torch.stack(tokens).to(placement_device)
                with torch.no_grad():
                    output = placement_model(token_batch)
                    value_logits = output["value"] if isinstance(output, dict) else output
                    value_probs = F.softmax(value_logits, dim=-1)
                    policy_probs = (
                        F.softmax(output["policy"], dim=-1).cpu().tolist()
                        if isinstance(output, dict)
                        else [[] for _ in valid]
                    )

                for batch_index, ((result_index, req), priors) in enumerate(
                    zip(valid, policy_probs)
                ):
                    results[result_index] = PriorResponseItem(
                        id=req.id,
                        priors=priors,
                        value_estimate=float(value_probs[batch_index, 0]),
                        player_win_probabilities=value_probs[batch_index].cpu().tolist(),
                    )

            return [result for result in results if result is not None]

        async def _placement_prior_worker() -> None:
            """Background task that continuously processes the placement prior queue."""
            batch_size = 32
            while True:
                batch = placement_prior_queue.dequeue_batch(batch_size)
                if batch:
                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, _infer_placement_prior_batch, batch)
                    placement_prior_queue.add_results(results)
                else:
                    await asyncio.sleep(0.005)

        _worker_coros.append(_placement_prior_worker)

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
                token_ids = tokenizer.tokenize_batch(request.states).to(state_device)
            except (KeyError, ValueError) as exc:
                logger.error(
                    "Tokenization failed in /state/predict: %s\n  First state (truncated): %.200s",
                    exc,
                    request.states[0] if request.states else "<empty>",
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            with torch.no_grad():
                output = state_model(token_ids)
                probs = F.softmax(_extract_logits(output), dim=-1)

            return PredictResponse(player_win_probabilities=probs.cpu().tolist())

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
                probs = F.softmax(_extract_logits(output), dim=-1)

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

        @app.post("/state/prior-flush")
        async def prior_flush() -> dict[str, str]:
            """Clear the priority queue and discard pending results."""
            prior_queue.flush()
            return {"status": "flushed"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ── Placement endpoint ────────────────────────────────────────────────────

    if placement_model is not None:

        @app.post("/placement/predict", response_model=PredictPlacementResponse)
        async def predict_placement(
            request: PredictPlacementRequest,
        ) -> PredictPlacementResponse:
            """Predict values and optional full policies from placement states."""
            if not request.states:
                raise HTTPException(
                    status_code=400,
                    detail="states list must not be empty",
                )
            try:
                token_batch = placement_tokenizer.tokenize_batch(request.states).to(
                    placement_device
                )
            except (KeyError, ValueError) as exc:
                logger.error(
                    "Tokenization failed in /placement/predict: %s\n"
                    "  First state (truncated): %.200s",
                    exc,
                    request.states[0] if request.states else "<empty>",
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            with torch.no_grad():
                output = placement_model(token_batch)
                value_logits = output["value"] if isinstance(output, dict) else output
                probs = F.softmax(value_logits, dim=-1)
                policy_probs = (
                    F.softmax(output["policy"], dim=-1).cpu().tolist()
                    if isinstance(output, dict)
                    else None
                )

            return PredictPlacementResponse(
                player_win_probabilities=probs.cpu().tolist(),
                policy_probabilities=policy_probs,
            )

        # ── Placement prior queue endpoints ──────────────────────────────────

        @app.post("/placement/prior-enqueue", status_code=202)
        async def prior_placement_enqueue(
            request: PlacementPriorEnqueueRequest,
        ) -> JSONResponse:
            """Accept a batch of placement prior requests into the priority queue."""
            accepted = 0
            dropped = 0
            for req in request.requests:
                if placement_prior_queue.enqueue(req):
                    accepted += 1
                else:
                    dropped += 1
            return JSONResponse(
                status_code=202,
                content={"accepted": accepted, "dropped": dropped},
            )

        @app.post("/placement/prior-collect", response_model=PriorCollectResponse)
        async def prior_placement_collect() -> PriorCollectResponse:
            """Return all completed placement prior inference results."""
            results = placement_prior_queue.collect_results()
            return PriorCollectResponse(responses=results)

        @app.post("/placement/prior-flush")
        async def prior_placement_flush() -> dict[str, str]:
            """Clear the placement priority queue and discard pending results."""
            placement_prior_queue.flush()
            return {"status": "flushed"}

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
            # Convert string path for --model and --placement-model.
            if attr in ("model", "placement_model"):
                value = Path(value)
            setattr(args, attr, value)

    # Determine which models to load.
    has_state = args.model is not None
    has_placement = args.placement_model is not None

    if not has_state and not has_placement:
        raise SystemExit(
            "Error: at least one model must be specified.\n"
            "  Use --model for a state model, --placement-model for a placement model, "
            "or both."
        )

    # Validate state model args.
    if has_state:
        if args.game_config is None:
            raise SystemExit("Error: --game-config is required when --model is specified.")
        if args.model_config is None:
            raise SystemExit("Error: --model-config is required when --model is specified.")

    # Validate placement model args.
    if has_placement:
        if args.game_config is None:
            raise SystemExit(
                "Error: --game-config is required when --placement-model is specified."
            )
        # Default --placement-model-config to --model-config when omitted.
        if args.placement_model_config is None:
            if args.model_config is not None:
                args.placement_model_config = args.model_config
            else:
                raise SystemExit(
                    "Error: --placement-model-config (or --model-config) is required "
                    "when --placement-model is specified."
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load state model ──────────────────────────────────────────────────
    loaded_state_model: GimburTransformer | None = None
    state_game_cfg: GameConfig | None = None
    state_output_mode = "value"
    if has_state:
        state_game_cfg = CONFIGS_BY_NAME[args.game_config]
        state_model_cfg = MODEL_CONFIGS_BY_NAME[args.model_config]
        try:
            state_ckpt = _load_checkpoint(args.model, device, "state_player_value_v1")
        except ValueError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        state_output_mode = str(state_ckpt.get("output_mode", "value"))
        if state_output_mode != getattr(state_model_cfg, "output_mode", "value"):
            state_model_cfg = _make_model_config(
                d_model=state_model_cfg.d_model,
                n_heads=state_model_cfg.n_heads,
                n_layers=state_model_cfg.n_layers,
                ffn_hidden_mult=state_model_cfg.ffn_hidden_mult,
                dropout=state_model_cfg.dropout,
                output_mode=state_output_mode,
            )
        loaded_state_model = GimburTransformer(state_game_cfg, state_model_cfg)
        loaded_state_model.load_state_dict(state_ckpt["model_state_dict"])
        loaded_state_model.to(device)
        loaded_state_model.eval()
        param_count = sum(p.numel() for p in loaded_state_model.parameters())
        print(
            f"Loaded state model ({args.model_config}) for {args.game_config} "
            f"({param_count:,} parameters) on {device}, output_mode={state_output_mode}"
        )

    # ── Load placement model ──────────────────────────────────────────────
    loaded_placement_model: GimburPlacementTransformer | None = None
    placement_game_cfg: GameConfig | None = None
    placement_target: str = "winrate"
    placement_output_mode: str = "value"
    if has_placement:
        placement_game_cfg = CONFIGS_BY_NAME[args.game_config]
        placement_model_cfg = MODEL_CONFIGS_BY_NAME[args.placement_model_config]
        try:
            placement_ckpt = _load_checkpoint(args.placement_model, device, "placement_state_v3")
        except ValueError as exc:
            raise SystemExit(f"Error: {exc}") from exc
        placement_target = str(placement_ckpt.get("target", "winrate"))
        placement_output_mode = str(placement_ckpt.get("output_mode", "value"))
        state_dict = placement_ckpt["model_state_dict"]

        # Apply output_mode to the model config if it differs from the preset.
        if placement_output_mode != getattr(placement_model_cfg, "output_mode", "value"):
            placement_model_cfg = _make_model_config(
                d_model=placement_model_cfg.d_model,
                n_heads=placement_model_cfg.n_heads,
                n_layers=placement_model_cfg.n_layers,
                ffn_hidden_mult=placement_model_cfg.ffn_hidden_mult,
                dropout=placement_model_cfg.dropout,
                output_mode=placement_output_mode,
            )

        loaded_placement_model = GimburPlacementTransformer(placement_game_cfg, placement_model_cfg)
        loaded_placement_model.load_state_dict(state_dict)
        loaded_placement_model.to(device)
        loaded_placement_model.eval()
        param_count = sum(p.numel() for p in loaded_placement_model.parameters())
        print(
            f"Loaded placement model ({args.placement_model_config}) for "
            f"{args.game_config} ({param_count:,} parameters) on {device}, "
            f"target={placement_target}, output_mode={placement_output_mode}"
        )

    app = create_app(
        state_model=loaded_state_model,
        state_device=device if has_state else None,
        state_game_cfg=state_game_cfg,
        state_output_mode=state_output_mode,
        placement_model=loaded_placement_model,
        placement_device=device if has_placement else None,
        placement_game_cfg=placement_game_cfg,
        placement_target=placement_target,
        placement_output_mode=placement_output_mode,
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
