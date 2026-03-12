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
from pathlib import Path

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .game_config import CONFIGS_BY_NAME, GameConfig
from .tokenizer import rotate_player_state, tokenize_batch
from .transformer_model import (
    MODEL_CONFIGS_BY_NAME,
    GimburTransformer,
)


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
    return parser.parse_args()


def create_app(
    model: GimburTransformer,
    device: torch.device,
    game_cfg: GameConfig,
) -> FastAPI:
    """Build the FastAPI application with the model captured in closure."""
    app = FastAPI(title="GimburNet Inference")

    def _expected_win_prob(probs: torch.Tensor) -> float:
        """Compute expected win probability from a 1-D bucket distribution."""
        n = probs.shape[0]
        centres = torch.arange(n, dtype=probs.dtype, device=probs.device)
        centres = (centres + 0.5) / n
        return float((probs * centres).sum())

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest) -> PredictResponse:
        if not request.states:
            raise HTTPException(status_code=400, detail="states list must not be empty")

        try:
            token_ids = tokenize_batch(request.states).to(device)
        except (KeyError, ValueError) as exc:
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
                rotate_player_state(s, p, game_cfg) for s, p in zip(request.states, request.players)
            ]
            token_ids = tokenize_batch(rotated).to(device)
        except (KeyError, ValueError) as exc:
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
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
