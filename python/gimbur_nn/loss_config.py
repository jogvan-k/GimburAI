"""
Configurable loss functions for bucket-based win-probability prediction.

Two loss modes are supported:

- **hard** — Standard cross-entropy with a single hard target bucket.
  All misclassifications are penalised equally regardless of distance
  from the true bucket.

- **gaussian** — Gaussian label smoothing.  The target is a soft
  probability distribution centred on the true bucket, with spread
  controlled by ``sigma`` (in bucket units).  The loss is the
  KL-divergence between the predicted log-softmax and the soft target.
  This penalises predictions proportionally to their distance from the
  true bucket.

Usage::

    from gimbur_nn.loss_config import LossConfig, build_loss_fn

    cfg = LossConfig(mode="gaussian", sigma=2.0)
    loss_fn = build_loss_fn(cfg, n_buckets=128)

    loss = loss_fn(logits, targets)   # targets: (batch,) long tensor
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Type alias for a loss function: (logits, targets) -> scalar loss.
LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def masked_soft_target_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Cross entropy for dense soft targets over legal actions only."""
    masked_logits = logits.masked_fill(~legal_mask.bool(), float("-inf"))
    log_probs = F.log_softmax(masked_logits, dim=-1).masked_fill(~legal_mask.bool(), 0.0)
    per_sample = -(targets * log_probs).sum(dim=-1)
    return per_sample.mean()


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy between raw logits and dense probability targets."""
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


@dataclass
class LossConfig:
    """Configuration for the training loss function.

    Attributes:
        mode: Loss mode — ``"hard"`` for standard cross-entropy,
            ``"gaussian"`` for Gaussian label smoothing.
        sigma: Standard deviation of the Gaussian kernel (in bucket
            units).  Only used when ``mode="gaussian"``.  Default 2.0.
    """

    mode: str = "ordinal"
    sigma: float = 2.0


# ── Loss mode: hard (standard cross-entropy) ────────────────────────


def _hard_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard cross-entropy loss with hard integer targets.

    Args:
        logits: ``(batch, n_buckets)`` raw logits.
        targets: ``(batch,)`` integer bucket indices.

    Returns:
        Scalar loss.
    """
    return F.cross_entropy(logits, targets)


def _ordinal_cdf_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Squared Earth-mover loss between predicted and target bucket CDFs."""
    probabilities = torch.softmax(logits, dim=-1)
    predicted_cdf = probabilities.cumsum(dim=-1)
    indices = torch.arange(logits.shape[-1], device=logits.device)
    target_cdf = (indices.unsqueeze(0) >= targets.unsqueeze(1)).to(logits.dtype)
    return (predicted_cdf - target_cdf).square().mean()


# ── Loss mode: gaussian (label-smoothed soft targets) ────────────────


def _build_gaussian_loss(n_buckets: int, sigma: float) -> LossFn:
    """Return a loss function that uses Gaussian-smoothed soft targets.

    The soft target for true bucket ``k`` is a normalised Gaussian
    centred at ``k`` with standard deviation ``sigma``::

        w[i] = exp(-0.5 * ((i - k) / sigma)^2)
        target[i] = w[i] / sum(w)

    The loss is ``KL(target || softmax(logits))``, computed via
    :func:`torch.nn.functional.kl_div` with ``reduction="batchmean"``.

    The Gaussian kernel matrix is pre-computed once and cached as a
    ``(n_buckets, n_buckets)`` tensor on the same device as the logits
    (moved lazily on first call).
    """
    # Pre-compute the kernel: kernel[k, i] = normalised Gaussian weight
    # for true bucket k at position i.
    indices = torch.arange(n_buckets, dtype=torch.float32)
    # (n_buckets, 1) - (1, n_buckets) -> (n_buckets, n_buckets)
    diff = indices.unsqueeze(1) - indices.unsqueeze(0)
    log_weights = -0.5 * (diff / sigma) ** 2
    # Normalise each row to a valid probability distribution.
    kernel = F.softmax(log_weights, dim=1)  # (n_buckets, n_buckets)

    # Mutable container for the device-local cached kernel.
    cache: dict[torch.device, torch.Tensor] = {}

    def _gaussian_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = logits.device
        if device not in cache:
            cache[device] = kernel.to(device)
        k = cache[device]

        # Build soft targets: gather rows from the kernel matrix.
        soft_targets = k[targets]  # (batch, n_buckets)

        log_probs = F.log_softmax(logits, dim=1)
        return F.kl_div(log_probs, soft_targets, reduction="batchmean")

    return _gaussian_loss


# ── Public API ───────────────────────────────────────────────────────

# Valid mode names for CLI / config validation.
LOSS_MODES = ("hard", "gaussian", "ordinal")


def build_loss_fn(cfg: LossConfig, *, n_buckets: int) -> LossFn:
    """Build a loss function from a :class:`LossConfig`.

    Args:
        cfg: Loss configuration.
        n_buckets: Number of output buckets (must match the model).

    Returns:
        A callable ``(logits, targets) -> scalar_loss``.

    Raises:
        ValueError: If ``cfg.mode`` is not recognised.
    """
    if cfg.mode == "hard":
        return _hard_cross_entropy
    if cfg.mode == "gaussian":
        return _build_gaussian_loss(n_buckets, cfg.sigma)
    if cfg.mode == "ordinal":
        return _ordinal_cdf_loss
    raise ValueError(f"Unknown loss mode {cfg.mode!r}. Valid modes: {', '.join(LOSS_MODES)}")
