from __future__ import annotations

import torch


def expected_values(logits: torch.Tensor) -> torch.Tensor:
    """Decode bucket logits to scalar probabilities using bucket centers."""
    probabilities = torch.softmax(logits, dim=-1)
    bucket_count = logits.shape[-1]
    centers = (torch.arange(bucket_count, device=logits.device) + 0.5) / bucket_count
    return probabilities @ centers


def value_metrics(
    logits: torch.Tensor, targets: torch.Tensor, calibration_bins: int = 10
) -> dict[str, float]:
    """Return scalar MAE, Brier score, and expected calibration error."""
    predictions = expected_values(logits)
    bucket_count = logits.shape[-1]
    target_values = (targets.float() + 0.5) / bucket_count
    errors = predictions - target_values

    ece = torch.zeros((), device=logits.device)
    bin_indices = torch.clamp((predictions * calibration_bins).long(), max=calibration_bins - 1)
    for bin_index in range(calibration_bins):
        mask = bin_indices == bin_index
        if mask.any():
            weight = mask.float().mean()
            ece += weight * (predictions[mask].mean() - target_values[mask].mean()).abs()

    return {
        "mae": errors.abs().mean().item(),
        "brier": errors.square().mean().item(),
        "ece": ece.item(),
    }


def candidate_ranking_accuracy(
    predictions: torch.Tensor, targets: torch.Tensor, group_ids: torch.Tensor
) -> float:
    """Measure how often the predicted and target best candidate agree per group."""
    correct = 0
    groups = torch.unique(group_ids)
    for group in groups:
        mask = group_ids == group
        correct += int(torch.argmax(predictions[mask]) == torch.argmax(targets[mask]))
    return correct / len(groups) if len(groups) else 0.0
