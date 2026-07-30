from __future__ import annotations

import pytest
import torch

from gimbur_nn.metrics import candidate_ranking_accuracy, expected_values, value_metrics


def test_expected_values_uses_bucket_centers() -> None:
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 100.0]])

    assert expected_values(logits).tolist() == pytest.approx([0.125, 0.875])


def test_value_metrics_are_zero_for_matching_bucket_centers() -> None:
    logits = torch.tensor([[100.0, 0.0], [0.0, 100.0]])
    targets = torch.tensor([0, 1])

    assert value_metrics(logits, targets) == pytest.approx(
        {"mae": 0.0, "brier": 0.0, "ece": 0.0}, abs=1e-6
    )


def test_candidate_ranking_accuracy_is_grouped() -> None:
    predictions = torch.tensor([0.8, 0.2, 0.6, 0.4])
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0])
    groups = torch.tensor([0, 0, 1, 1])

    assert candidate_ranking_accuracy(predictions, targets, groups) == 0.5
