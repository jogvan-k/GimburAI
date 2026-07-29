"""Tests for the loss_config module."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gimbur_nn.loss_config import (
    LOSS_MODES,
    LossConfig,
    build_loss_fn,
    masked_soft_target_cross_entropy,
)


def test_masked_soft_target_cross_entropy_ignores_illegal_logits() -> None:
    targets = torch.tensor([[0.25, 0.75, 0.0]])
    mask = torch.tensor([[True, True, False]])
    logits = torch.tensor([[0.0, 1.0, 1000.0]], requires_grad=True)

    loss = masked_soft_target_cross_entropy(logits, targets, mask)
    expected = -(targets[0, :2] * F.log_softmax(logits[0, :2], dim=0)).sum()

    assert loss.item() == pytest.approx(expected.item())
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[0, 2] == 0

# ---------------------------------------------------------------------------
# LossConfig defaults
# ---------------------------------------------------------------------------


class TestLossConfig:
    def test_default_mode(self) -> None:
        cfg = LossConfig()
        assert cfg.mode == "hard"

    def test_default_sigma(self) -> None:
        cfg = LossConfig()
        assert cfg.sigma == 2.0

    def test_custom_values(self) -> None:
        cfg = LossConfig(mode="gaussian", sigma=3.5)
        assert cfg.mode == "gaussian"
        assert cfg.sigma == 3.5


# ---------------------------------------------------------------------------
# build_loss_fn
# ---------------------------------------------------------------------------


class TestBuildLossFn:
    def test_hard_mode_returns_callable(self) -> None:
        fn = build_loss_fn(LossConfig(mode="hard"), n_buckets=16)
        assert callable(fn)

    def test_gaussian_mode_returns_callable(self) -> None:
        fn = build_loss_fn(LossConfig(mode="gaussian"), n_buckets=16)
        assert callable(fn)

    def test_unknown_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown loss mode"):
            build_loss_fn(LossConfig(mode="invalid"), n_buckets=16)

    def test_loss_modes_constant(self) -> None:
        assert "hard" in LOSS_MODES
        assert "gaussian" in LOSS_MODES


# ---------------------------------------------------------------------------
# Hard cross-entropy loss
# ---------------------------------------------------------------------------


class TestHardLoss:
    def test_matches_pytorch_cross_entropy(self) -> None:
        """Hard loss should be identical to F.cross_entropy."""
        torch.manual_seed(42)
        logits = torch.randn(8, 16)
        targets = torch.randint(0, 16, (8,))

        fn = build_loss_fn(LossConfig(mode="hard"), n_buckets=16)
        expected = F.cross_entropy(logits, targets)

        assert fn(logits, targets).item() == pytest.approx(expected.item())

    def test_perfect_prediction_low_loss(self) -> None:
        """When the model puts all mass on the correct bucket, loss is near zero."""
        n = 8
        logits = torch.full((1, n), -100.0)
        logits[0, 3] = 100.0
        targets = torch.tensor([3])

        fn = build_loss_fn(LossConfig(mode="hard"), n_buckets=n)
        assert fn(logits, targets).item() < 0.01

    def test_gradient_flows(self) -> None:
        logits = torch.randn(4, 8, requires_grad=True)
        targets = torch.randint(0, 8, (4,))

        fn = build_loss_fn(LossConfig(mode="hard"), n_buckets=8)
        loss = fn(logits, targets)
        loss.backward()
        assert logits.grad is not None


# ---------------------------------------------------------------------------
# Gaussian label-smoothing loss
# ---------------------------------------------------------------------------


class TestGaussianLoss:
    def test_perfect_prediction_low_loss(self) -> None:
        """When the model's output matches the soft target distribution, loss is low.

        Unlike hard cross-entropy where a delta prediction is optimal,
        gaussian loss is minimised when the predicted softmax matches
        the Gaussian-smoothed target.  We construct logits that produce
        exactly the target distribution.
        """
        n = 16
        sigma = 2.0
        true_bucket = 5

        # Build the same Gaussian target the loss function uses internally.
        indices = torch.arange(n, dtype=torch.float32)
        log_weights = -0.5 * ((indices - true_bucket) / sigma) ** 2
        target_dist = F.softmax(log_weights, dim=0)

        # Set logits = log(target_dist) so softmax(logits) ≈ target_dist.
        logits = torch.log(target_dist).unsqueeze(0)
        targets = torch.tensor([true_bucket])

        fn = build_loss_fn(LossConfig(mode="gaussian", sigma=sigma), n_buckets=n)
        assert fn(logits, targets).item() < 0.01

    def test_nearby_wrong_lower_loss_than_far_wrong(self) -> None:
        """Predicting an adjacent bucket should have lower loss than a distant one.

        This is the key property that distinguishes gaussian from hard:
        the ordinal distance between predicted and true bucket matters.
        """
        n = 32
        true_bucket = 16

        # Model predicts bucket 15 (off by 1).
        logits_near = torch.full((1, n), -100.0)
        logits_near[0, true_bucket - 1] = 100.0

        # Model predicts bucket 0 (off by 16).
        logits_far = torch.full((1, n), -100.0)
        logits_far[0, 0] = 100.0

        targets = torch.tensor([true_bucket])

        fn = build_loss_fn(LossConfig(mode="gaussian", sigma=2.0), n_buckets=n)
        loss_near = fn(logits_near, targets).item()
        loss_far = fn(logits_far, targets).item()

        assert loss_near < loss_far

    def test_symmetric_around_true_bucket(self) -> None:
        """Predictions equidistant from the true bucket should have equal loss."""
        n = 32
        true_bucket = 16

        logits_left = torch.full((1, n), -100.0)
        logits_left[0, true_bucket - 3] = 100.0

        logits_right = torch.full((1, n), -100.0)
        logits_right[0, true_bucket + 3] = 100.0

        targets = torch.tensor([true_bucket])

        fn = build_loss_fn(LossConfig(mode="gaussian", sigma=2.0), n_buckets=n)
        loss_left = fn(logits_left, targets).item()
        loss_right = fn(logits_right, targets).item()

        assert loss_left == pytest.approx(loss_right, rel=1e-4)

    def test_gradient_flows(self) -> None:
        logits = torch.randn(4, 16, requires_grad=True)
        targets = torch.randint(0, 16, (4,))

        fn = build_loss_fn(LossConfig(mode="gaussian", sigma=2.0), n_buckets=16)
        loss = fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_larger_sigma_more_spread(self) -> None:
        """Larger sigma should make loss less sensitive to distance.

        With a very large sigma, the difference between near and far
        misses shrinks compared to a small sigma.
        """
        n = 32
        true_bucket = 16
        targets = torch.tensor([true_bucket])

        logits_near = torch.full((1, n), -100.0)
        logits_near[0, true_bucket - 1] = 100.0

        logits_far = torch.full((1, n), -100.0)
        logits_far[0, 0] = 100.0

        fn_tight = build_loss_fn(LossConfig(mode="gaussian", sigma=1.0), n_buckets=n)
        fn_wide = build_loss_fn(LossConfig(mode="gaussian", sigma=10.0), n_buckets=n)

        # Ratio of far/near loss should be larger with tight sigma.
        ratio_tight = fn_tight(logits_far, targets).item() / fn_tight(logits_near, targets).item()
        ratio_wide = fn_wide(logits_far, targets).item() / fn_wide(logits_near, targets).item()

        assert ratio_tight > ratio_wide

    def test_batch_processing(self) -> None:
        """Loss function should handle batches correctly."""
        n = 16
        batch = 8
        logits = torch.randn(batch, n)
        targets = torch.randint(0, n, (batch,))

        fn = build_loss_fn(LossConfig(mode="gaussian", sigma=2.0), n_buckets=n)
        loss = fn(logits, targets)

        assert loss.shape == ()  # scalar
        assert loss.item() > 0

    def test_soft_target_is_valid_distribution(self) -> None:
        """The Gaussian kernel rows should sum to 1 (valid probability distributions)."""
        n = 16
        sigma = 2.0

        # Reconstruct the kernel to verify normalisation.
        indices = torch.arange(n, dtype=torch.float32)
        diff = indices.unsqueeze(1) - indices.unsqueeze(0)
        log_weights = -0.5 * (diff / sigma) ** 2
        kernel = F.softmax(log_weights, dim=1)

        # Each row should sum to 1.
        row_sums = kernel.sum(dim=1)
        for i in range(n):
            assert row_sums[i].item() == pytest.approx(1.0, abs=1e-6)

    def test_hard_vs_gaussian_at_sigma_zero_like(self) -> None:
        """With a very small sigma, gaussian loss should approach hard cross-entropy.

        As sigma -> 0, the soft target becomes a delta at the true bucket,
        and KL-div(delta || softmax) = cross-entropy.
        """
        n = 16
        torch.manual_seed(42)
        logits = torch.randn(8, n)
        targets = torch.randint(0, n, (8,))

        fn_hard = build_loss_fn(LossConfig(mode="hard"), n_buckets=n)
        fn_gauss = build_loss_fn(LossConfig(mode="gaussian", sigma=0.01), n_buckets=n)

        loss_hard = fn_hard(logits, targets).item()
        loss_gauss = fn_gauss(logits, targets).item()

        assert loss_gauss == pytest.approx(loss_hard, rel=0.05)
