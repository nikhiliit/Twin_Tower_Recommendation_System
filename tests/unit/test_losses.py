"""Unit tests for loss functions."""

from __future__ import annotations

import torch
import pytest

from src.losses.bpr_loss import BPRLoss
from src.losses.in_batch_softmax import InBatchSoftmaxLoss
from src.losses.hard_negative_loss import HardNegativeLoss


class TestBPRLoss:
    """Tests for BPR loss."""

    def test_positive_loss(self) -> None:
        """BPR loss should always be non-negative."""
        loss_fn = BPRLoss(reduction="mean")
        user_emb = torch.randn(8, 64)
        pos_emb = torch.randn(8, 64)
        neg_emb = torch.randn(8, 64)
        loss = loss_fn(user_emb, pos_emb, neg_emb)
        assert loss.item() >= 0

    def test_perfect_ranking(self) -> None:
        """When pos >> neg, loss should be near zero."""
        loss_fn = BPRLoss(reduction="mean")
        user_emb = torch.ones(4, 64)
        pos_emb = torch.ones(4, 64) * 10.0
        neg_emb = torch.ones(4, 64) * -10.0
        loss = loss_fn(user_emb, pos_emb, neg_emb)
        assert loss.item() < 0.01

    def test_bad_ranking(self) -> None:
        """When neg >> pos, loss should be high."""
        loss_fn = BPRLoss(reduction="mean")
        user_emb = torch.ones(4, 64)
        pos_emb = torch.ones(4, 64) * -10.0
        neg_emb = torch.ones(4, 64) * 10.0
        loss = loss_fn(user_emb, pos_emb, neg_emb)
        assert loss.item() > 1.0

    def test_reduction_sum(self) -> None:
        """Sum reduction should give larger values than mean."""
        loss_fn_mean = BPRLoss(reduction="mean")
        loss_fn_sum = BPRLoss(reduction="sum")
        user_emb = torch.randn(8, 64)
        pos_emb = torch.randn(8, 64)
        neg_emb = torch.randn(8, 64)
        loss_mean = loss_fn_mean(user_emb, pos_emb, neg_emb)
        loss_sum = loss_fn_sum(user_emb, pos_emb, neg_emb)
        assert loss_sum.item() >= loss_mean.item()

    def test_gradient_flow(self) -> None:
        """Gradients should flow back through the loss."""
        loss_fn = BPRLoss()
        user_emb = torch.randn(4, 32, requires_grad=True)
        pos_emb = torch.randn(4, 32, requires_grad=True)
        neg_emb = torch.randn(4, 32, requires_grad=True)
        loss = loss_fn(user_emb, pos_emb, neg_emb)
        loss.backward()
        assert user_emb.grad is not None
        assert pos_emb.grad is not None
        assert neg_emb.grad is not None


class TestInBatchSoftmaxLoss:
    """Tests for in-batch softmax loss."""

    def test_output_type(self) -> None:
        """Should return (loss, metrics_dict)."""
        loss_fn = InBatchSoftmaxLoss(temperature=0.1)
        user_emb = torch.randn(8, 64)
        item_emb = torch.randn(8, 64)
        loss, metrics = loss_fn(user_emb, item_emb)
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)
        assert "avg_pos_sim" in metrics
        assert "in_batch_accuracy" in metrics

    def test_positive_loss(self) -> None:
        """Loss should be non-negative (cross-entropy)."""
        loss_fn = InBatchSoftmaxLoss(temperature=0.1)
        user_emb = torch.randn(8, 32)
        item_emb = torch.randn(8, 32)
        loss, _ = loss_fn(user_emb, item_emb)
        assert loss.item() >= 0

    def test_with_frequency_correction(self) -> None:
        """Should work with frequency correction enabled."""
        loss_fn = InBatchSoftmaxLoss(
            temperature=0.1,
            use_frequency_correction=True,
        )
        user_emb = torch.randn(8, 32)
        item_emb = torch.randn(8, 32)
        freqs = torch.rand(8)
        loss, metrics = loss_fn(user_emb, item_emb, freqs)
        assert loss.item() >= 0

    def test_frequency_required(self) -> None:
        """Should raise if freq correction on but no freqs."""
        loss_fn = InBatchSoftmaxLoss(
            use_frequency_correction=True,
        )
        user_emb = torch.randn(4, 32)
        item_emb = torch.randn(4, 32)
        with pytest.raises(ValueError):
            loss_fn(user_emb, item_emb, None)

    def test_gradient_flow(self) -> None:
        """Gradients should flow."""
        loss_fn = InBatchSoftmaxLoss(temperature=0.1)
        user_emb = torch.randn(4, 32, requires_grad=True)
        item_emb = torch.randn(4, 32, requires_grad=True)
        loss, _ = loss_fn(user_emb, item_emb)
        loss.backward()
        assert user_emb.grad is not None


class TestHardNegativeLoss:
    """Tests for hard negative loss."""

    def test_output_type(self) -> None:
        """Should return (loss, metrics_dict)."""
        loss_fn = HardNegativeLoss(temperature=0.1)
        B, D, K = 8, 32, 3
        user_emb = torch.randn(B, D)
        pos_emb = torch.randn(B, D)
        in_batch_emb = torch.randn(B, D)
        hard_neg_emb = torch.randn(B, K, D)
        loss, metrics = loss_fn(
            user_emb, pos_emb, in_batch_emb, hard_neg_emb
        )
        assert isinstance(loss, torch.Tensor)
        assert "avg_pos_sim" in metrics
        assert "n_hard_negatives" in metrics

    def test_no_hard_negatives(self) -> None:
        """Should work with empty hard negatives."""
        loss_fn = HardNegativeLoss(temperature=0.1)
        B, D = 4, 16
        user_emb = torch.randn(B, D)
        pos_emb = torch.randn(B, D)
        in_batch_emb = torch.randn(B, D)
        hard_neg_emb = torch.empty(B, 0, D)
        loss, metrics = loss_fn(
            user_emb, pos_emb, in_batch_emb, hard_neg_emb
        )
        assert loss.item() >= 0
        assert metrics["n_hard_negatives"] == 0

    def test_gradient_flow(self) -> None:
        """Gradients should flow through all inputs."""
        loss_fn = HardNegativeLoss(temperature=0.1)
        B, D, K = 4, 16, 2
        user_emb = torch.randn(B, D, requires_grad=True)
        pos_emb = torch.randn(B, D, requires_grad=True)
        in_batch_emb = torch.randn(B, D, requires_grad=True)
        hard_neg_emb = torch.randn(
            B, K, D, requires_grad=True
        )
        loss, _ = loss_fn(
            user_emb, pos_emb, in_batch_emb, hard_neg_emb
        )
        loss.backward()
        assert user_emb.grad is not None
        assert hard_neg_emb.grad is not None
