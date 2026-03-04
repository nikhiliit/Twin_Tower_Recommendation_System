"""Regression test for metric stability.

Verifies that metrics computed on a fixed synthetic dataset produce
known expected values, guarding against accidental changes in
metric computation logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


# Fixed synthetic data for regression testing.
# 5 users, retrieved list of 10 items each.
RETRIEVED = np.array([
    [0, 5, 3, 7, 1, 9, 2, 8, 4, 6],
    [2, 0, 4, 6, 8, 1, 3, 5, 7, 9],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    [1, 3, 5, 7, 9, 0, 2, 4, 6, 8],
    [4, 2, 6, 8, 0, 1, 3, 5, 7, 9],
])

# Ground truth: one positive per user.
GROUND_TRUTH = np.array([0, 4, 0, 5, 4])


class TestMetricRegression:
    """Regression tests ensuring metric values remain stable."""

    def test_recall_at_5_stable(self) -> None:
        """Recall@5 should match pre-computed value."""
        result = recall_at_k(RETRIEVED, GROUND_TRUTH, k=5)
        # User 0: 0 at pos 0 -> hit
        # User 1: 4 at pos 2 -> hit
        # User 2: 0 at pos 9 -> miss (not in top 5)
        # User 3: 5 at pos 2 -> hit
        # User 4: 4 at pos 0 -> hit
        expected = 4 / 5  # 0.8
        assert abs(result - expected) < 1e-6

    def test_recall_at_10_stable(self) -> None:
        """Recall@10 should be 1.0 (all items found)."""
        result = recall_at_k(RETRIEVED, GROUND_TRUTH, k=10)
        assert abs(result - 1.0) < 1e-6

    def test_mrr_stable(self) -> None:
        """MRR should match pre-computed value."""
        result = mean_reciprocal_rank(RETRIEVED, GROUND_TRUTH)
        # User 0: rank 1 -> 1/1
        # User 1: rank 3 -> 1/3
        # User 2: rank 10 -> 1/10
        # User 3: rank 3 -> 1/3
        # User 4: rank 1 -> 1/1
        expected = (1.0 + 1 / 3 + 1 / 10 + 1 / 3 + 1.0) / 5
        assert abs(result - expected) < 1e-6

    def test_ndcg_at_5_stable(self) -> None:
        """NDCG@5 should match pre-computed value."""
        result = ndcg_at_k(RETRIEVED, GROUND_TRUTH, k=5)
        # User 0: rank 1 -> 1/log2(2) = 1.0
        # User 1: rank 3 -> 1/log2(4)
        # User 2: not in top 5 -> 0
        # User 3: rank 3 -> 1/log2(4)
        # User 4: rank 1 -> 1.0
        expected = (
            1.0
            + 1.0 / np.log2(4)
            + 0.0
            + 1.0 / np.log2(4)
            + 1.0
        ) / 5
        assert abs(result - expected) < 1e-6

    def test_all_metrics_consistency(self) -> None:
        """compute_all_metrics should be consistent with
        individual functions."""
        all_metrics = compute_all_metrics(
            RETRIEVED, GROUND_TRUTH, k_values=[5, 10]
        )
        assert abs(
            all_metrics["recall_at_5"]
            - recall_at_k(RETRIEVED, GROUND_TRUTH, k=5)
        ) < 1e-6
        assert abs(
            all_metrics["mrr"]
            - mean_reciprocal_rank(RETRIEVED, GROUND_TRUTH)
        ) < 1e-6
