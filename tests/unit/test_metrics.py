"""Unit tests for evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_all_metrics,
    compute_cohort_metrics,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


class TestRecallAtK:
    """Tests for recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """All ground truth items at position 0 → recall = 1.0."""
        retrieved = np.array([[0, 1, 2], [3, 4, 5]])
        ground_truth = np.array([0, 3])
        assert recall_at_k(retrieved, ground_truth, k=3) == 1.0

    def test_zero_recall(self) -> None:
        """No hits → recall = 0.0."""
        retrieved = np.array([[1, 2, 3], [4, 5, 6]])
        ground_truth = np.array([0, 7])
        assert recall_at_k(retrieved, ground_truth, k=3) == 0.0

    def test_partial_recall(self) -> None:
        """One of two users has a hit → recall = 0.5."""
        retrieved = np.array([[0, 1, 2], [4, 5, 6]])
        ground_truth = np.array([0, 7])
        assert recall_at_k(retrieved, ground_truth, k=3) == 0.5

    def test_k_boundary(self) -> None:
        """Hit at position k-1 → counts; at k → doesn't."""
        retrieved = np.array([[1, 2, 0]])
        ground_truth = np.array([0])
        assert recall_at_k(retrieved, ground_truth, k=3) == 1.0
        assert recall_at_k(retrieved, ground_truth, k=2) == 0.0

    def test_invalid_k(self) -> None:
        """k > list length should raise ValueError."""
        retrieved = np.array([[0, 1]])
        ground_truth = np.array([0])
        with pytest.raises(ValueError):
            recall_at_k(retrieved, ground_truth, k=5)


class TestMRR:
    """Tests for mean_reciprocal_rank function."""

    def test_perfect_mrr(self) -> None:
        """All items at rank 1 → MRR = 1.0."""
        retrieved = np.array([[5, 1, 2], [3, 4, 6]])
        ground_truth = np.array([5, 3])
        assert mean_reciprocal_rank(retrieved, ground_truth) == 1.0

    def test_known_mrr(self) -> None:
        """Item at rank 1 and 3 → MRR = (1 + 1/3) / 2."""
        retrieved = np.array([[5, 1, 2], [4, 6, 3]])
        ground_truth = np.array([5, 3])
        expected = (1.0 + 1.0 / 3) / 2
        assert abs(
            mean_reciprocal_rank(retrieved, ground_truth)
            - expected
        ) < 1e-6

    def test_not_found(self) -> None:
        """Items not in list → MRR includes 0 for those."""
        retrieved = np.array([[1, 2, 3], [4, 5, 6]])
        ground_truth = np.array([1, 99])
        expected = (1.0 + 0.0) / 2
        assert abs(
            mean_reciprocal_rank(retrieved, ground_truth)
            - expected
        ) < 1e-6


class TestNDCG:
    """Tests for ndcg_at_k function."""

    def test_perfect_ndcg(self) -> None:
        """Item at rank 1 → NDCG = 1.0."""
        retrieved = np.array([[5, 1, 2]])
        ground_truth = np.array([5])
        assert abs(
            ndcg_at_k(retrieved, ground_truth, k=3) - 1.0
        ) < 1e-6

    def test_rank_two(self) -> None:
        """Item at rank 2 → NDCG = 1/log2(3)."""
        retrieved = np.array([[1, 5, 2]])
        ground_truth = np.array([5])
        expected = 1.0 / np.log2(3)
        assert abs(
            ndcg_at_k(retrieved, ground_truth, k=3) - expected
        ) < 1e-6

    def test_not_in_top_k(self) -> None:
        """Item not in top-K → NDCG = 0."""
        retrieved = np.array([[1, 2, 3, 5]])
        ground_truth = np.array([5])
        assert ndcg_at_k(retrieved, ground_truth, k=3) == 0.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""

    def test_all_metrics_keys(self) -> None:
        """Check that all expected metric keys are present."""
        retrieved = np.array([[0, 1, 2, 3, 4]])
        ground_truth = np.array([0])
        metrics = compute_all_metrics(
            retrieved, ground_truth, k_values=[2, 4]
        )
        assert "recall_at_2" in metrics
        assert "recall_at_4" in metrics
        assert "ndcg_at_2" in metrics
        assert "ndcg_at_4" in metrics
        assert "mrr" in metrics


class TestCohortMetrics:
    """Tests for compute_cohort_metrics function."""

    def test_cohort_split(self) -> None:
        """Verify correct splitting into cohorts."""
        retrieved = np.array([
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 3, 0, 1],
        ])
        ground_truth = np.array([0, 1, 0])
        user_activity = np.array([3, 10, 25])

        result = compute_cohort_metrics(
            retrieved,
            ground_truth,
            user_activity,
            thresholds=[5, 20],
            k_values=[2, 4],
        )

        assert "cold" in result
        assert "medium" in result
        assert "power" in result
        assert result["cold"]["n_users"] == 1.0
        assert result["medium"]["n_users"] == 1.0
        assert result["power"]["n_users"] == 1.0
