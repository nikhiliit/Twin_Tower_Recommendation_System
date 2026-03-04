"""Unit tests for negative sampler."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.negative_sampler import NegativeSampler


class TestUniformSampler:
    """Tests for uniform negative sampling."""

    def test_correct_count(self) -> None:
        """Sampler returns exactly n_negatives items."""
        sampler = NegativeSampler(
            n_items=100, strategy="uniform", n_negatives=5
        )
        rng = np.random.default_rng(42)
        negs = sampler.sample(0, {0, 1, 2}, rng)
        assert len(negs) == 5

    def test_excludes_positives(self) -> None:
        """Sampled items should not be in the positive set."""
        sampler = NegativeSampler(
            n_items=1000, strategy="uniform", n_negatives=10
        )
        rng = np.random.default_rng(42)
        positives = {0, 1, 2, 3, 4}
        negs = sampler.sample(0, positives, rng)
        for neg in negs:
            assert neg not in positives

    def test_valid_range(self) -> None:
        """All sampled items within [0, n_items)."""
        sampler = NegativeSampler(
            n_items=50, strategy="uniform", n_negatives=20
        )
        rng = np.random.default_rng(42)
        negs = sampler.sample(0, set(), rng)
        assert all(0 <= n < 50 for n in negs)

    def test_reproducibility(self) -> None:
        """Same seed produces same results."""
        sampler = NegativeSampler(
            n_items=100, strategy="uniform", n_negatives=5
        )
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        negs1 = sampler.sample(0, set(), rng1)
        negs2 = sampler.sample(0, set(), rng2)
        np.testing.assert_array_equal(negs1, negs2)


class TestHardNegativeSampler:
    """Tests for hard negative sampling."""

    def test_mixed_sampling(self) -> None:
        """With hard neg pool, returns mix of hard and uniform."""
        sampler = NegativeSampler(
            n_items=100,
            strategy="static_hard",
            n_negatives=10,
            hard_neg_ratio=0.3,
        )
        # Set up pool.
        pool = {0: np.array([50, 51, 52, 53, 54])}
        sampler.update_hard_negative_pool(pool)

        rng = np.random.default_rng(42)
        negs = sampler.sample(0, {0, 1, 2}, rng)
        assert len(negs) == 10

    def test_without_pool(self) -> None:
        """Without pool, falls back to uniform."""
        sampler = NegativeSampler(
            n_items=100,
            strategy="static_hard",
            n_negatives=5,
            hard_neg_ratio=0.5,
        )
        rng = np.random.default_rng(42)
        negs = sampler.sample(0, set(), rng)
        assert len(negs) == 5


class TestBatchSampling:
    """Tests for batch-level negative sampling."""

    def test_batch_shape(self) -> None:
        """Batch sampling returns correct shape."""
        sampler = NegativeSampler(
            n_items=100, strategy="uniform", n_negatives=3
        )
        user_indices = np.array([0, 1, 2, 3])
        user_positives = {
            0: {10},
            1: {20},
            2: {30},
            3: {40},
        }
        rng = np.random.default_rng(42)
        batch = sampler.sample_batch(
            user_indices, user_positives, rng
        )
        assert batch.shape == (4, 3)
