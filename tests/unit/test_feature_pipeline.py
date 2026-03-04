"""Unit tests for the feature pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.feature_store import (
    build_id_maps,
    compute_item_frequencies,
    compute_user_stats,
    extract_year,
)
from src.data.preprocessing import (
    binarize_interactions,
    encode_genres,
    temporal_train_val_test_split,
)


class TestBinarize:
    """Tests for binarize_interactions."""

    def test_threshold_filtering(self) -> None:
        """Only ratings >= threshold should be kept."""
        df = pd.DataFrame({
            "userId": [1, 1, 2, 2],
            "movieId": [10, 20, 10, 30],
            "rating": [4.0, 2.0, 5.0, 3.0],
            "timestamp": [1, 2, 3, 4],
        })
        result = binarize_interactions(df, threshold=3.5)
        assert len(result) == 2
        assert all(result["label"] == 1)
        assert set(result["rating"].values) == {4.0, 5.0}

    def test_empty_result(self) -> None:
        """Very high threshold should return empty DataFrame."""
        df = pd.DataFrame({
            "userId": [1],
            "movieId": [10],
            "rating": [3.0],
            "timestamp": [1],
        })
        result = binarize_interactions(df, threshold=4.0)
        assert len(result) == 0


class TestTemporalSplit:
    """Tests for temporal_train_val_test_split."""

    def test_split_sizes(self) -> None:
        """Verify correct split with min_interactions."""
        df = pd.DataFrame({
            "userId": [1] * 6 + [2] * 6,
            "movieId": list(range(12)),
            "rating": [4.0] * 12,
            "timestamp": list(range(12)),
            "label": [1] * 12,
        })
        train, val, test = temporal_train_val_test_split(
            df, test_size=1, val_size=1, min_interactions=3
        )
        assert len(test) == 2  # 1 per user.
        assert len(val) == 2
        assert len(train) == 8  # 4 per user.

    def test_filter_low_activity_users(self) -> None:
        """Users with < min_interactions should be dropped."""
        df = pd.DataFrame({
            "userId": [1, 1, 2],
            "movieId": [10, 20, 30],
            "rating": [4.0, 4.0, 4.0],
            "timestamp": [1, 2, 3],
            "label": [1, 1, 1],
        })
        train, val, test = temporal_train_val_test_split(
            df, test_size=1, val_size=1, min_interactions=5
        )
        # Both users have < 5 interactions.
        assert len(train) == 0
        assert len(val) == 0
        assert len(test) == 0


class TestEncodeGenres:
    """Tests for encode_genres."""

    def test_multi_hot(self) -> None:
        """Pipe-separated genres should create multi-hot vectors."""
        movies = pd.DataFrame({
            "movieId": [1, 2],
            "genres": ["Action|Comedy", "Drama"],
        })
        matrix, vocab = encode_genres(movies)
        assert matrix.shape == (2, len(vocab))
        assert matrix.dtype == np.float32
        # First movie has 2 genres.
        assert matrix[0].sum() == 2
        # Second movie has 1 genre.
        assert matrix[1].sum() == 1


class TestIDMaps:
    """Tests for build_id_maps."""

    def test_contiguous_ids(self) -> None:
        """IDs should be contiguous starting from 0."""
        df = pd.DataFrame({
            "userId": [10, 20, 10],
            "movieId": [100, 200, 300],
        })
        user_map, item_map = build_id_maps(df)
        assert set(user_map.values()) == {0, 1}
        assert set(item_map.values()) == {0, 1, 2}


class TestUserStats:
    """Tests for compute_user_stats."""

    def test_stats_shape(self) -> None:
        """Should return [n_users, 2] array."""
        df = pd.DataFrame({
            "userId": [0, 0, 1],
            "movieId": [10, 20, 30],
            "rating": [4.0, 2.0, 5.0],
        })
        user_map = {0: 0, 1: 1}
        stats = compute_user_stats(df, user_map)
        assert stats.shape == (2, 2)
        assert stats.dtype == np.float32


class TestExtractYear:
    """Tests for extract_year."""

    def test_valid_year(self) -> None:
        """Standard format should extract year."""
        assert extract_year("Toy Story (1995)") == 1995.0

    def test_no_year(self) -> None:
        """No year pattern should return 0.0."""
        assert extract_year("No Year Movie") == 0.0

    def test_multiple_parens(self) -> None:
        """Should extract first year match."""
        assert extract_year("Test (ABC) (2001)") == 2001.0
