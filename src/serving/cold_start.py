"""Popularity-based cold-start fallback handler.

For users not present in the feature store (new users with no
interaction history), returns the globally most popular items
ranked by interaction frequency from training data.
"""

from __future__ import annotations

import logging

import numpy as np

from src.data.feature_store import FeatureStore

logger = logging.getLogger(__name__)


class ColdStartHandler:
    """Handles recommendations for unknown / cold-start users.

    Pre-computes a ranked list of popular items from the feature
    store's item frequency array at construction time so that
    cold-start responses are O(1) at serve time.

    Args:
        feature_store: Loaded ``FeatureStore`` with item frequencies
            and ID maps.
    """

    def __init__(self, feature_store: FeatureStore) -> None:
        self.feature_store = feature_store

        # Build reverse map: contiguous item index → original movie ID.
        self._idx_to_movie_id: dict[int, int] = {
            v: k for k, v in feature_store.item_id_map.items()
        }

        # Pre-rank items by frequency descending.
        freqs = feature_store.item_frequencies
        self._popular_indices: np.ndarray = np.argsort(freqs)[
            ::-1
        ].copy()

        logger.info(
            "ColdStartHandler ready. Top item index=%d, "
            "freq=%.2f.",
            self._popular_indices[0],
            freqs[self._popular_indices[0]],
        )

    def is_cold_start(self, user_id: int) -> bool:
        """Check whether a user ID is unknown to the feature store.

        Args:
            user_id: Original (external) user ID.

        Returns:
            ``True`` if the user is not in the feature store.
        """
        return user_id not in self.feature_store.user_id_map

    def get_popular_items(self, k: int = 50) -> list[int]:
        """Return top-k most popular original movie IDs.

        Args:
            k: Number of items to return.

        Returns:
            List of original movie IDs ranked by popularity.
        """
        top_indices = self._popular_indices[:k]
        return [
            self._idx_to_movie_id[int(idx)]
            for idx in top_indices
            if int(idx) in self._idx_to_movie_id
        ][:k]
