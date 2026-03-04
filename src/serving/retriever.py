"""Two-tower retriever for inference.

Provides a high-level API for encoding users and retrieving
top-K items using a trained model and pre-built FAISS index.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.data.feature_store import FeatureStore
from src.models.two_tower import TwoTowerModel
from src.serving.index_builder import FAISSIndexBuilder

logger = logging.getLogger(__name__)


class TwoTowerRetriever:
    """Inference retriever for two-tower recommendations.

    Encodes a user with the user tower and retrieves the top-K
    nearest items from a pre-built FAISS index.

    Args:
        model: Trained ``TwoTowerModel``.
        feature_store: Pre-computed features.
        index_builder: Pre-built FAISS index.
        device: Torch device for inference.
    """

    def __init__(
        self,
        model: TwoTowerModel,
        feature_store: FeatureStore,
        index_builder: FAISSIndexBuilder,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.feature_store = feature_store
        self.index_builder = index_builder
        self.device = torch.device(device)

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def retrieve(
        self,
        user_idx: int,
        k: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-K items for a single user.

        Args:
            user_idx: Contiguous user index.
            k: Number of items to retrieve.

        Returns:
            Tuple of ``(scores [k], item_ids [k])``.

        Raises:
            ValueError: If ``user_idx`` is out of bounds.
        """
        fs = self.feature_store

        if user_idx < 0 or user_idx >= fs.n_users:
            raise ValueError(
                f"user_idx {user_idx} out of range "
                f"[0, {fs.n_users})."
            )

        # Build user features.
        user_ids = torch.tensor(
            [user_idx], dtype=torch.long, device=self.device
        )
        user_stats = torch.tensor(
            fs.user_stats[user_idx: user_idx + 1],
            dtype=torch.float32,
            device=self.device,
        )

        history = fs.user_histories.get(user_idx, [])
        hist_ids = np.zeros(50, dtype=np.int64)
        hist_mask = np.zeros(50, dtype=np.float32)
        if history:
            padded = history[-50:]
            hist_ids[: len(padded)] = padded
            hist_mask[: len(padded)] = 1.0

        hist_ids_t = torch.tensor(
            hist_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        hist_mask_t = torch.tensor(
            hist_mask, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        # Encode user.
        user_emb = self.model.encode_users(
            user_ids, hist_ids_t, hist_mask_t, user_stats
        )
        user_emb_np = user_emb.cpu().numpy()

        # Search FAISS index.
        scores, item_ids = self.index_builder.search(
            user_emb_np, k=k
        )

        return scores[0], item_ids[0]

    @torch.no_grad()
    def retrieve_batch(
        self,
        user_indices: np.ndarray,
        k: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve top-K items for a batch of users.

        Args:
            user_indices: Array of user indices ``[B]``.
            k: Number of items to retrieve per user.

        Returns:
            Tuple of ``(scores [B, k], item_ids [B, k])``.
        """
        fs = self.feature_store
        B = len(user_indices)

        user_ids = torch.tensor(
            user_indices, dtype=torch.long, device=self.device
        )
        user_stats = torch.tensor(
            fs.user_stats[user_indices],
            dtype=torch.float32,
            device=self.device,
        )

        hist_ids = np.zeros((B, 50), dtype=np.int64)
        hist_mask = np.zeros((B, 50), dtype=np.float32)
        for i, uid in enumerate(user_indices):
            history = fs.user_histories.get(uid, [])
            if history:
                padded = history[-50:]
                hist_ids[i, : len(padded)] = padded
                hist_mask[i, : len(padded)] = 1.0

        hist_ids_t = torch.tensor(
            hist_ids, dtype=torch.long, device=self.device
        )
        hist_mask_t = torch.tensor(
            hist_mask, dtype=torch.float32, device=self.device
        )

        user_emb = self.model.encode_users(
            user_ids, hist_ids_t, hist_mask_t, user_stats
        )
        user_emb_np = user_emb.cpu().numpy()

        scores, item_ids = self.index_builder.search(
            user_emb_np, k=k
        )

        return scores, item_ids

    def get_item_id_mapping(self) -> dict[int, int]:
        """Return the item ID mapping (original -> contiguous).

        Returns:
            Dictionary mapping original movie IDs to contiguous
            indices.
        """
        return self.feature_store.item_id_map
