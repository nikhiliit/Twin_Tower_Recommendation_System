"""Two-tower model combining user and item encoders.

Provides a unified interface for computing user and item embeddings
and similarity scores for training and inference.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from src.models.item_tower import ItemTower
from src.models.user_tower import UserTower

logger = logging.getLogger(__name__)


class TwoTowerModel(nn.Module):
    """Two-tower retrieval model combining user and item towers.

    Computes L2-normalized embeddings from both towers and
    similarity scores via inner product (= cosine similarity
    after L2 normalization).

    Args:
        n_users: Total number of unique users.
        n_items: Total number of unique items.
        n_genres: Number of genre categories.
        genome_dim: Dimensionality of genome features.
        embedding_dim: Output embedding dimensionality.
        user_hidden_dims: Hidden layer sizes for user tower.
        item_hidden_dims: Hidden layer sizes for item tower.
        dropout: Dropout rate for both towers.
        use_genome: Whether to use genome features.
        history_length: Max interaction history length.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_genres: int,
        genome_dim: int = 1128,
        embedding_dim: int = 64,
        user_hidden_dims: list[int] | None = None,
        item_hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_genome: bool = True,
        history_length: int = 50,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim

        self.user_tower = UserTower(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            hidden_dims=user_hidden_dims,
            history_length=history_length,
            dropout=dropout,
        )

        self.item_tower = ItemTower(
            n_items=n_items,
            n_genres=n_genres,
            genome_dim=genome_dim,
            embedding_dim=embedding_dim,
            hidden_dims=item_hidden_dims,
            dropout=dropout,
            use_genome=use_genome,
        )

        n_user_params = sum(
            p.numel() for p in self.user_tower.parameters()
        )
        n_item_params = sum(
            p.numel() for p in self.item_tower.parameters()
        )
        logger.info(
            "TwoTowerModel: user_params=%d, item_params=%d, "
            "total=%d.",
            n_user_params,
            n_item_params,
            n_user_params + n_item_params,
        )

    def encode_users(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        user_stats: torch.Tensor,
    ) -> torch.Tensor:
        """Compute user embeddings.

        Args:
            user_ids: User indices ``[B]``.
            history_item_ids: History item indices
                ``[B, history_length]``.
            history_mask: History mask ``[B, history_length]``.
            user_stats: User stats ``[B, 2]``.

        Returns:
            L2-normalized user embeddings ``[B, D]``.
        """
        return self.user_tower(
            user_ids, history_item_ids, history_mask, user_stats
        )

    def encode_items(
        self,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor,
        genome_features: torch.Tensor,
        year_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute item embeddings.

        Args:
            item_ids: Item indices ``[B]`` or ``[N]``.
            genre_features: Genre multi-hot ``[B, n_genres]``.
            genome_features: Genome scores ``[B, genome_dim]``.
            year_features: Normalized year ``[B, 1]``.

        Returns:
            L2-normalized item embeddings ``[B, D]``.
        """
        return self.item_tower(
            item_ids, genre_features, genome_features,
            year_features,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Full forward pass for a training batch.

        Computes user and positive item embeddings from the batch.
        Optionally computes negative item embeddings if present.

        Args:
            batch: Collated batch dictionary from ``collate_fn``.

        Returns:
            Dictionary with keys:
                - ``user_emb``: ``[B, D]``
                - ``pos_item_emb``: ``[B, D]``
                - ``neg_item_emb``: ``[B, n_neg, D]`` (if negatives
                  present)
        """
        user_emb = self.encode_users(
            user_ids=batch["user_idx"],
            history_item_ids=batch["history_item_ids"],
            history_mask=batch["history_mask"],
            user_stats=batch["user_stats"],
        )

        pos_item_emb = self.encode_items(
            item_ids=batch["item_idx"],
            genre_features=batch["genre_features"],
            genome_features=batch["genome_features"],
            year_features=batch["year_features"],
        )

        output = {
            "user_emb": user_emb,
            "pos_item_emb": pos_item_emb,
        }

        # Handle explicit negatives (for BPR loss).
        neg_indices = batch.get("neg_item_indices")
        if neg_indices is not None and neg_indices.numel() > 0:
            B, n_neg = neg_indices.shape
            flat_neg = neg_indices.reshape(-1)

            # We need features for negative items.
            # These should be provided via the feature store
            # at training time.
            output["neg_item_indices"] = neg_indices

        return output

    def compute_similarity(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity scores via inner product.

        Since embeddings are L2-normalized, this equals cosine
        similarity.

        Args:
            user_emb: User embeddings ``[B, D]``.
            item_emb: Item embeddings ``[N, D]``.

        Returns:
            Similarity matrix ``[B, N]``.
        """
        return torch.matmul(user_emb, item_emb.t())
