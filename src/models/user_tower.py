"""User tower encoder for two-tower retrieval.

Encodes user features (ID embedding, interaction history, and
statistics) into a fixed-size L2-normalized embedding vector.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class UserTower(nn.Module):
    """User encoder for two-tower retrieval.

    Architecture::

        [user_id_emb | history_mean_emb | user_stats]
        -> Linear -> BatchNorm -> ReLU -> Dropout
        -> Linear
        -> L2 normalize
        -> 64-dim unit vector

    Args:
        n_users: Total number of unique users.
        n_items: Total number of unique items (for history
            embedding lookup).
        embedding_dim: Output embedding dimensionality.
        hidden_dims: List of hidden layer sizes.
        history_length: Max number of past interactions to consider.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        history_length: int = 50,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.embedding_dim = embedding_dim
        self.history_length = history_length

        # User ID embedding.
        self.user_embedding = nn.Embedding(
            n_users, embedding_dim
        )

        # Shared item embedding for history aggregation.
        self.history_embedding = nn.Embedding(
            n_items, embedding_dim, padding_idx=0
        )

        # Input dim: user_emb + history_emb + user_stats (2).
        input_dim = embedding_dim + embedding_dim + 2

        # MLP layers.
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights()

        logger.info(
            "UserTower: n_users=%d, embedding_dim=%d, "
            "hidden_dims=%s.",
            n_users,
            embedding_dim,
            hidden_dims,
        )

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(
            self.user_embedding.weight, mean=0.0, std=0.01
        )
        nn.init.normal_(
            self.history_embedding.weight, mean=0.0, std=0.01
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        history_item_ids: torch.Tensor,
        history_mask: torch.Tensor,
        user_stats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns L2-normalized user embeddings.

        Args:
            user_ids: User indices ``[B]``.
            history_item_ids: Item history indices
                ``[B, history_length]``.
            history_mask: Binary mask for valid history entries
                ``[B, history_length]``.
            user_stats: User statistics ``[B, 2]``
                (avg_rating, log_activity).

        Returns:
            L2-normalized user embeddings ``[B, embedding_dim]``.
        """
        # User ID embedding: [B, D].
        u_emb = self.user_embedding(user_ids)

        # History aggregation: masked mean pooling.
        # [B, history_length, D].
        hist_emb = self.history_embedding(history_item_ids)
        # [B, history_length, 1].
        mask = history_mask.unsqueeze(-1)
        hist_sum = (hist_emb * mask).sum(dim=1)  # [B, D]
        hist_count = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        hist_mean = hist_sum / hist_count  # [B, D]

        # Concatenate all features.
        x = torch.cat([u_emb, hist_mean, user_stats], dim=-1)

        # MLP projection.
        x = self.mlp(x)

        # L2 normalization.
        x = F.normalize(x, p=2, dim=-1)

        return x
