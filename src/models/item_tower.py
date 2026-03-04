"""Item tower encoder for two-tower retrieval.

Encodes item features (ID embedding, genre multi-hot, genome
scores, release year) into a fixed-size L2-normalized embedding.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ItemTower(nn.Module):
    """Item encoder for two-tower retrieval.

    Architecture::

        [item_id_emb | genre_emb | genome_emb | year_normalized]
        -> Linear -> BatchNorm -> ReLU -> Dropout
        -> Linear
        -> L2 normalize
        -> 64-dim unit vector

    Args:
        n_items: Total number of unique items.
        n_genres: Number of genre categories (multi-hot dim).
        genome_dim: Dimensionality of genome feature vector (1128).
        embedding_dim: Output embedding dimensionality.
        hidden_dims: List of hidden layer sizes.
        dropout: Dropout rate.
        use_genome: Whether to include genome features.
    """

    def __init__(
        self,
        n_items: int,
        n_genres: int,
        genome_dim: int = 1128,
        embedding_dim: int = 64,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        use_genome: bool = True,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [192, 128]

        self.embedding_dim = embedding_dim
        self.use_genome = use_genome

        # Item ID embedding.
        self.item_embedding = nn.Embedding(
            n_items, embedding_dim
        )

        # Genre projection.
        self.genre_proj = nn.Linear(n_genres, embedding_dim)

        # Genome projection (optional).
        if use_genome:
            self.genome_proj = nn.Linear(
                genome_dim, embedding_dim
            )

        # Input dim: item_emb + genre_emb + (genome_emb) + year.
        input_dim = embedding_dim + embedding_dim + 1
        if use_genome:
            input_dim += embedding_dim

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
            "ItemTower: n_items=%d, n_genres=%d, genome_dim=%d, "
            "embedding_dim=%d, use_genome=%s.",
            n_items,
            n_genres,
            genome_dim,
            embedding_dim,
            use_genome,
        )

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.normal_(
            self.item_embedding.weight, mean=0.0, std=0.01
        )
        nn.init.xavier_uniform_(self.genre_proj.weight)
        if self.use_genome:
            nn.init.xavier_uniform_(self.genome_proj.weight)

    def forward(
        self,
        item_ids: torch.Tensor,
        genre_features: torch.Tensor,
        genome_features: torch.Tensor,
        year_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass. Returns L2-normalized item embeddings.

        Args:
            item_ids: Item indices ``[B]`` or ``[N]`` for full
                corpus.
            genre_features: Multi-hot genre vectors
                ``[B, n_genres]``.
            genome_features: Genome score vectors
                ``[B, genome_dim]``.
            year_features: Normalized release year ``[B, 1]``.

        Returns:
            L2-normalized item embeddings ``[B, embedding_dim]``.
        """
        # Item ID embedding: [B, D].
        i_emb = self.item_embedding(item_ids)

        # Genre projection: [B, D].
        g_emb = self.genre_proj(genre_features)

        # Concatenate features.
        features = [i_emb, g_emb]

        if self.use_genome:
            gn_emb = self.genome_proj(genome_features)
            features.append(gn_emb)

        features.append(year_features)
        x = torch.cat(features, dim=-1)

        # MLP projection.
        x = self.mlp(x)

        # L2 normalization.
        x = F.normalize(x, p=2, dim=-1)

        return x
