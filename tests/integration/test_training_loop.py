"""Integration test for the training loop.

Creates a tiny synthetic dataset and runs a few training steps
to verify the full pipeline works end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import MovieLensDataset
from src.data.data_loader import create_data_loader
from src.data.feature_store import FeatureStore
from src.data.negative_sampler import NegativeSampler
from src.losses.bpr_loss import BPRLoss
from src.losses.in_batch_softmax import InBatchSoftmaxLoss
from src.models.two_tower import TwoTowerModel
from src.training.optimizer import create_optimizer, create_scheduler


def _create_synthetic_feature_store(
    n_users: int = 20,
    n_items: int = 30,
    n_genres: int = 5,
    genome_dim: int = 16,
) -> FeatureStore:
    """Create a minimal synthetic feature store."""
    return FeatureStore(
        genre_matrix=np.random.rand(
            n_items, n_genres
        ).astype(np.float32),
        genre_vocab=[f"genre_{i}" for i in range(n_genres)],
        genome_matrix=np.random.rand(
            n_items, genome_dim
        ).astype(np.float32),
        year_array=np.random.rand(n_items, 1).astype(
            np.float32
        ),
        user_stats=np.random.rand(n_users, 2).astype(
            np.float32
        ),
        item_id_map={i: i for i in range(n_items)},
        user_id_map={i: i for i in range(n_users)},
        item_frequencies=np.random.rand(n_items).astype(
            np.float32
        ),
        user_histories={
            i: list(range(min(5, n_items)))
            for i in range(n_users)
        },
    )


def _create_synthetic_interactions(
    n_users: int = 20,
    n_items: int = 30,
    n_interactions: int = 100,
) -> pd.DataFrame:
    """Create synthetic interactions."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "userId": rng.integers(0, n_users, n_interactions),
        "movieId": rng.integers(0, n_items, n_interactions),
        "rating": rng.uniform(3.5, 5.0, n_interactions),
        "timestamp": np.arange(n_interactions),
    })


class TestTrainingLoopIntegration:
    """Integration tests for the training loop."""

    def test_bpr_forward_backward(self) -> None:
        """Full forward-backward pass with BPR loss."""
        n_users, n_items, n_genres = 20, 30, 5
        genome_dim = 16

        fs = _create_synthetic_feature_store(
            n_users, n_items, n_genres, genome_dim
        )
        interactions = _create_synthetic_interactions(
            n_users, n_items
        )
        sampler = NegativeSampler(
            n_items=n_items, strategy="uniform", n_negatives=1
        )
        dataset = MovieLensDataset(
            interactions, fs, sampler
        )
        loader = create_data_loader(
            dataset, batch_size=16, shuffle=True,
            num_workers=0,
        )

        model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            n_genres=n_genres,
            genome_dim=genome_dim,
            embedding_dim=16,
            user_hidden_dims=[32],
            item_hidden_dims=[32],
            use_genome=True,
        )

        loss_fn = BPRLoss()
        optimizer = create_optimizer(model, learning_rate=1e-3)

        # Run 2 steps.
        model.train()
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            output = model(batch)
            # Simple BPR with shuffled negatives.
            B = output["user_emb"].shape[0]
            perm = torch.randperm(B)
            loss = loss_fn(
                output["user_emb"],
                output["pos_item_emb"],
                output["pos_item_emb"][perm],
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            assert loss.item() > 0

    def test_inbatch_forward_backward(self) -> None:
        """Full forward-backward pass with in-batch softmax."""
        n_users, n_items, n_genres = 20, 30, 5
        genome_dim = 16

        fs = _create_synthetic_feature_store(
            n_users, n_items, n_genres, genome_dim
        )
        interactions = _create_synthetic_interactions(
            n_users, n_items
        )
        dataset = MovieLensDataset(interactions, fs, None)
        loader = create_data_loader(
            dataset, batch_size=16, shuffle=True,
            num_workers=0,
        )

        model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            n_genres=n_genres,
            genome_dim=genome_dim,
            embedding_dim=16,
            user_hidden_dims=[32],
            item_hidden_dims=[32],
        )

        loss_fn = InBatchSoftmaxLoss(temperature=0.1)
        optimizer = create_optimizer(model, learning_rate=1e-3)

        model.train()
        for i, batch in enumerate(loader):
            if i >= 2:
                break
            output = model(batch)
            loss, metrics = loss_fn(
                output["user_emb"],
                output["pos_item_emb"],
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            assert loss.item() > 0
            assert "in_batch_accuracy" in metrics

    def test_model_output_shapes(self) -> None:
        """Verify model output shapes are correct."""
        n_users, n_items = 10, 15
        emb_dim = 16

        fs = _create_synthetic_feature_store(
            n_users, n_items, 5, 16
        )
        interactions = _create_synthetic_interactions(
            n_users, n_items, 50
        )
        dataset = MovieLensDataset(interactions, fs, None)
        loader = create_data_loader(
            dataset, batch_size=8, shuffle=False,
            num_workers=0,
        )

        model = TwoTowerModel(
            n_users=n_users,
            n_items=n_items,
            n_genres=5,
            genome_dim=16,
            embedding_dim=emb_dim,
            user_hidden_dims=[32],
            item_hidden_dims=[32],
        )

        batch = next(iter(loader))
        output = model(batch)
        B = batch["user_idx"].shape[0]
        assert output["user_emb"].shape == (B, emb_dim)
        assert output["pos_item_emb"].shape == (B, emb_dim)

        # Check L2 normalization.
        norms = output["user_emb"].norm(dim=1)
        torch.testing.assert_close(
            norms, torch.ones(B), atol=1e-5, rtol=1e-5
        )
