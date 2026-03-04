"""Offline evaluator for the two-tower retrieval model.

Computes item embeddings for the full catalog, builds a FAISS index,
retrieves top-K candidates for each user, and computes metrics.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.feature_store import FeatureStore
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_cohort_metrics,
)
from src.models.two_tower import TwoTowerModel
from src.serving.index_builder import FAISSIndexBuilder

logger = logging.getLogger(__name__)


class Evaluator:
    """Offline evaluator for two-tower retrieval.

    Runs the full evaluation pipeline:
    1. Encode all items with the item tower.
    2. Build a FAISS index over item embeddings.
    3. Encode all eval users with the user tower.
    4. Retrieve top-K candidates via FAISS.
    5. Compute recall, MRR, NDCG, and cohort metrics.

    Args:
        model: Trained ``TwoTowerModel``.
        feature_store: Pre-computed feature store.
        k_values: List of K values for metrics.
        primary_metric: Metric name used for model selection.
        cohort_thresholds: Activity thresholds for cohort eval.
        device: Torch device.
        faiss_config: Optional FAISS index configuration dict.
    """

    def __init__(
        self,
        model: TwoTowerModel,
        feature_store: FeatureStore,
        k_values: list[int] | None = None,
        primary_metric: str = "recall_at_50",
        cohort_thresholds: list[int] | None = None,
        device: str = "cpu",
        faiss_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.feature_store = feature_store
        self.k_values = k_values or [10, 20, 50, 100]
        self.primary_metric = primary_metric
        self.cohort_thresholds = cohort_thresholds or [5, 20]
        self.device = torch.device(device)
        self.faiss_config = faiss_config or {
            "index_type": "Flat",
            "metric": "inner_product",
        }

    @torch.no_grad()
    def compute_all_item_embeddings(
        self,
        batch_size: int = 2048,
    ) -> np.ndarray:
        """Compute embeddings for all items in the catalog.

        Args:
            batch_size: Batch size for item encoding.

        Returns:
            Item embeddings ``[n_items, D]`` as NumPy array.
        """
        self.model.eval()
        fs = self.feature_store
        n_items = fs.n_items
        emb_dim = self.model.embedding_dim

        all_embeddings = np.zeros(
            (n_items, emb_dim), dtype=np.float32
        )

        for start in range(0, n_items, batch_size):
            end = min(start + batch_size, n_items)
            indices = np.arange(start, end)

            item_ids = torch.tensor(
                indices, dtype=torch.long, device=self.device
            )
            genre = torch.tensor(
                fs.genre_matrix[indices],
                dtype=torch.float32,
                device=self.device,
            )
            genome = torch.tensor(
                fs.genome_matrix[indices],
                dtype=torch.float32,
                device=self.device,
            )
            year = torch.tensor(
                fs.year_array[indices],
                dtype=torch.float32,
                device=self.device,
            )

            emb = self.model.encode_items(
                item_ids, genre, genome, year
            )
            all_embeddings[start:end] = emb.cpu().numpy()

        logger.info(
            "Computed embeddings for %d items.", n_items
        )
        return all_embeddings

    @torch.no_grad()
    def compute_user_embeddings(
        self,
        user_indices: np.ndarray,
        batch_size: int = 2048,
    ) -> np.ndarray:
        """Compute embeddings for specified users.

        Args:
            user_indices: Array of user indices to encode.
            batch_size: Batch size for user encoding.

        Returns:
            User embeddings ``[n_users, D]`` as NumPy array.
        """
        self.model.eval()
        fs = self.feature_store
        emb_dim = self.model.embedding_dim
        n_users = len(user_indices)

        all_embeddings = np.zeros(
            (n_users, emb_dim), dtype=np.float32
        )
        history_length = 50

        for start in range(0, n_users, batch_size):
            end = min(start + batch_size, n_users)
            batch_indices = user_indices[start:end]
            B = len(batch_indices)

            user_ids = torch.tensor(
                batch_indices, dtype=torch.long,
                device=self.device,
            )
            user_stats = torch.tensor(
                fs.user_stats[batch_indices],
                dtype=torch.float32,
                device=self.device,
            )

            # Build history tensors.
            hist_ids = np.zeros(
                (B, history_length), dtype=np.int64
            )
            hist_mask = np.zeros(
                (B, history_length), dtype=np.float32
            )
            for i, uid in enumerate(batch_indices):
                history = fs.user_histories.get(uid, [])
                if history:
                    padded = history[-history_length:]
                    hist_ids[i, : len(padded)] = padded
                    hist_mask[i, : len(padded)] = 1.0

            hist_ids_t = torch.tensor(
                hist_ids, dtype=torch.long,
                device=self.device,
            )
            hist_mask_t = torch.tensor(
                hist_mask, dtype=torch.float32,
                device=self.device,
            )

            emb = self.model.encode_users(
                user_ids, hist_ids_t, hist_mask_t, user_stats
            )
            all_embeddings[start:end] = emb.cpu().numpy()

        return all_embeddings

    def evaluate(
        self,
        eval_user_indices: np.ndarray,
        eval_ground_truth: np.ndarray,
        user_activity: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation pipeline.

        Args:
            eval_user_indices: Contiguous user indices to
                evaluate ``[n_eval]``.
            eval_ground_truth: True positive item indices for
                each eval user ``[n_eval]``.
            user_activity: Number of training interactions per
                eval user ``[n_eval]``. Required for cohort eval.

        Returns:
            Dictionary with ``overall``, ``cohort``, and
            ``latency`` metrics.
        """
        start_time = time.time()

        # Step 1: Compute all item embeddings.
        item_embeddings = self.compute_all_item_embeddings()

        # Step 2: Build FAISS index.
        max_k = max(self.k_values)
        index_builder = FAISSIndexBuilder(
            index_type=self.faiss_config.get(
                "index_type", "Flat"
            ),
            embedding_dim=self.model.embedding_dim,
            nlist=self.faiss_config.get("nlist", 100),
            nprobe=self.faiss_config.get("nprobe", 10),
            metric=self.faiss_config.get(
                "metric", "inner_product"
            ),
        )
        item_ids = np.arange(len(item_embeddings))
        index_builder.build(item_embeddings, item_ids)

        # Step 3: Encode eval users.
        user_embeddings = self.compute_user_embeddings(
            eval_user_indices
        )

        # Step 4: Retrieve top-K.
        _, retrieved = index_builder.search(
            user_embeddings, k=max_k
        )

        # Step 5: Compute metrics.
        overall_metrics = compute_all_metrics(
            retrieved, eval_ground_truth, self.k_values
        )

        result: dict[str, Any] = {
            "overall": overall_metrics,
        }

        # Cohort metrics.
        if user_activity is not None:
            cohort_metrics = compute_cohort_metrics(
                retrieved,
                eval_ground_truth,
                user_activity,
                self.cohort_thresholds,
                self.k_values,
            )
            result["cohort"] = cohort_metrics

        elapsed = time.time() - start_time
        result["latency"] = {
            "total_eval_seconds": elapsed,
            "n_eval_users": len(eval_user_indices),
        }

        logger.info(
            "Evaluation complete in %.1fs. %s=%.4f.",
            elapsed,
            self.primary_metric,
            overall_metrics.get(self.primary_metric, 0.0),
        )

        return result
