"""FAISS index builder entry point.

Builds a FAISS index from a trained model checkpoint and saves it
to disk for serving.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.data.feature_store import FeatureStore
from src.evaluation.evaluator import Evaluator
from src.models.two_tower import TwoTowerModel
from src.serving.index_builder import FAISSIndexBuilder
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Main index building function."""
    parser = argparse.ArgumentParser(
        description="Build FAISS Index."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to experiment config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="checkpoints/faiss_index",
        help="Output base path for FAISS index.",
    )
    args = parser.parse_args()

    setup_logging()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    processed_dir = Path(
        data_cfg.get("processed_dir", "data/processed")
    )

    # Load feature store.
    with open(processed_dir / "feature_store.pkl", "rb") as f:
        feature_store: FeatureStore = pickle.load(f)

    # Build and load model.
    model_cfg = config.get("model", {})
    model = TwoTowerModel(
        n_users=feature_store.n_users,
        n_items=feature_store.n_items,
        n_genres=feature_store.n_genres,
        genome_dim=feature_store.genome_dim,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        user_hidden_dims=model_cfg.get(
            "user_hidden_dims", [256, 128]
        ),
        item_hidden_dims=model_cfg.get(
            "item_hidden_dims", [192, 128]
        ),
        dropout=model_cfg.get("dropout", 0.1),
        use_genome=model_cfg.get("use_genome_features", True),
        history_length=model_cfg.get("history_length", 50),
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from %s.", args.checkpoint)

    # Compute item embeddings.
    device_str = config.get("experiment", {}).get("device", "cpu")
    evaluator = Evaluator(
        model=model,
        feature_store=feature_store,
        device=device_str,
    )
    item_embeddings = evaluator.compute_all_item_embeddings()

    # Build FAISS index.
    faiss_cfg = config.get("faiss", {})
    index_builder = FAISSIndexBuilder(
        index_type=faiss_cfg.get("index_type", "IVFFlat"),
        embedding_dim=model_cfg.get("embedding_dim", 64),
        nlist=faiss_cfg.get("nlist", 100),
        nprobe=faiss_cfg.get("nprobe", 10),
        metric=faiss_cfg.get("metric", "inner_product"),
    )
    item_ids = np.arange(len(item_embeddings))
    index_builder.build(item_embeddings, item_ids)

    # Save.
    index_builder.save(args.output)

    # Benchmark.
    latency = index_builder.benchmark_latency()
    print(f"\nLatency: {latency}")


if __name__ == "__main__":
    main()
