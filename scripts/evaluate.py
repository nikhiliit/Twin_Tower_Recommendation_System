"""Offline evaluation entry point.

Loads a trained model checkpoint and runs evaluation on the test
set, including cohort-stratified metrics and latency benchmarks.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import MovieLensDataset
from src.data.feature_store import FeatureStore
from src.evaluation.evaluator import Evaluator
from src.models.two_tower import TwoTowerModel
from src.utils.device_utils import detect_device, print_system_info
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Two-Tower Retrieval Model."
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

    # Build model.
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

    # Load checkpoint.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from %s.", args.checkpoint)

    # Build evaluator.
    # Detect device.
    preferred_device = config.get("experiment", {}).get(
        "device", "auto"
    )
    device_info = detect_device(preferred=preferred_device)
    device_str = device_info.device_type
    print_system_info(device_info)

    eval_cfg = config.get("evaluation", {})
    faiss_cfg = config.get("faiss", {})

    evaluator = Evaluator(
        model=model,
        feature_store=feature_store,
        k_values=eval_cfg.get("k_values", [10, 20, 50, 100]),
        primary_metric=eval_cfg.get(
            "primary_metric", "recall_at_50"
        ),
        cohort_thresholds=eval_cfg.get(
            "cohort_thresholds", [5, 20]
        ),
        device=device_str,
        faiss_config=faiss_cfg,
    )

    # Load test data.
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    test_dataset = MovieLensDataset(test_df, feature_store, None)

    test_user_indices = np.array(
        test_dataset.user_indices, dtype=np.int64
    )
    test_ground_truth = np.array(
        test_dataset.item_indices, dtype=np.int64
    )

    # Deduplicate.
    unique_test: dict[int, int] = {}
    for i in range(len(test_user_indices)):
        unique_test[test_user_indices[i]] = test_ground_truth[i]

    test_users = np.array(
        list(unique_test.keys()), dtype=np.int64
    )
    test_gt = np.array(
        list(unique_test.values()), dtype=np.int64
    )

    test_activity = np.array(
        [
            len(feature_store.user_histories.get(uid, []))
            for uid in test_users
        ],
        dtype=np.int64,
    )

    results = evaluator.evaluate(
        eval_user_indices=test_users,
        eval_ground_truth=test_gt,
        user_activity=test_activity,
    )

    # Print results.
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print("\nOverall Metrics:")
    for k, v in results.get("overall", {}).items():
        print(f"  {k}: {v:.4f}")

    if "cohort" in results:
        print("\nCohort Metrics:")
        for cohort, metrics in results["cohort"].items():
            print(f"\n  {cohort}:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")

    print(f"\nLatency: {results.get('latency', {})}")
    print("=" * 60)


if __name__ == "__main__":
    main()
