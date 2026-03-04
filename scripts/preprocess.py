"""Preprocessing script for MovieLens-25M.

Runs the full preprocessing pipeline:
1. Load and validate ratings.
2. Binarize interactions.
3. Temporal train/val/test split.
4. Build feature store.
5. Save processed data to disk.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.feature_store import build_feature_store
from src.data.preprocessing import (
    binarize_interactions,
    load_and_validate_ratings,
    temporal_train_val_test_split,
)
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import set_global_seed

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    """Main preprocessing entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens-25M dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to configuration file.",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    set_global_seed(config.get("experiment", {}).get("seed", 42))

    data_cfg = config.get("data", {})
    raw_dir = Path(data_cfg.get("raw_dir", "data/raw"))
    processed_dir = Path(
        data_cfg.get("processed_dir", "data/processed")
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and validate ratings.
    ratings_path = raw_dir / "ml-25m" / "ratings.csv"
    ratings_df = load_and_validate_ratings(str(ratings_path))

    # Step 2: Binarize.
    threshold = data_cfg.get("rating_threshold", 3.5)
    pos_df = binarize_interactions(ratings_df, threshold)

    # Step 3: Temporal split.
    train_df, val_df, test_df = temporal_train_val_test_split(
        pos_df,
        test_size=data_cfg.get("test_size", 1),
        val_size=data_cfg.get("val_size", 1),
        min_interactions=data_cfg.get("min_interactions", 5),
    )

    # Save splits.
    train_df.to_parquet(processed_dir / "train.parquet")
    val_df.to_parquet(processed_dir / "val.parquet")
    test_df.to_parquet(processed_dir / "test.parquet")
    logger.info("Saved train/val/test splits to %s.", processed_dir)

    # Step 4: Build feature store.
    movies_path = raw_dir / "ml-25m" / "movies.csv"
    movies_df = pd.read_csv(str(movies_path))

    genome_path = raw_dir / "ml-25m" / "genome-scores.csv"
    genome_df = None
    if genome_path.exists():
        genome_df = pd.read_csv(str(genome_path))

    history_length = config.get("model", {}).get(
        "history_length", 50
    )
    feature_store = build_feature_store(
        train_df, movies_df, genome_df, history_length
    )

    # Save feature store.
    fs_path = processed_dir / "feature_store.pkl"
    with open(fs_path, "wb") as f:
        pickle.dump(feature_store, f)
    logger.info("Saved feature store to %s.", fs_path)

    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
