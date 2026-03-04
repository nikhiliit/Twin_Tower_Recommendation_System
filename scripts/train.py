"""Main training entry point for the two-tower retrieval model.

Usage:
    python scripts/train.py --config configs/base_config.yaml
    python scripts/train.py --config configs/bpr_config.yaml
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from src.data.data_loader import create_data_loader
from src.data.dataset import MovieLensDataset
from src.data.feature_store import FeatureStore
from src.data.negative_sampler import NegativeSampler
from src.evaluation.evaluator import Evaluator
from src.losses.bpr_loss import BPRLoss
from src.losses.hard_negative_loss import HardNegativeLoss
from src.losses.in_batch_softmax import InBatchSoftmaxLoss
from src.models.two_tower import TwoTowerModel
from src.serving.index_builder import FAISSIndexBuilder
from src.training.optimizer import create_optimizer, create_scheduler
from src.training.trainer import Trainer
from src.utils import mlflow_utils
from src.utils.device_utils import (
    detect_device,
    get_optimal_batch_size,
    get_optimal_workers,
    print_system_info,
)
from src.utils.logging_utils import setup_logging
from src.utils.seed_utils import set_global_seed

logger = logging.getLogger(__name__)


def load_config(
    config_path: str,
    base_path: str = "configs/base_config.yaml",
) -> dict[str, Any]:
    """Load and merge configuration files.

    Loads the base config and merges with experiment-specific
    overrides.

    Args:
        config_path: Path to experiment config.
        base_path: Path to base config.

    Returns:
        Merged configuration dictionary.
    """
    with open(base_path, "r") as f:
        base = yaml.safe_load(f) or {}

    if config_path != base_path:
        with open(config_path, "r") as f:
            overrides = yaml.safe_load(f) or {}
        base = _deep_merge(base, overrides)

    return base


def _deep_merge(
    base: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Deep merge two dictionaries (overrides take priority).

    Args:
        base: Base dictionary.
        overrides: Override dictionary.

    Returns:
        Merged dictionary.
    """
    merged = base.copy()
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_loss_fn(config: dict[str, Any]) -> torch.nn.Module:
    """Create loss function from config.

    Args:
        config: Full configuration dictionary.

    Returns:
        Loss function module.

    Raises:
        ValueError: If loss type is unknown.
    """
    loss_type = config.get("training", {}).get(
        "loss", "in_batch_softmax"
    )
    loss_cfg = config.get("loss", {})

    if loss_type == "bpr":
        return BPRLoss(reduction="mean")
    elif loss_type == "in_batch_softmax":
        return InBatchSoftmaxLoss(
            temperature=loss_cfg.get("temperature", 0.07),
            use_frequency_correction=loss_cfg.get(
                "use_frequency_correction", True
            ),
        )
    elif loss_type == "hard_negative":
        return HardNegativeLoss(
            temperature=loss_cfg.get("temperature", 0.07),
            hard_neg_ratio=loss_cfg.get("hard_neg_ratio", 0.3),
            use_frequency_correction=loss_cfg.get(
                "use_frequency_correction", True
            ),
            initial_temperature=loss_cfg.get(
                "initial_temperature", None
            ),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train Two-Tower Retrieval Model."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to experiment config.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Key=value config overrides.",
    )
    args = parser.parse_args()

    setup_logging()

    # Load config.
    config = load_config(args.config)

    # Apply CLI overrides.
    for override in args.overrides:
        key, value = override.split("=", 1)
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Try to parse value as number.
        try:
            d[keys[-1]] = int(value)
        except ValueError:
            try:
                d[keys[-1]] = float(value)
            except ValueError:
                d[keys[-1]] = value

    # Set seed.
    seed = config.get("experiment", {}).get("seed", 42)
    set_global_seed(seed)

    # Detect device.
    preferred_device = config.get("experiment", {}).get(
        "device", "auto"
    )
    device_info = detect_device(preferred=preferred_device)
    device_str = device_info.device_type
    print_system_info(device_info)

    # Override config with detected settings.
    config["experiment"]["device"] = device_str

    # Initialize MLflow.
    mlflow_cfg = config.get("mlflow", {})
    experiment_id = mlflow_utils.init_mlflow(
        tracking_uri=mlflow_cfg.get("tracking_uri", "mlruns/"),
        experiment_name=mlflow_cfg.get(
            "experiment_name", "two_tower_retrieval"
        ),
    )

    run_name = config.get("experiment", {}).get(
        "name", "two_tower"
    )

    with mlflow_utils.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    ):
        # Log all config params.
        mlflow_utils.log_params_flat(config)

        # Load processed data.
        data_cfg = config.get("data", {})
        processed_dir = Path(
            data_cfg.get("processed_dir", "data/processed")
        )

        train_df = pd.read_parquet(
            processed_dir / "train.parquet"
        )
        val_df = pd.read_parquet(processed_dir / "val.parquet")
        test_df = pd.read_parquet(processed_dir / "test.parquet")

        # Load feature store.
        with open(
            processed_dir / "feature_store.pkl", "rb"
        ) as f:
            feature_store: FeatureStore = pickle.load(f)

        logger.info(
            "Loaded data: %d train, %d val, %d test, "
            "%d users, %d items.",
            len(train_df),
            len(val_df),
            len(test_df),
            feature_store.n_users,
            feature_store.n_items,
        )

        # Build negative sampler (for BPR loss).
        loss_type = config.get("training", {}).get(
            "loss", "in_batch_softmax"
        )
        loss_cfg = config.get("loss", {})
        negative_sampler = None
        if loss_type in ("bpr", "hard_negative"):
            strategy = (
                "uniform"
                if loss_type == "bpr"
                else "dynamic_hard"
            )
            negative_sampler = NegativeSampler(
                n_items=feature_store.n_items,
                strategy=strategy,
                n_negatives=loss_cfg.get("bpr_n_negatives", 1),
                hard_neg_ratio=loss_cfg.get(
                    "hard_neg_ratio", 0.3
                ),
                item_frequencies=feature_store.item_frequencies,
            )

        # Build datasets.
        train_dataset = MovieLensDataset(
            train_df, feature_store, negative_sampler
        )
        val_dataset = MovieLensDataset(
            val_df, feature_store, None
        )

        # Build data loaders.
        train_cfg = config.get("training", {})
        train_loader = create_data_loader(
            train_dataset,
            batch_size=train_cfg.get("batch_size", 1024),
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 4),
            drop_last=True,
        )
        val_loader = create_data_loader(
            val_dataset,
            batch_size=train_cfg.get("batch_size", 1024),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 4),
        )

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
            use_genome=model_cfg.get(
                "use_genome_features", True
            ),
            history_length=model_cfg.get("history_length", 50),
        )

        # Build loss, optimizer, scheduler.
        loss_fn = build_loss_fn(config)
        optimizer = create_optimizer(
            model,
            learning_rate=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            epochs=train_cfg.get("epochs", 20),
        )

        # Build evaluator.
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

        # Train.
        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            evaluator=evaluator,
            config=config,
            feature_store=feature_store,
        )

        best_metrics = trainer.train()

        # Build FAISS index with best checkpoint.
        logger.info("Building final FAISS index...")
        checkpoint_dir = Path(
            train_cfg.get("checkpoint_dir", "checkpoints/")
        )
        best_ckpt = checkpoint_dir / "best_model.pt"
        if best_ckpt.exists():
            ckpt = torch.load(
                best_ckpt, map_location="cpu"
            )
            model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded best checkpoint.")

        item_embeddings = evaluator.compute_all_item_embeddings()
        import numpy as np

        index_builder = FAISSIndexBuilder(
            index_type=faiss_cfg.get("index_type", "IVFFlat"),
            embedding_dim=model_cfg.get("embedding_dim", 64),
            nlist=faiss_cfg.get("nlist", 100),
            nprobe=faiss_cfg.get("nprobe", 10),
            metric=faiss_cfg.get("metric", "inner_product"),
        )
        item_ids = np.arange(len(item_embeddings))
        index_builder.build(item_embeddings, item_ids)

        index_path = str(checkpoint_dir / "faiss_index")
        index_builder.save(index_path)

        # Benchmark latency.
        latency = index_builder.benchmark_latency()

        # Log artifacts.
        try:
            mlflow_utils.log_artifact(
                str(best_ckpt)
            )
        except Exception:
            pass

        # Final evaluation on test set.
        logger.info("Running final evaluation on test set...")
        test_dataset = MovieLensDataset(
            test_df, feature_store, None
        )
        test_user_indices = np.array(
            test_dataset.user_indices, dtype=np.int64
        )
        test_ground_truth = np.array(
            test_dataset.item_indices, dtype=np.int64
        )

        # Deduplicate.
        unique_test: dict[int, int] = {}
        for i in range(len(test_user_indices)):
            unique_test[test_user_indices[i]] = (
                test_ground_truth[i]
            )
        test_users = np.array(
            list(unique_test.keys()), dtype=np.int64
        )
        test_gt = np.array(
            list(unique_test.values()), dtype=np.int64
        )

        test_activity = np.array(
            [
                len(
                    feature_store.user_histories.get(uid, [])
                )
                for uid in test_users
            ],
            dtype=np.int64,
        )

        final_results = evaluator.evaluate(
            eval_user_indices=test_users,
            eval_ground_truth=test_gt,
            user_activity=test_activity,
        )

        # Log final metrics.
        try:
            mlflow_utils.log_metrics(
                {
                    f"test_{k}": v
                    for k, v in final_results.get(
                        "overall", {}
                    ).items()
                }
            )
        except Exception:
            pass

        # Print summary.
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Experiment: {run_name}")
        print(f"Loss: {loss_type}")
        print(f"Device: {device_str}")
        print()
        print("Best Validation Metrics:")
        for k, v in best_metrics.items():
            print(f"  {k}: {v:.4f}")
        print()
        print("Test Metrics:")
        for k, v in final_results.get("overall", {}).items():
            print(f"  {k}: {v:.4f}")
        print()
        print("Latency Benchmarks:")
        for k, v in latency.items():
            print(f"  {k}: {v:.2f}ms")
        print("=" * 60)


if __name__ == "__main__":
    main()
