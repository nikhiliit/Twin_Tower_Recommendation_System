"""Main Trainer class for the two-tower retrieval model.

Orchestrates the full training loop including gradient clipping,
validation, checkpointing, MLflow logging, dynamic hard negative
mining, and early stopping.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.feature_store import FeatureStore
from src.evaluation.evaluator import Evaluator
from src.losses.bpr_loss import BPRLoss
from src.losses.hard_negative_loss import HardNegativeLoss
from src.losses.in_batch_softmax import InBatchSoftmaxLoss
from src.serving.index_builder import FAISSIndexBuilder
from src.training.callbacks import (
    EarlyStopping,
    GradientMonitor,
    ModelCheckpoint,
)
from src.utils import mlflow_utils

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the full training loop with experiment tracking.

    Responsibilities:
        - Training loop with gradient clipping.
        - Validation after each epoch.
        - Checkpointing (model + optimizer + epoch + best metric).
        - MLflow logging (params, metrics, artifacts).
        - Dynamic hard negative mining (rebuild FAISS each epoch).
        - Early stopping.
        - Gradient norm monitoring.

    Args:
        model: ``TwoTowerModel`` instance.
        loss_fn: Loss function instance.
        optimizer: PyTorch optimizer.
        scheduler: LR scheduler.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        evaluator: ``Evaluator`` instance.
        config: Full experiment config dict.
        feature_store: Feature store for item lookups.
        callbacks: Optional list of callback instances.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        evaluator: Evaluator,
        config: dict[str, Any],
        feature_store: FeatureStore,
        callbacks: list[Any] | None = None,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator
        self.config = config
        self.feature_store = feature_store

        # Device.
        device_str = config.get("experiment", {}).get(
            "device", "cpu"
        )
        self.device = torch.device(device_str)
        self.model.to(self.device)

        # Training config.
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 20)
        self.gradient_clip_norm = train_cfg.get(
            "gradient_clip_norm", 1.0
        )
        self.log_every_n_steps = train_cfg.get(
            "log_every_n_steps", 100
        )
        self.checkpoint_dir = train_cfg.get(
            "checkpoint_dir", "checkpoints/"
        )

        # Loss config.
        loss_cfg = config.get("loss", {})
        self.dynamic_mining = loss_cfg.get(
            "dynamic_mining", False
        )
        self.warmup_epochs = loss_cfg.get(
            "warmup_epochs", 0
        )

        # Callbacks.
        primary_metric = config.get("evaluation", {}).get(
            "primary_metric", "recall_at_50"
        )
        patience = train_cfg.get("early_stopping_patience", 3)

        self.early_stopping = EarlyStopping(
            patience=patience,
            metric_name=primary_metric,
        )
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            metric_name=primary_metric,
        )
        self.grad_monitor = GradientMonitor(
            log_interval=self.log_every_n_steps
        )

        self.best_metrics: dict[str, float] = {}

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Best validation metrics dictionary.
        """
        logger.info(
            "Starting training for %d epochs on %s.",
            self.epochs,
            self.device,
        )

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            # Temperature annealing for hard negative loss.
            if hasattr(self.loss_fn, "set_epoch"):
                self.loss_fn.set_epoch(epoch, self.epochs)

            # Log warmup status.
            in_warmup = (
                epoch <= self.warmup_epochs
                and isinstance(self.loss_fn, HardNegativeLoss)
            )
            if in_warmup:
                logger.info(
                    "Epoch %d/%d — WARMUP (in-batch only, "
                    "no hard negatives).",
                    epoch,
                    self.epochs,
                )
            elif (
                epoch == self.warmup_epochs + 1
                and self.warmup_epochs > 0
                and isinstance(self.loss_fn, HardNegativeLoss)
            ):
                logger.info(
                    "Warmup complete. Introducing hard "
                    "negatives from epoch %d.",
                    epoch,
                )

            # Train one epoch.
            train_loss = self._train_epoch(epoch)

            # Validate.
            val_metrics = self._validate(epoch)

            # Scheduler step.
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_time = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d — train_loss=%.4f, %s=%.4f, "
                "lr=%.2e, time=%.1fs.",
                epoch,
                self.epochs,
                train_loss,
                self.evaluator.primary_metric,
                val_metrics.get(
                    self.evaluator.primary_metric, 0.0
                ),
                current_lr,
                epoch_time,
            )

            # MLflow logging.
            try:
                mlflow_utils.log_metrics(
                    {
                        "train_loss": train_loss,
                        "learning_rate": current_lr,
                        **{
                            f"val_{k}": v
                            for k, v in val_metrics.items()
                        },
                    },
                    step=epoch,
                )
            except Exception:
                pass  # MLflow not required.

            # Checkpoint.
            self.checkpoint_callback(
                self.model, self.optimizer, epoch, val_metrics
            )

            # Update best metrics.
            primary_val = val_metrics.get(
                self.evaluator.primary_metric, 0.0
            )
            best_val = self.best_metrics.get(
                self.evaluator.primary_metric, 0.0
            )
            if primary_val > best_val:
                self.best_metrics = val_metrics.copy()

            # Early stopping.
            if self.early_stopping(val_metrics):
                logger.info(
                    "Early stopping triggered at epoch %d.",
                    epoch,
                )
                break

            # Dynamic hard negative mining (skip during warmup).
            if (
                self.dynamic_mining
                and isinstance(self.loss_fn, HardNegativeLoss)
                and epoch > self.warmup_epochs
            ):
                self._rebuild_hard_negative_index()

        logger.info(
            "Training complete. Best %s=%.4f.",
            self.evaluator.primary_metric,
            self.best_metrics.get(
                self.evaluator.primary_metric, 0.0
            ),
        )
        return self.best_metrics

    def _train_epoch(self, epoch: int) -> float:
        """Train for a single epoch.

        Args:
            epoch: Current epoch number (1-indexed).

        Returns:
            Mean training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.grad_monitor.reset()
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for step, batch in enumerate(pbar):
            # Move batch to device.
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            # Forward pass.
            self.optimizer.zero_grad()

            output = self.model(batch)
            user_emb = output["user_emb"]
            pos_item_emb = output["pos_item_emb"]

            # Compute loss based on loss function type.
            if isinstance(self.loss_fn, BPRLoss):
                loss = self._compute_bpr_loss(
                    user_emb, pos_item_emb, batch
                )
                metrics = {}
            elif isinstance(self.loss_fn, InBatchSoftmaxLoss):
                item_freq = self._get_item_frequencies(batch)
                loss, metrics = self.loss_fn(
                    user_emb, pos_item_emb, item_freq
                )
            elif isinstance(self.loss_fn, HardNegativeLoss):
                loss, metrics = self._compute_hard_neg_loss(
                    user_emb, pos_item_emb, batch, epoch
                )
            else:
                raise ValueError(
                    f"Unknown loss function: "
                    f"{type(self.loss_fn)}"
                )

            # Backward pass.
            loss.backward()

            # Gradient monitoring.
            grad_info = self.grad_monitor(self.model)

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip_norm,
            )

            # Optimizer step.
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Progress bar update.
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                grad=f"{grad_info['grad_norm']:.3f}",
            )

            # Step-level MLflow logging.
            if (step + 1) % self.log_every_n_steps == 0:
                try:
                    global_step = (
                        (epoch - 1) * len(self.train_loader)
                        + step
                    )
                    mlflow_utils.log_metrics(
                        {
                            "step_loss": loss.item(),
                            "step_grad_norm": (
                                grad_info["grad_norm"]
                            ),
                        },
                        step=global_step,
                    )
                except Exception:
                    pass

        return total_loss / max(n_batches, 1)

    def _validate(self, epoch: int) -> dict[str, float]:
        """Run validation evaluation.

        Args:
            epoch: Current epoch number.

        Returns:
            Validation metrics dictionary.
        """
        # Extract eval user indices and ground truth from val
        # loader's dataset.
        val_dataset = self.val_loader.dataset
        eval_user_indices = np.array(
            val_dataset.user_indices, dtype=np.int64
        )
        eval_ground_truth = np.array(
            val_dataset.item_indices, dtype=np.int64
        )

        # Deduplicate: take last entry per user.
        unique_users: dict[int, int] = {}
        for i in range(len(eval_user_indices)):
            unique_users[eval_user_indices[i]] = (
                eval_ground_truth[i]
            )

        user_indices = np.array(
            list(unique_users.keys()), dtype=np.int64
        )
        ground_truth = np.array(
            list(unique_users.values()), dtype=np.int64
        )

        # Compute user activity for cohort metrics.
        user_activity = np.array(
            [
                len(
                    self.feature_store.user_histories.get(
                        uid, []
                    )
                )
                for uid in user_indices
            ],
            dtype=np.int64,
        )

        result = self.evaluator.evaluate(
            eval_user_indices=user_indices,
            eval_ground_truth=ground_truth,
            user_activity=user_activity,
        )

        return result.get("overall", {})

    def _compute_bpr_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute BPR loss with negative item embeddings.

        Args:
            user_emb: User embeddings ``[B, D]``.
            pos_item_emb: Positive item embeddings ``[B, D]``.
            batch: Full batch dictionary.

        Returns:
            Scalar BPR loss.
        """
        neg_indices = batch.get("neg_item_indices")
        if neg_indices is None or neg_indices.numel() == 0:
            # Fall back: random negatives from batch.
            B = user_emb.shape[0]
            perm = torch.randperm(B, device=user_emb.device)
            neg_item_emb = pos_item_emb[perm]
        else:
            # Get first negative per sample.
            neg_idx = neg_indices[:, 0]
            fs = self.feature_store
            neg_genre = torch.tensor(
                fs.genre_matrix[neg_idx.cpu().numpy()],
                dtype=torch.float32,
                device=self.device,
            )
            neg_genome = torch.tensor(
                fs.genome_matrix[neg_idx.cpu().numpy()],
                dtype=torch.float32,
                device=self.device,
            )
            neg_year = torch.tensor(
                fs.year_array[neg_idx.cpu().numpy()],
                dtype=torch.float32,
                device=self.device,
            )
            neg_item_emb = self.model.encode_items(
                neg_idx, neg_genre, neg_genome, neg_year
            )

        return self.loss_fn(user_emb, pos_item_emb, neg_item_emb)

    def _compute_hard_neg_loss(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        batch: dict[str, torch.Tensor],
        epoch: int = 1,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute hard negative loss.

        Args:
            user_emb: User embeddings ``[B, D]``.
            pos_item_emb: Positive item embeddings ``[B, D]``.
            batch: Full batch dictionary.

        Returns:
            Tuple of ``(loss, metrics_dict)``.
        """
        # For hard negatives, use in-batch as base.
        # Hard negatives come from the sampler if available.
        # During warmup, skip hard negatives entirely.
        in_warmup = epoch <= self.warmup_epochs
        neg_indices = batch.get("neg_item_indices")
        if (
            neg_indices is not None
            and neg_indices.numel() > 0
            and not in_warmup
        ):
            B, K = neg_indices.shape
            flat_neg = neg_indices.reshape(-1)
            fs = self.feature_store
            neg_cpu = flat_neg.cpu().numpy()
            neg_genre = torch.tensor(
                fs.genre_matrix[neg_cpu],
                dtype=torch.float32,
                device=self.device,
            )
            neg_genome = torch.tensor(
                fs.genome_matrix[neg_cpu],
                dtype=torch.float32,
                device=self.device,
            )
            neg_year = torch.tensor(
                fs.year_array[neg_cpu],
                dtype=torch.float32,
                device=self.device,
            )
            hard_neg_emb = self.model.encode_items(
                flat_neg.to(self.device),
                neg_genre,
                neg_genome,
                neg_year,
            )
            hard_neg_emb = hard_neg_emb.reshape(B, K, -1)
        else:
            hard_neg_emb = torch.empty(
                (user_emb.shape[0], 0, user_emb.shape[1]),
                device=self.device,
            )

        item_freq = self._get_item_frequencies(batch)
        return self.loss_fn(
            user_emb,
            pos_item_emb,
            pos_item_emb,  # in-batch items.
            hard_neg_emb,
            item_freq,
        )

    def _get_item_frequencies(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        """Get item frequencies for the batch.

        Args:
            batch: Batch dictionary with ``item_idx``.

        Returns:
            Log-frequency tensor ``[B]`` or ``None``.
        """
        item_indices = batch["item_idx"].cpu().numpy()
        freqs = self.feature_store.item_frequencies[item_indices]
        return torch.tensor(
            freqs, dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def _rebuild_hard_negative_index(self) -> None:
        """Rebuild FAISS index for dynamic hard negative mining.

        Computes all item embeddings with current model weights
        and rebuilds the FAISS index. For each user, retrieves
        top-M candidates to use as hard negatives in the next
        epoch.
        """
        logger.info("Rebuilding hard negative FAISS index...")

        item_embeddings = self.evaluator.compute_all_item_embeddings()

        loss_cfg = self.config.get("loss", {})
        pool_size = loss_cfg.get("hard_neg_pool_size", 50)

        index_builder = FAISSIndexBuilder(
            index_type="Flat",
            embedding_dim=self.model.embedding_dim,
            metric="inner_product",
        )
        item_ids = np.arange(len(item_embeddings))
        index_builder.build(item_embeddings, item_ids)

        # Get unique users from training data.
        train_dataset = self.train_loader.dataset
        unique_users = np.unique(train_dataset.user_indices)

        user_embeddings = self.evaluator.compute_user_embeddings(
            unique_users
        )
        _, retrieved = index_builder.search(
            user_embeddings, k=pool_size
        )

        # Update sampler if available.
        if hasattr(train_dataset, "negative_sampler"):
            sampler = train_dataset.negative_sampler
            if sampler is not None:
                user_hard_negs = {
                    uid: retrieved[i]
                    for i, uid in enumerate(unique_users)
                }
                sampler.update_hard_negative_pool(user_hard_negs)

        logger.info(
            "Hard negative index rebuilt for %d users.",
            len(unique_users),
        )
