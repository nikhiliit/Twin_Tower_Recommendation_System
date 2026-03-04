"""Training callbacks for monitoring and checkpointing.

Implements EarlyStopping, ModelCheckpoint, and GradientMonitor
callbacks that plug into the training loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping based on validation metric.

    Stops training when the monitored metric has not improved
    for ``patience`` consecutive epochs.

    Args:
        patience: Number of epochs to wait for improvement.
        metric_name: Name of the metric to monitor.
        mode: ``'max'`` (higher is better) or ``'min'``.
    """

    def __init__(
        self,
        patience: int = 3,
        metric_name: str = "recall_at_50",
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode

        self.best_value: float | None = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metrics: dict[str, float]) -> bool:
        """Check if training should stop.

        Args:
            metrics: Current epoch validation metrics.

        Returns:
            ``True`` if training should stop.
        """
        value = metrics.get(self.metric_name)
        if value is None:
            logger.warning(
                "Metric '%s' not found in metrics dict.",
                self.metric_name,
            )
            return False

        if self.best_value is None:
            self.best_value = value
            self.counter = 0
            return False

        improved = (
            (self.mode == "max" and value > self.best_value)
            or (self.mode == "min" and value < self.best_value)
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                "EarlyStopping: no improvement for %d/%d "
                "epochs (best %s=%.4f).",
                self.counter,
                self.patience,
                self.metric_name,
                self.best_value,
            )

        self.should_stop = self.counter >= self.patience
        return self.should_stop


class ModelCheckpoint:
    """Save model checkpoints based on validation metric.

    Saves the model state, optimizer state, epoch, and best
    metric value to disk whenever the metric improves.

    Args:
        checkpoint_dir: Directory for saving checkpoints.
        metric_name: Metric to monitor.
        mode: ``'max'`` or ``'min'``.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/",
        metric_name: str = "recall_at_50",
        mode: str = "max",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metric_name = metric_name
        self.mode = mode

        self.best_value: float | None = None

    def __call__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict[str, float],
    ) -> bool:
        """Potentially save a checkpoint.

        Args:
            model: Current model.
            optimizer: Current optimizer.
            epoch: Current epoch number.
            metrics: Validation metrics dict.

        Returns:
            ``True`` if a checkpoint was saved.
        """
        value = metrics.get(self.metric_name)
        if value is None:
            return False

        improved = self.best_value is None or (
            (self.mode == "max" and value > self.best_value)
            or (self.mode == "min" and value < self.best_value)
        )

        if improved:
            self.best_value = value
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": {
                    self.metric_name: value,
                },
                "metrics": metrics,
            }
            path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, path)
            logger.info(
                "Saved checkpoint at epoch %d "
                "(%s=%.4f) -> %s.",
                epoch,
                self.metric_name,
                value,
                path,
            )
            return True

        return False


class GradientMonitor:
    """Monitor gradient norms during training.

    Tracks the L2 norm of gradients across all model parameters
    and logs statistics for debugging.

    Args:
        log_interval: Log gradient stats every N steps.
    """

    def __init__(self, log_interval: int = 100) -> None:
        self.log_interval = log_interval
        self._step_count = 0
        self._grad_norms: list[float] = []

    def __call__(self, model: nn.Module) -> dict[str, float]:
        """Compute and log gradient norms.

        Should be called after ``loss.backward()`` and before
        ``optimizer.step()``.

        Args:
            model: Model with computed gradients.

        Returns:
            Dictionary with ``grad_norm`` and
            ``grad_norm_max``.
        """
        total_norm = 0.0
        max_norm = 0.0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)

        total_norm = total_norm ** 0.5
        self._grad_norms.append(total_norm)
        self._step_count += 1

        if self._step_count % self.log_interval == 0:
            recent = self._grad_norms[-self.log_interval:]
            avg = sum(recent) / len(recent)
            logger.debug(
                "Gradient norms (step %d): avg=%.4f, "
                "current=%.4f, max_param=%.4f.",
                self._step_count,
                avg,
                total_norm,
                max_norm,
            )

        return {
            "grad_norm": total_norm,
            "grad_norm_max": max_norm,
        }

    def reset(self) -> None:
        """Reset internal state for a new epoch."""
        self._grad_norms.clear()
        self._step_count = 0
