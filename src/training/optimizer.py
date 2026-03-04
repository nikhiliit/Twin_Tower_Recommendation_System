"""Optimizer and learning rate scheduler factory.

Creates optimizers and schedulers from configuration dictionaries,
supporting AdamW with cosine annealing or step decay.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    StepLR,
)

logger = logging.getLogger(__name__)


def create_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
) -> AdamW:
    """Create an AdamW optimizer.

    Separates parameters into weight-decay and no-weight-decay
    groups (biases and BatchNorm parameters skip decay).

    Args:
        model: PyTorch model.
        learning_rate: Base learning rate.
        weight_decay: L2 regularization strength.

    Returns:
        Configured ``AdamW`` optimizer.
    """
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(param_groups, lr=learning_rate)

    logger.info(
        "Created AdamW optimizer: lr=%.2e, weight_decay=%.2e, "
        "decay_params=%d, no_decay_params=%d.",
        learning_rate,
        weight_decay,
        len(decay_params),
        len(no_decay_params),
    )
    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    epochs: int = 20,
    step_size: int = 5,
    gamma: float = 0.5,
) -> LRScheduler:
    """Create a learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer.
        scheduler_type: ``'cosine'`` or ``'step'``.
        epochs: Total training epochs (for cosine).
        step_size: Step size for StepLR.
        gamma: LR decay factor for StepLR.

    Returns:
        Learning rate scheduler.

    Raises:
        ValueError: If ``scheduler_type`` is unknown.
    """
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6
        )
    elif scheduler_type == "step":
        scheduler = StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    else:
        raise ValueError(
            f"Unknown scheduler_type: {scheduler_type}. "
            f"Use 'cosine' or 'step'."
        )

    logger.info(
        "Created %s scheduler (epochs=%d).",
        scheduler_type,
        epochs,
    )
    return scheduler
