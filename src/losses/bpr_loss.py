"""Bayesian Personalized Ranking (BPR) pairwise loss.

Implements the BPR objective for implicit feedback recommendation,
optimizing the pairwise ranking of positive over negative items.

Reference:
    Rendle et al., "BPR: Bayesian Personalized Ranking from
    Implicit Feedback", UAI 2009.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking loss.

    For each ``(user, positive_item, negative_item)`` triple,
    optimizes::

        L = -log(sigmoid(s_pos - s_neg))

    where ``s = dot(user_emb, item_emb)``.

    Args:
        reduction: Reduction mode — ``'mean'`` or ``'sum'`` over
            the batch.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(
                f"Invalid reduction '{reduction}'. "
                f"Expected 'mean' or 'sum'."
            )
        self.reduction = reduction

    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        neg_item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute BPR loss.

        Args:
            user_emb: User embeddings ``[B, D]``.
            pos_item_emb: Positive item embeddings ``[B, D]``.
            neg_item_emb: Negative item embeddings ``[B, D]``.

        Returns:
            Scalar BPR loss.
        """
        # Positive scores: [B].
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)

        # Negative scores: [B].
        neg_scores = (user_emb * neg_item_emb).sum(dim=-1)

        # BPR loss: -log(sigmoid(s_pos - s_neg)).
        loss = -F.logsigmoid(pos_scores - neg_scores)

        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()
