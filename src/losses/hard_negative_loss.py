"""Sampled softmax loss with hard negatives.

Combines in-batch negatives with hard negatives mined from a
FAISS index to create a more challenging training signal.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HardNegativeLoss(nn.Module):
    """Sampled softmax loss with mixed in-batch and hard negatives.

    For each ``(user, pos_item)`` pair, negatives consist of
    in-batch items union top-K hard negatives from FAISS that
    are not positives.

    The hard negative ratio controls the mix::

        n_hard = int(hard_neg_ratio * (batch_size - 1))

    Args:
        temperature: Softmax temperature (used as final temperature
            when annealing is enabled).
        hard_neg_ratio: Fraction of negatives that are hard.
        use_frequency_correction: Apply log-frequency debiasing.
        correction_alpha: Strength of frequency correction.
        initial_temperature: Starting temperature for annealing.
            If ``None``, no annealing is applied and
            ``temperature`` is used throughout. Defaults to ``None``.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        hard_neg_ratio: float = 0.3,
        use_frequency_correction: bool = False,
        correction_alpha: float = 1.0,
        initial_temperature: float | None = None,
    ) -> None:
        super().__init__()
        self.final_temperature = temperature
        self.initial_temperature = (
            initial_temperature
            if initial_temperature is not None
            else temperature
        )
        self.temperature = self.initial_temperature
        self.hard_neg_ratio = hard_neg_ratio
        self.use_frequency_correction = use_frequency_correction
        self.correction_alpha = correction_alpha

    def set_epoch(
        self, epoch: int, total_epochs: int
    ) -> None:
        """Update temperature via linear annealing.

        Linearly anneals from ``initial_temperature`` to
        ``final_temperature`` over ``total_epochs``.

        Args:
            epoch: Current epoch (1-indexed).
            total_epochs: Total number of training epochs.
        """
        if total_epochs <= 1:
            self.temperature = self.final_temperature
            return
        progress = min((epoch - 1) / (total_epochs - 1), 1.0)
        self.temperature = (
            self.initial_temperature
            + progress
            * (self.final_temperature - self.initial_temperature)
        )
        logger.info(
            "Temperature annealed to %.4f (epoch %d/%d).",
            self.temperature,
            epoch,
            total_epochs,
        )

    def forward(
        self,
        user_emb: torch.Tensor,
        pos_item_emb: torch.Tensor,
        in_batch_item_emb: torch.Tensor,
        hard_neg_emb: torch.Tensor,
        item_frequencies: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute hard negative sampled softmax loss.

        Args:
            user_emb: User embeddings ``[B, D]``.
            pos_item_emb: Positive item embeddings ``[B, D]``.
            in_batch_item_emb: All in-batch item embeddings
                ``[B, D]``.
            hard_neg_emb: Hard negative embeddings ``[B, K, D]``.
            item_frequencies: Log-frequencies ``[B + K]`` for
                frequency correction. Optional.

        Returns:
            Tuple of ``(loss, metrics_dict)``.
        """
        B = user_emb.shape[0]
        K = hard_neg_emb.shape[1] if hard_neg_emb.dim() == 3 else 0

        # Positive scores: [B].
        pos_scores = (user_emb * pos_item_emb).sum(dim=-1)

        # In-batch negative scores: [B, B].
        in_batch_logits = torch.matmul(
            user_emb, in_batch_item_emb.t()
        )

        if K > 0:
            # Hard negative scores: [B, K].
            # user_emb: [B, 1, D] x hard_neg_emb: [B, K, D]
            hard_scores = torch.bmm(
                user_emb.unsqueeze(1),
                hard_neg_emb.transpose(1, 2),
            ).squeeze(1)  # [B, K]

            # Combine all negative logits: [B, B + K].
            all_logits = torch.cat(
                [in_batch_logits, hard_scores], dim=1
            )
        else:
            all_logits = in_batch_logits

        # Apply frequency correction.
        if (
            self.use_frequency_correction
            and item_frequencies is not None
        ):
            correction = (
                self.correction_alpha * item_frequencies
            )
            # Trim or pad to match logits width.
            n_cols = all_logits.shape[1]
            if correction.shape[0] >= n_cols:
                correction = correction[:n_cols]
            else:
                pad = torch.zeros(
                    n_cols - correction.shape[0],
                    device=correction.device,
                )
                correction = torch.cat([correction, pad])
            all_logits = all_logits - correction.unsqueeze(0)

        # Scale by temperature.
        pos_scores_scaled = pos_scores / self.temperature
        all_logits_scaled = all_logits / self.temperature

        # Log-sum-exp of all candidates (positive + negatives).
        # For numerical stability, include positive in the
        # denominator as well.
        lse = torch.logsumexp(
            torch.cat(
                [
                    pos_scores_scaled.unsqueeze(1),
                    all_logits_scaled,
                ],
                dim=1,
            ),
            dim=1,
        )  # [B]

        # Loss: -pos_score + log_sum_exp.
        loss = (-pos_scores_scaled + lse).mean()

        # Metrics.
        with torch.no_grad():
            avg_pos_sim = pos_scores.mean().item()
            avg_neg_sim = all_logits.mean().item()

            # Top-1 accuracy: check if positive has highest score.
            full_scores = torch.cat(
                [
                    pos_scores_scaled.unsqueeze(1),
                    all_logits_scaled,
                ],
                dim=1,
            )
            preds = full_scores.argmax(dim=1)
            accuracy = (preds == 0).float().mean().item()

        metrics = {
            "avg_pos_sim": avg_pos_sim,
            "avg_neg_sim": avg_neg_sim,
            "hard_neg_accuracy": accuracy,
            "n_hard_negatives": K,
        }

        return loss, metrics
