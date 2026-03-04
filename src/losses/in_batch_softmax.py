"""In-batch softmax (noise contrastive estimation) loss.

Uses all other items within a batch as negatives for each user,
with optional log-frequency correction to debias popular items.

Reference:
    Yi et al., "Sampling-Bias-Corrected Neural Modeling for
    Large Corpus Item Recommendations", RecSys 2019.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class InBatchSoftmaxLoss(nn.Module):
    """In-batch softmax loss with optional frequency correction.

    Within a batch of B ``(user, item)`` pairs, treats all ``B-1``
    other items as negatives for each user. The similarity matrix
    ``S[i][j] = u_i . v_j``.

    Loss::

        L = (1/B) * sum_i [
            -S[i,i]/tau + log(sum_j exp(S[i,j]/tau))
        ]

    With optional log-frequency correction::

        S_corrected[i,j] = S[i,j] - alpha * log(freq[j])

    Args:
        temperature: Softmax temperature tau. Default ``0.07``.
        use_frequency_correction: Apply log-frequency debiasing.
        correction_alpha: Strength of frequency correction.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_frequency_correction: bool = False,
        correction_alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_frequency_correction = use_frequency_correction
        self.correction_alpha = correction_alpha

    def forward(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        item_frequencies: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute in-batch softmax loss.

        Args:
            user_emb: User embeddings ``[B, D]``.
            item_emb: Item embeddings ``[B, D]``.
            item_frequencies: Log-frequencies ``[B]`` for each
                item in the batch. Required when
                ``use_frequency_correction=True``.

        Returns:
            Tuple of ``(loss, metrics_dict)`` where metrics_dict
            contains ``avg_pos_sim``, ``avg_neg_sim``, and
            ``accuracy`` (fraction where positive is top-scored).

        Raises:
            ValueError: If frequency correction is enabled but
                ``item_frequencies`` is ``None``.
        """
        B = user_emb.shape[0]

        # Similarity matrix: [B, B].
        logits = torch.matmul(user_emb, item_emb.t())

        # Apply frequency correction.
        if self.use_frequency_correction:
            if item_frequencies is None:
                raise ValueError(
                    "item_frequencies required when "
                    "use_frequency_correction=True."
                )
            # Correction: subtract alpha * log(freq) for each
            # item column.
            correction = (
                self.correction_alpha * item_frequencies
            )
            logits = logits - correction.unsqueeze(0)

        # Scale by temperature.
        logits = logits / self.temperature

        # Labels: diagonal elements are positives.
        labels = torch.arange(B, device=user_emb.device)

        # Cross-entropy loss.
        loss = F.cross_entropy(logits, labels)

        # Metrics.
        with torch.no_grad():
            # Extract diagonal (positive) similarities.
            pos_sim = torch.diagonal(logits).mean().item()
            # Mean of all off-diagonal (negative) similarities.
            mask = ~torch.eye(B, dtype=torch.bool,
                              device=user_emb.device)
            neg_sim = logits[mask].mean().item()
            # Top-1 accuracy.
            preds = logits.argmax(dim=1)
            accuracy = (
                (preds == labels).float().mean().item()
            )

        metrics = {
            "avg_pos_sim": pos_sim * self.temperature,
            "avg_neg_sim": neg_sim * self.temperature,
            "in_batch_accuracy": accuracy,
        }

        return loss, metrics
