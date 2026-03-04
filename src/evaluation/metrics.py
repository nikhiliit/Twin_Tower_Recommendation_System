"""Evaluation metrics for retrieval quality assessment.

All metrics are implemented with NumPy only (no PyTorch dependency)
for fast offline evaluation. Supports recall@K, MRR, NDCG@K, and
cohort-stratified evaluation.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def recall_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Recall@K averaged over all users.

    For each user, checks whether the held-out positive item
    appears in the top-K retrieved candidates.

    Args:
        retrieved: Retrieved item indices ``[n_users, max_k]``.
        ground_truth: True positive item index ``[n_users]``.
        k: Number of top candidates to consider.

    Returns:
        Mean Recall@K across all users as a float in ``[0, 1]``.

    Raises:
        ValueError: If ``k`` exceeds retrieved list length.
    """
    if k > retrieved.shape[1]:
        raise ValueError(
            f"k={k} exceeds retrieved list length "
            f"{retrieved.shape[1]}."
        )

    top_k = retrieved[:, :k]  # [n_users, k]
    gt_expanded = ground_truth.reshape(-1, 1)  # [n_users, 1]
    hits = np.any(top_k == gt_expanded, axis=1)  # [n_users]
    return float(hits.mean())


def mean_reciprocal_rank(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """Mean Reciprocal Rank (MRR).

    For each user, computes ``1 / rank`` of the positive item in
    the retrieved list. If the positive item is not found, the
    reciprocal rank is 0.

    Args:
        retrieved: Retrieved item indices ``[n_users, max_k]``.
        ground_truth: True positive item index ``[n_users]``.

    Returns:
        Mean reciprocal rank as a float.
    """
    n_users = len(ground_truth)
    rr_sum = 0.0

    for i in range(n_users):
        matches = np.where(retrieved[i] == ground_truth[i])[0]
        if len(matches) > 0:
            rank = matches[0] + 1  # 1-indexed
            rr_sum += 1.0 / rank

    return rr_sum / max(n_users, 1)


def ndcg_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    For single positive per user::

        DCG = 1 / log2(rank + 1) if found in top-K, else 0
        IDCG = 1 / log2(2) = 1.0 (best possible rank = 1)
        NDCG = DCG / IDCG

    Args:
        retrieved: Retrieved item indices ``[n_users, max_k]``.
        ground_truth: True positive item index ``[n_users]``.
        k: Number of top candidates to consider.

    Returns:
        Mean NDCG@K across all users.
    """
    if k > retrieved.shape[1]:
        raise ValueError(
            f"k={k} exceeds retrieved list length "
            f"{retrieved.shape[1]}."
        )

    n_users = len(ground_truth)
    idcg = 1.0  # Best case: item at rank 1 -> 1/log2(2) = 1
    ndcg_sum = 0.0

    top_k = retrieved[:, :k]

    for i in range(n_users):
        matches = np.where(top_k[i] == ground_truth[i])[0]
        if len(matches) > 0:
            rank = matches[0] + 1  # 1-indexed
            dcg = 1.0 / np.log2(rank + 1)
            ndcg_sum += dcg / idcg

    return ndcg_sum / max(n_users, 1)


def compute_all_metrics(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    """Compute full metrics dictionary.

    Computes recall@K and NDCG@K for each K, plus MRR.

    Args:
        retrieved: Retrieved item indices ``[n_users, max_k]``.
        ground_truth: True positive item index ``[n_users]``.
        k_values: List of K values for recall and NDCG.

    Returns:
        Dictionary mapping metric names to values, e.g.::

            {
                'recall_at_10': 0.15,
                'recall_at_50': 0.28,
                'mrr': 0.07,
                'ndcg_at_10': 0.10,
                'ndcg_at_50': 0.18,
            }
    """
    metrics: dict[str, float] = {}

    for k in k_values:
        if k <= retrieved.shape[1]:
            metrics[f"recall_at_{k}"] = recall_at_k(
                retrieved, ground_truth, k
            )
            metrics[f"ndcg_at_{k}"] = ndcg_at_k(
                retrieved, ground_truth, k
            )

    metrics["mrr"] = mean_reciprocal_rank(
        retrieved, ground_truth
    )

    return metrics


def compute_cohort_metrics(
    retrieved: np.ndarray,
    ground_truth: np.ndarray,
    user_activity: np.ndarray,
    thresholds: list[int],
    k_values: list[int],
) -> dict[str, dict[str, float]]:
    """Compute metrics stratified by user activity cohort.

    Cohorts are defined by activity thresholds:
        - cold: ``activity < thresholds[0]``
        - medium: ``thresholds[0] <= activity <= thresholds[1]``
        - power: ``activity > thresholds[1]``

    Args:
        retrieved: Retrieved item indices ``[n_users, max_k]``.
        ground_truth: True positive item index ``[n_users]``.
        user_activity: Number of training interactions per user
            ``[n_users]``.
        thresholds: List of two integers defining cohort
            boundaries, e.g. ``[5, 20]``.
        k_values: List of K values for recall and NDCG.

    Returns:
        Nested dictionary::

            {
                'cold': {'recall_at_10': 0.05, ...},
                'medium': {'recall_at_10': 0.12, ...},
                'power': {'recall_at_10': 0.25, ...},
            }
    """
    if len(thresholds) != 2:
        raise ValueError(
            f"Expected 2 thresholds, got {len(thresholds)}."
        )

    low, high = thresholds

    # Define cohort masks.
    cold_mask = user_activity < low
    medium_mask = (user_activity >= low) & (
        user_activity <= high
    )
    power_mask = user_activity > high

    cohorts = {
        "cold": cold_mask,
        "medium": medium_mask,
        "power": power_mask,
    }

    result: dict[str, dict[str, float]] = {}

    for cohort_name, mask in cohorts.items():
        n_users = mask.sum()
        if n_users == 0:
            logger.warning(
                "Cohort '%s' has 0 users, skipping.",
                cohort_name,
            )
            result[cohort_name] = {}
            continue

        cohort_retrieved = retrieved[mask]
        cohort_gt = ground_truth[mask]

        result[cohort_name] = compute_all_metrics(
            cohort_retrieved, cohort_gt, k_values
        )
        result[cohort_name]["n_users"] = float(n_users)

        logger.info(
            "Cohort '%s': %d users, recall@%d=%.4f.",
            cohort_name,
            n_users,
            k_values[0] if k_values else 0,
            result[cohort_name].get(
                f"recall_at_{k_values[0]}", 0.0
            )
            if k_values
            else 0.0,
        )

    return result
