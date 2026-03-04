"""Deterministic hash-based A/B traffic router.

Assigns users to model variants (A or B) using a stable hash of
their user ID, ensuring the same user always gets the same variant
across requests for consistent experimentation.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import defaultdict
from typing import Literal

logger = logging.getLogger(__name__)

Variant = Literal["A", "B"]


class ABRouter:
    """Deterministic A/B traffic router.

    Uses MD5 hashing of the user ID to assign users to variant A
    or B. The assignment is stable — the same user will always be
    routed to the same variant regardless of server restart.

    Args:
        traffic_split: Fraction of users routed to variant A.
            Must be in ``[0.0, 1.0]``. Default is ``0.5`` (50/50).
        variant_a_name: Human-readable name for variant A.
        variant_b_name: Human-readable name for variant B.
    """

    def __init__(
        self,
        traffic_split: float = 0.5,
        variant_a_name: str = "A",
        variant_b_name: str = "B",
    ) -> None:
        if not 0.0 <= traffic_split <= 1.0:
            raise ValueError(
                f"traffic_split must be in [0, 1], "
                f"got {traffic_split}."
            )
        self.traffic_split = traffic_split
        self.variant_a_name = variant_a_name
        self.variant_b_name = variant_b_name

        # Thread-safe counters.
        self._lock = threading.Lock()
        self._counters: dict[str, dict[str, int | float]] = {
            "A": defaultdict(int),
            "B": defaultdict(int),
        }

        logger.info(
            "ABRouter initialised: %.0f%% → %s, %.0f%% → %s.",
            traffic_split * 100,
            variant_a_name,
            (1 - traffic_split) * 100,
            variant_b_name,
        )

    def assign(self, user_id: int) -> Variant:
        """Assign a user to a variant via stable hashing.

        Args:
            user_id: External user ID (any integer).

        Returns:
            ``"A"`` or ``"B"``.
        """
        digest = hashlib.md5(
            str(user_id).encode(), usedforsecurity=False
        ).hexdigest()
        # Use first 8 hex chars → 32-bit bucket in [0, 2^32).
        bucket = int(digest[:8], 16) / (2**32)
        return "A" if bucket < self.traffic_split else "B"

    def record(
        self,
        variant: Variant,
        latency_ms: float,
        cold_start: bool,
    ) -> None:
        """Record a served request for metrics tracking.

        Args:
            variant: The variant that served the request.
            latency_ms: End-to-end request latency in milliseconds.
            cold_start: Whether this was a cold-start fallback.
        """
        with self._lock:
            c = self._counters[variant]
            c["requests"] += 1
            c["cold_starts"] += int(cold_start)
            c["total_latency_ms"] += latency_ms

    def get_stats(self) -> dict[str, dict[str, float]]:
        """Return aggregated per-variant statistics.

        Returns:
            Dict with keys ``"A"`` and ``"B"``, each containing
            ``requests``, ``cold_starts``, ``cold_start_rate``,
            and ``avg_latency_ms``.
        """
        with self._lock:
            stats: dict[str, dict[str, float]] = {}
            for variant, c in self._counters.items():
                n = c["requests"] or 1  # avoid div-by-zero
                stats[variant] = {
                    "requests": c["requests"],
                    "cold_starts": c["cold_starts"],
                    "cold_start_rate": c["cold_starts"] / n,
                    "avg_latency_ms": c["total_latency_ms"] / n,
                }
            return stats
