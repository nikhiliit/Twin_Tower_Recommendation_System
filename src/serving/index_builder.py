"""FAISS index construction and management.

Supports Flat, IVFFlat, and IVFPQ index types for approximate
nearest neighbor retrieval of item embeddings.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSIndexBuilder:
    """Builds and manages FAISS index for ANN retrieval.

    Supports three index types:
        - **Flat**: Exact search. O(N) per query. For eval ground
          truth.
        - **IVFFlat**: Approximate. Inverted file index.
          Configurable ``nlist``/``nprobe``.
        - **IVFPQ**: Approximate + compressed. Best for
          large-scale serving.

    Args:
        index_type: One of ``'Flat'``, ``'IVFFlat'``, ``'IVFPQ'``.
        embedding_dim: Dimensionality of item embeddings.
        nlist: Number of Voronoi cells (IVF variants only).
        nprobe: Number of cells to search at query time.
        metric: ``'inner_product'`` or ``'l2'``.
    """

    def __init__(
        self,
        index_type: str = "IVFFlat",
        embedding_dim: int = 64,
        nlist: int = 100,
        nprobe: int = 10,
        metric: str = "inner_product",
    ) -> None:
        self.index_type = index_type
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.nprobe = nprobe
        self.metric = metric

        self._index: faiss.Index | None = None
        self._id_map: np.ndarray | None = None

        # FAISS metric type.
        if metric == "inner_product":
            self._faiss_metric = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            self._faiss_metric = faiss.METRIC_L2
        else:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Use 'inner_product' or 'l2'."
            )

    def build(
        self,
        item_embeddings: np.ndarray,
        item_ids: np.ndarray,
    ) -> None:
        """Train and populate FAISS index.

        Args:
            item_embeddings: L2-normalized item embeddings
                ``[N, D]``.
            item_ids: Original item IDs for mapping ``[N]``.

        Raises:
            ValueError: If embeddings shape doesn't match
                ``embedding_dim``.
        """
        N, D = item_embeddings.shape
        if D != self.embedding_dim:
            raise ValueError(
                f"Expected embedding_dim={self.embedding_dim}, "
                f"got {D}."
            )

        self._id_map = item_ids.copy()
        embeddings = np.ascontiguousarray(
            item_embeddings, dtype=np.float32
        )

        start_time = time.time()

        if self.index_type == "Flat":
            if self.metric == "inner_product":
                self._index = faiss.IndexFlatIP(D)
            else:
                self._index = faiss.IndexFlatL2(D)
            self._index.add(embeddings)

        elif self.index_type == "IVFFlat":
            if self.metric == "inner_product":
                quantizer = faiss.IndexFlatIP(D)
            else:
                quantizer = faiss.IndexFlatL2(D)
            nlist = min(self.nlist, N)
            self._index = faiss.IndexIVFFlat(
                quantizer, D, nlist, self._faiss_metric
            )
            self._index.train(embeddings)
            self._index.add(embeddings)
            self._index.nprobe = self.nprobe

        elif self.index_type == "IVFPQ":
            if self.metric == "inner_product":
                quantizer = faiss.IndexFlatIP(D)
            else:
                quantizer = faiss.IndexFlatL2(D)
            nlist = min(self.nlist, N)
            # PQ params: m sub-quantizers, 8 bits each.
            m = min(8, D)
            # m must divide D.
            while D % m != 0 and m > 1:
                m -= 1
            self._index = faiss.IndexIVFPQ(
                quantizer, D, nlist, m, 8
            )
            self._index.train(embeddings)
            self._index.add(embeddings)
            self._index.nprobe = self.nprobe

        else:
            raise ValueError(
                f"Unknown index_type: {self.index_type}. "
                f"Use 'Flat', 'IVFFlat', or 'IVFPQ'."
            )

        elapsed = time.time() - start_time
        logger.info(
            "Built %s index with %d vectors in %.2fs.",
            self.index_type,
            N,
            elapsed,
        )

    def search(
        self,
        query_embeddings: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the index for nearest neighbors.

        Args:
            query_embeddings: Query vectors ``[Q, D]``.
            k: Number of nearest neighbors to retrieve.

        Returns:
            Tuple of ``(scores [Q, k], item_ids [Q, k])``.

        Raises:
            RuntimeError: If index has not been built.
        """
        if self._index is None:
            raise RuntimeError(
                "Index not built. Call build() first."
            )

        queries = np.ascontiguousarray(
            query_embeddings, dtype=np.float32
        )
        scores, indices = self._index.search(queries, k)

        # Map internal indices back to original item IDs.
        if self._id_map is not None:
            # Handle -1 (not found) indices.
            valid_mask = indices >= 0
            mapped_ids = np.full_like(indices, -1)
            mapped_ids[valid_mask] = self._id_map[
                indices[valid_mask]
            ]
        else:
            mapped_ids = indices

        return scores, mapped_ids

    def save(self, path: str) -> None:
        """Serialize index and ID mapping to disk.

        Args:
            path: Base path for saving (without extension). Two
                files will be created: ``{path}.index`` and
                ``{path}.ids.npy``.

        Raises:
            RuntimeError: If index has not been built.
        """
        if self._index is None:
            raise RuntimeError(
                "Index not built. Call build() first."
            )

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self._index, str(save_path.with_suffix(".index"))
        )
        np.save(
            str(save_path.with_suffix(".ids.npy")),
            self._id_map,
        )

        logger.info("Saved FAISS index to %s.", save_path)

    def load(self, path: str) -> None:
        """Load serialized index from disk.

        Args:
            path: Base path (without extension) used during save.

        Raises:
            FileNotFoundError: If index files don't exist.
        """
        load_path = Path(path)
        index_file = load_path.with_suffix(".index")
        ids_file = load_path.with_suffix(".ids.npy")

        if not index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {index_file}"
            )
        if not ids_file.exists():
            raise FileNotFoundError(
                f"ID mapping file not found: {ids_file}"
            )

        self._index = faiss.read_index(str(index_file))
        self._id_map = np.load(str(ids_file))

        logger.info(
            "Loaded FAISS index from %s (%d vectors).",
            load_path,
            self._index.ntotal,
        )

    def benchmark_latency(
        self,
        n_queries: int = 1000,
        k: int = 50,
    ) -> dict[str, float]:
        """Measure query latency percentiles.

        Generates random query vectors and measures search latency.

        Args:
            n_queries: Number of random queries to benchmark.
            k: Number of nearest neighbors per query.

        Returns:
            Dictionary with ``p50_ms``, ``p95_ms``, ``p99_ms``.

        Raises:
            RuntimeError: If index has not been built.
        """
        if self._index is None:
            raise RuntimeError(
                "Index not built. Call build() first."
            )

        rng = np.random.default_rng(42)
        queries = rng.standard_normal(
            (n_queries, self.embedding_dim)
        ).astype(np.float32)

        # L2 normalize queries.
        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        queries = queries / norms

        latencies: list[float] = []
        for i in range(n_queries):
            q = queries[i: i + 1]
            start = time.perf_counter()
            self._index.search(q, k)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        latencies_arr = np.array(latencies)
        result = {
            "p50_ms": float(np.percentile(latencies_arr, 50)),
            "p95_ms": float(np.percentile(latencies_arr, 95)),
            "p99_ms": float(np.percentile(latencies_arr, 99)),
        }

        logger.info(
            "Latency benchmark: p50=%.2fms, p95=%.2fms, "
            "p99=%.2fms.",
            result["p50_ms"],
            result["p95_ms"],
            result["p99_ms"],
        )
        return result
