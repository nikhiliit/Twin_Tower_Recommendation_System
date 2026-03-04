"""Integration test for FAISS retrieval."""

from __future__ import annotations

import numpy as np
import pytest

from src.serving.index_builder import FAISSIndexBuilder


class TestFAISSRetrieval:
    """Integration tests for FAISS index building and search."""

    def test_flat_index_exact(self) -> None:
        """Flat index should return exact nearest neighbors."""
        D = 32
        N = 100
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((N, D)).astype(
            np.float32
        )
        # L2 normalize.
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        embeddings = embeddings / norms
        item_ids = np.arange(N)

        builder = FAISSIndexBuilder(
            index_type="Flat",
            embedding_dim=D,
            metric="inner_product",
        )
        builder.build(embeddings, item_ids)

        # Query with first item should return itself as top-1.
        query = embeddings[:1]
        scores, ids = builder.search(query, k=5)
        assert ids[0, 0] == 0
        assert abs(scores[0, 0] - 1.0) < 1e-4

    def test_ivfflat_index(self) -> None:
        """IVFFlat index should return reasonable results."""
        D = 32
        N = 200
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((N, D)).astype(
            np.float32
        )
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        embeddings = embeddings / norms
        item_ids = np.arange(N)

        builder = FAISSIndexBuilder(
            index_type="IVFFlat",
            embedding_dim=D,
            nlist=10,
            nprobe=5,
            metric="inner_product",
        )
        builder.build(embeddings, item_ids)

        query = embeddings[:5]
        scores, ids = builder.search(query, k=10)
        assert scores.shape == (5, 10)
        assert ids.shape == (5, 10)

    def test_save_load_roundtrip(self, tmp_path: str) -> None:
        """Index should be identical after save/load."""
        D = 16
        N = 50
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((N, D)).astype(
            np.float32
        )
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        embeddings = embeddings / norms
        item_ids = np.arange(N)

        builder = FAISSIndexBuilder(
            index_type="Flat",
            embedding_dim=D,
            metric="inner_product",
        )
        builder.build(embeddings, item_ids)

        # Save.
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_index")
            builder.save(path)

            # Load into new builder.
            loaded = FAISSIndexBuilder(
                index_type="Flat",
                embedding_dim=D,
                metric="inner_product",
            )
            loaded.load(path)

            # Compare search results.
            query = embeddings[:3]
            _, ids_orig = builder.search(query, k=5)
            _, ids_loaded = loaded.search(query, k=5)
            np.testing.assert_array_equal(
                ids_orig, ids_loaded
            )

    def test_benchmark_latency(self) -> None:
        """Benchmark should return valid percentile dict."""
        D = 16
        N = 100
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((N, D)).astype(
            np.float32
        )
        norms = np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )
        embeddings = embeddings / norms

        builder = FAISSIndexBuilder(
            index_type="Flat",
            embedding_dim=D,
            metric="inner_product",
        )
        builder.build(embeddings, np.arange(N))

        latency = builder.benchmark_latency(
            n_queries=10, k=5
        )
        assert "p50_ms" in latency
        assert "p95_ms" in latency
        assert "p99_ms" in latency
        assert all(v > 0 for v in latency.values())
