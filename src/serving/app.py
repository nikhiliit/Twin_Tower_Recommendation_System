"""FastAPI serving application for Two-Tower Retrieval.

Loads trained model(s), FAISS index, and feature store at startup,
then exposes three endpoints:

- POST /recommend  — retrieve top-K items for a user (A/B routed,
                     cold-start aware)
- GET  /health     — readiness check with model metadata
- GET  /metrics    — per-variant request statistics

Environment variables
---------------------
MODEL_A_CHECKPOINT  : path to model-A .pt checkpoint
MODEL_B_CHECKPOINT  : path to model-B .pt checkpoint (can equal A)
FAISS_INDEX_PATH    : base path for the FAISS index files
PROCESSED_DIR       : path to data/processed/ directory
AB_TRAFFIC_SPLIT    : float in [0,1], fraction routed to variant A
CONFIG_PATH         : path to YAML config (default: configs/base_config.yaml)
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.feature_store import FeatureStore
from src.models.two_tower import TwoTowerModel
from src.serving.ab_router import ABRouter
from src.serving.cold_start import ColdStartHandler
from src.serving.index_builder import FAISSIndexBuilder
from src.serving.retriever import TwoTowerRetriever
from src.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    """Request body for the /recommend endpoint."""

    user_id: int = Field(
        ..., description="External (original) user ID."
    )
    k: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of items to retrieve.",
    )
    request_id: str | None = Field(
        default=None,
        description="Optional caller-supplied request ID for tracing.",
    )


class RecommendResponse(BaseModel):
    """Response body for the /recommend endpoint."""

    user_id: int
    items: list[int] = Field(
        description="Ordered list of recommended original movie IDs."
    )
    scores: list[float] = Field(
        description="Corresponding similarity scores."
    )
    model_variant: str = Field(
        description="A/B variant that served this request."
    )
    cold_start: bool = Field(
        description="True if popularity fallback was used."
    )
    latency_ms: float = Field(
        description="End-to-end serving latency in milliseconds."
    )
    request_id: str | None = None


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    n_items: int
    n_users: int
    variant_a: str
    variant_b: str
    uptime_s: float


class MetricsResponse(BaseModel):
    """Response body for the /metrics endpoint."""

    variants: dict[str, dict[str, float]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(
    checkpoint_path: str,
    feature_store: FeatureStore,
    config: dict[str, Any],
    device: str,
) -> TwoTowerRetriever:
    """Load a model checkpoint and wrap it in a TwoTowerRetriever."""
    model_cfg = config.get("model", {})
    model = TwoTowerModel(
        n_users=feature_store.n_users,
        n_items=feature_store.n_items,
        n_genres=feature_store.n_genres,
        genome_dim=feature_store.genome_dim,
        embedding_dim=model_cfg.get("embedding_dim", 64),
        user_hidden_dims=model_cfg.get(
            "user_hidden_dims", [256, 128]
        ),
        item_hidden_dims=model_cfg.get(
            "item_hidden_dims", [192, 128]
        ),
        dropout=model_cfg.get("dropout", 0.1),
        use_genome=model_cfg.get("use_genome_features", True),
        history_length=model_cfg.get("history_length", 50),
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    # Load FAISS index.
    faiss_cfg = config.get("faiss", {})
    model_cfg = config.get("model", {})
    index_builder = FAISSIndexBuilder(
        index_type=faiss_cfg.get("index_type", "IVFFlat"),
        embedding_dim=model_cfg.get("embedding_dim", 64),
        nlist=faiss_cfg.get("nlist", 100),
        nprobe=faiss_cfg.get("nprobe", 10),
        metric=faiss_cfg.get("metric", "inner_product"),
    )
    faiss_path = os.getenv(
        "FAISS_INDEX_PATH", "checkpoints/faiss_index"
    )
    index_builder.load(faiss_path)

    return TwoTowerRetriever(
        model=model,
        feature_store=feature_store,
        index_builder=index_builder,
        device=device,
    )


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all serving artifacts at startup."""
    config_path = os.getenv(
        "CONFIG_PATH", "configs/base_config.yaml"
    )
    processed_dir = Path(
        os.getenv("PROCESSED_DIR", "data/processed")
    )
    ckpt_a = os.getenv(
        "MODEL_A_CHECKPOINT", "checkpoints/best_model.pt"
    )
    ckpt_b = os.getenv(
        "MODEL_B_CHECKPOINT", "checkpoints/best_model.pt"
    )
    traffic_split = float(
        os.getenv("AB_TRAFFIC_SPLIT", "0.5")
    )
    device = os.getenv("DEVICE", "cpu")

    logger.info("Loading config from %s.", config_path)
    with open(config_path, "r") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info("Loading feature store from %s.", processed_dir)
    with open(processed_dir / "feature_store.pkl", "rb") as f:
        feature_store: FeatureStore = pickle.load(f)

    logger.info("Loading model A from %s.", ckpt_a)
    retriever_a = _load_model(ckpt_a, feature_store, config, device)

    if ckpt_b == ckpt_a:
        logger.info(
            "MODEL_B_CHECKPOINT == MODEL_A_CHECKPOINT; "
            "sharing retriever instance."
        )
        retriever_b = retriever_a
    else:
        logger.info("Loading model B from %s.", ckpt_b)
        retriever_b = _load_model(
            ckpt_b, feature_store, config, device
        )

    cold_start_handler = ColdStartHandler(feature_store)
    ab_router = ABRouter(
        traffic_split=traffic_split,
        variant_a_name=Path(ckpt_a).stem,
        variant_b_name=Path(ckpt_b).stem,
    )

    # Store in app.state for access by route handlers.
    app.state.retriever_a = retriever_a
    app.state.retriever_b = retriever_b
    app.state.cold_start = cold_start_handler
    app.state.ab_router = ab_router
    app.state.feature_store = feature_store
    app.state.ckpt_a = ckpt_a
    app.state.ckpt_b = ckpt_b

    logger.info(
        "Serving pipeline ready. %d users, %d items.",
        feature_store.n_users,
        feature_store.n_items,
    )
    yield
    # Shutdown — nothing to clean up here.
    logger.info("Shutting down serving app.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Two-Tower Retrieval API",
    description=(
        "Movie recommendation API backed by a two-tower dual-encoder "
        "model with FAISS ANN retrieval, cold-start fallback, and "
        "A/B traffic routing."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get top-K movie recommendations for a user.",
)
async def recommend(req: RecommendRequest) -> RecommendResponse:
    """Recommend top-K movies for a given user.

    - If the ``user_id`` is unknown (cold-start), returns the
      globally most popular items as fallback.
    - Applies A/B routing deterministically based on ``user_id``.
    """
    t0 = time.perf_counter()

    cold_handler: ColdStartHandler = app.state.cold_start
    ab_router: ABRouter = app.state.ab_router
    feature_store: FeatureStore = app.state.feature_store

    is_cold = cold_handler.is_cold_start(req.user_id)

    if is_cold:
        popular = cold_handler.get_popular_items(req.k)
        scores = [0.0] * len(popular)
        variant = ab_router.assign(req.user_id)
        latency_ms = (time.perf_counter() - t0) * 1000
        ab_router.record(variant, latency_ms, cold_start=True)
        return RecommendResponse(
            user_id=req.user_id,
            items=popular,
            scores=scores,
            model_variant=variant,
            cold_start=True,
            latency_ms=round(latency_ms, 3),
            request_id=req.request_id,
        )

    # Look up internal user index.
    user_idx = feature_store.user_id_map.get(req.user_id)
    if user_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {req.user_id} not found.",
        )

    # A/B routing.
    variant = ab_router.assign(req.user_id)
    retriever: TwoTowerRetriever = (
        app.state.retriever_a
        if variant == "A"
        else app.state.retriever_b
    )

    scores_arr, item_ids_arr = retriever.retrieve(
        user_idx=user_idx, k=req.k
    )

    # Map contiguous item indices → original movie IDs.
    idx_to_movie: dict[int, int] = {
        v: k for k, v in feature_store.item_id_map.items()
    }
    movie_ids = [
        idx_to_movie.get(int(iid), int(iid))
        for iid in item_ids_arr
        if int(iid) >= 0
    ]
    scores_list = [
        float(s)
        for s, iid in zip(scores_arr, item_ids_arr)
        if int(iid) >= 0
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    ab_router.record(variant, latency_ms, cold_start=False)

    return RecommendResponse(
        user_id=req.user_id,
        items=movie_ids,
        scores=scores_list,
        model_variant=variant,
        cold_start=False,
        latency_ms=round(latency_ms, 3),
        request_id=req.request_id,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and readiness check.",
)
async def health() -> HealthResponse:
    """Return service health metadata."""
    fs: FeatureStore = app.state.feature_store
    return HealthResponse(
        status="ok",
        n_items=fs.n_items,
        n_users=fs.n_users,
        variant_a=app.state.ckpt_a,
        variant_b=app.state.ckpt_b,
        uptime_s=round(time.time() - _START_TIME, 1),
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Per-variant A/B traffic and latency metrics.",
)
async def metrics() -> MetricsResponse:
    """Return aggregated per-variant request statistics."""
    ab_router: ABRouter = app.state.ab_router
    return MetricsResponse(variants=ab_router.get_stats())
