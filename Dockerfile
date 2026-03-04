# ── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from the builder stage.
COPY --from=builder /install /usr/local

# Copy source code only (no data or checkpoints — mounted via volumes).
COPY src/ ./src/
COPY configs/ ./configs/
COPY setup.py .

# Install the package itself (editable-free, no data deps).
RUN pip install --no-cache-dir -e . --no-deps

# Create directories for volume mounts.
RUN mkdir -p checkpoints data/processed

# Non-root user for security.
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# Expose the API port.
EXPOSE 8000

# Healthcheck.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default environment variables (override via docker-compose or -e flags).
ENV CONFIG_PATH=configs/hardneg_config.yaml \
    PROCESSED_DIR=data/processed \
    MODEL_A_CHECKPOINT=checkpoints/best_model.pt \
    MODEL_B_CHECKPOINT=checkpoints/best_model.pt \
    FAISS_INDEX_PATH=checkpoints/faiss_index \
    AB_TRAFFIC_SPLIT=0.5 \
    DEVICE=cpu

CMD ["uvicorn", "src.serving.app:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "1", \
    "--log-level", "info"]
