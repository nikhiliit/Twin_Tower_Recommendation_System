"""MLflow experiment tracking helpers.

Provides thin wrappers around the MLflow API for experiment
creation, parameter logging, metric logging, and artifact saving.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def init_mlflow(
    tracking_uri: str = "mlruns/",
    experiment_name: str = "two_tower_retrieval",
) -> str:
    """Initialize MLflow tracking.

    Sets the tracking URI and creates or retrieves the experiment.

    Args:
        tracking_uri: Path or URI for the MLflow backend store.
        experiment_name: Name of the MLflow experiment.

    Returns:
        The experiment ID as a string.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(
            "Created MLflow experiment '%s' (id=%s).",
            experiment_name,
            experiment_id,
        )
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            "Using existing MLflow experiment '%s' (id=%s).",
            experiment_name,
            experiment_id,
        )
    return experiment_id


def start_run(
    experiment_id: str,
    run_name: str | None = None,
) -> mlflow.ActiveRun:
    """Start a new MLflow run.

    Args:
        experiment_id: The experiment to log under.
        run_name: Optional human-readable run name.

    Returns:
        An ``mlflow.ActiveRun`` context manager.
    """
    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
    )


def log_params_flat(
    config: dict[str, Any],
    prefix: str = "",
) -> None:
    """Recursively log a nested config dict as flat MLflow params.

    Args:
        config: Nested dictionary of configuration values.
        prefix: Dot-separated prefix for nested keys.

    Example:
        >>> log_params_flat({"model": {"dim": 64}})
        # Logs: model.dim = 64
    """
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_params_flat(value, prefix=full_key)
        else:
            # MLflow truncates param values at 500 chars.
            mlflow.log_param(full_key, value)


def log_metrics(
    metrics: dict[str, float],
    step: int | None = None,
) -> None:
    """Log a dictionary of metrics to the active MLflow run.

    Args:
        metrics: Metric name -> value mapping.
        step: Optional global step or epoch number.
    """
    mlflow.log_metrics(metrics, step=step)


def log_artifact(path: str) -> None:
    """Log a local file as an MLflow artifact.

    Args:
        path: Absolute or relative path to the artifact file.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {artifact_path}"
        )
    mlflow.log_artifact(str(artifact_path))
    logger.info("Logged artifact: %s", artifact_path.name)


def end_run() -> None:
    """End the active MLflow run."""
    mlflow.end_run()
