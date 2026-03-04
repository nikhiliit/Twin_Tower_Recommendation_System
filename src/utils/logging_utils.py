"""Structured logging setup for the training pipeline.

Configures Python's built-in logging with a consistent format,
optional file output, and configurable verbosity levels.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
) -> None:
    """Configure the root logger with structured formatting.

    Args:
        level: Logging verbosity level (e.g. ``logging.INFO``).
        log_file: Optional path to a log file. If provided, logs are
            written to both stdout and the specified file.

    Example:
        >>> setup_logging(level=logging.DEBUG, log_file="train.log")
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates.
    root_logger.handlers.clear()

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler.
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional).
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Convenience wrapper around ``logging.getLogger`` to ensure
    consistent usage across the codebase.

    Args:
        name: Logger name, typically ``__name__``.

    Returns:
        Configured ``logging.Logger`` instance.
    """
    return logging.getLogger(name)
