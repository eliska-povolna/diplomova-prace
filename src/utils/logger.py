"""Structured logging setup for training and evaluation.

Provides both console and file logging with consistent formatting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path


class JsonFormatter(logging.Formatter):
    """JSON-formatted log output for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Parameters
        ----------
        record:
            The log record to format.

        Returns
        -------
        str
            JSON-formatted log entry.
        """
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logger(
    name: str,
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger with console and optional file handlers.

    Parameters
    ----------
    name:
        Logger name (typically ``__name__``).
    log_dir:
        If provided, create a JSON log file in this directory.
    level:
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler with human-readable format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler with JSON format (if log_dir provided)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{name.replace('.', '_')}.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
        logger.info(f"Logging to {log_file}")

    return logger
