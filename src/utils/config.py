"""Configuration management for the training pipeline.

Loads hyperparameters and training settings from YAML files.
Settings can be overridden by command-line arguments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Config:
    """Configuration container for training and evaluation.

    Provides dict-like access with dot notation fallback.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        """Initialize config from a dictionary.

        Parameters
        ----------
        data:
            Configuration dictionary (typically loaded from YAML).
        """
        self._data = data

    def __getitem__(self, key: str) -> Any:
        """Access config values with bracket notation."""
        return self._data[key]

    def __getattr__(self, key: str) -> Any:
        """Access config values with dot notation."""
        if key.startswith("_"):
            return super().__getattribute__(key)
        return self._data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with optional default."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return raw config dictionary."""
        return self._data

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(config_path: str | Path) -> Config:
    """Load configuration from a YAML file.

    Parameters
    ----------
    config_path:
        Path to the YAML config file.

    Returns
    -------
    Config
        Configuration object.

    Raises
    ------
    FileNotFoundError
        If config file does not exist.
    yaml.YAMLError
        If YAML parsing fails.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    logger.info(f"Loading config from {config_path}")

    with config_path.open("r") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary, got {type(data).__name__}")

    logger.info(f"Config loaded with keys: {list(data.keys())}")
    return Config(data)
