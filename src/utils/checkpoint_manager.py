"""Checkpoint management for model save/load operations."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Handle saving and loading of model checkpoints with metadata.

    Saves model state dict, optimizer state, epoch info, and metrics.
    """

    def __init__(self, checkpoint_dir: str | Path) -> None:
        """Initialize checkpoint manager.

        Parameters
        ----------
        checkpoint_dir:
            Directory to store checkpoints.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int | None = None,
        metrics: dict[str, float] | None = None,
        name: str = "checkpoint",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save a model checkpoint.

        Parameters
        ----------
        model:
            The model to save.
        optimizer:
            The optimizer state to save (optional).
        epoch:
            Current epoch number (optional).
        metrics:
            Dictionary of metrics to save (optional).
        name:
            Checkpoint name (default: "checkpoint").
        metadata:
            Dictionary of dataset/model metadata to save (e.g., n_items, latent_dim).

        Returns
        -------
        Path
            Path to the saved checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint_dict = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics or {},
            "metadata": metadata or {},
        }

        if optimizer is not None:
            checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint_dict, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        if metadata:
            logger.info(f"  Metadata: {metadata}")

        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        checkpoint_name: str = "checkpoint",
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Load a model checkpoint.

        Parameters
        ----------
        model:
            The model to load state into.
        optimizer:
            The optimizer to load state into (optional).
        checkpoint_name:
            Checkpoint name to load.
        device:
            Device to load to (default: "cpu").

        Returns
        -------
        dict
            Checkpoint dictionary with epoch and metrics.

        Raises
        ------
        FileNotFoundError
            If checkpoint does not exist.
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")

        return {
            "epoch": checkpoint.get("epoch"),
            "metrics": checkpoint.get("metrics", {}),
        }

    def save_metrics(
        self,
        metrics: dict[str, Any],
        split: str = "train",
    ) -> Path:
        """Save metrics to a JSON file.

        Parameters
        ----------
        metrics:
            Metrics dictionary to save.
        split:
            Data split name (e.g., "train", "val", "test").

        Returns
        -------
        Path
            Path to the saved metrics file.
        """
        metrics_path = self.checkpoint_dir / f"metrics_{split}.json"

        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {metrics_path}")
        return metrics_path

    def load_metrics(self, split: str = "train") -> dict[str, Any]:
        """Load metrics from a JSON file.

        Parameters
        ----------
        split:
            Data split name (e.g., "train", "val", "test").

        Returns
        -------
        dict
            Loaded metrics dictionary.

        Raises
        ------
        FileNotFoundError
            If metrics file does not exist.
        """
        metrics_path = self.checkpoint_dir / f"metrics_{split}.json"

        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

        with metrics_path.open("r") as f:
            return json.load(f)
