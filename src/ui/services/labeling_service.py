"""Labeling service for neuron interpretation."""

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class LabelingService:
    """
    Provide human-readable labels for neurons.

    Strategy: Lazy-load from LLM on first access, cache in session.
    """

    def __init__(
        self, labels_json_path: Path, interpreter=None, config: Optional[Dict] = None,
        data_service=None
    ):
        """
        Initialize labeling service.

        Args:
            labels_json_path: Path to pre-computed labels.json
            interpreter: NeuronInterpreter instance (from notebook)
            config: Configuration dict
            data_service: DataService instance for POI retrieval (optional)
        """
        self.labels_json_path = Path(labels_json_path)
        self.interpreter = interpreter
        self.config = config or {}
        self.data_service = data_service

        # In-memory cache of labels
        self.labels_cache = {}
        self._load_cached_labels()

        logger.info(f"Labeling service ready ({len(self.labels_cache)} cached)")

    def _load_cached_labels(self):
        """Load pre-computed labels from JSON file."""
        if self.labels_json_path.exists():
            try:
                with open(self.labels_json_path, "r") as f:
                    data = json.load(f)
                    # Extract neuron_labels if structure is {metadata: ..., neuron_labels: {...}}
                    if isinstance(data, dict) and "neuron_labels" in data:
                        self.labels_cache = data["neuron_labels"]
                    else:
                        # Fallback if structure is different
                        self.labels_cache = data
                logger.info(f"✅ Loaded {len(self.labels_cache)} cached neuron labels from {self.labels_json_path.name}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to load labels.json: {e}")
        else:
            logger.warning(f"Labels file not found: {self.labels_json_path}")

    def get_label(self, neuron_idx: int) -> str:
        """
        Get label for a neuron.

        Strategy:
        1. Return cached label if available
        2. Generate via LLM if interpreter available
        3. Fallback to "Feature N"

        Args:
            neuron_idx: Index of neuron to label

        Returns:
            String label (e.g., "Italian Restaurants" or "Feature 42")
        """
        # Check cache first
        cached_key = str(neuron_idx)
        if cached_key in self.labels_cache:
            return self.labels_cache[cached_key]

        # Try LLM if available
        if self.interpreter:
            try:
                label = self._generate_label_via_llm(neuron_idx)
                self.labels_cache[cached_key] = label
                self._save_label(neuron_idx, label)
                return label
            except Exception as e:
                logger.debug(f"LLM label generation failed for {neuron_idx}: {e}")

        # Fallback
        fallback = f"Feature {neuron_idx}"
        self.labels_cache[cached_key] = fallback
        return fallback

    def _generate_label_via_llm(self, neuron_idx: int) -> str:
        """
        Generate label via LLM using NeuronInterpreter.

        This requires the interpreter to have access to:
        - Top POIs activating this neuron
        - SAE decoder weights

        Returns:
            Generated label string
        """
        if not self.interpreter:
            raise ValueError("No interpreter available")

        # Delegate to interpreter (from notebook)
        try:
            label = self.interpreter.label_neuron(neuron_idx)
            return label
        except Exception as e:
            logger.warning(f"Interpreter failed: {e}")
            raise

    def get_pois_for_neuron(self, neuron_idx: int, top_k: int = 10) -> List[Dict]:
        """
        Get POIs that maximally activate this neuron.

        Note: This is a placeholder that returns empty list. 
        In a future implementation, this could be connected to activation data
        computed during training or stored in a separate index.

        Args:
            neuron_idx: Neuron index
            top_k: Number of top POIs to return

        Returns:
            List of POI dicts with name, category, activation
            Currently returns empty list (feature for future enhancement)
        """
        # Placeholder implementation - returns empty list
        # In a full implementation, this would:
        # 1. Look up pre-computed activations for this neuron
        # 2. Query data_service for POI details
        # 3. Return ranked POI list
        logger.debug(f"POI retrieval for neuron {neuron_idx} not yet implemented (placeholder)")
        return []

    def _save_label(self, neuron_idx: int, label: str):
        """Persist label to JSON file."""
        try:
            with open(self.labels_json_path, "w") as f:
                json.dump(self.labels_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save label: {e}")

    def precompute_all_labels(self, num_neurons: int):
        """
        Pre-compute labels for all neurons (batch operation).

        Useful for warming up cache before interactive session.
        """
        if not self.interpreter:
            logger.warning("No interpreter available for batch labeling")
            return

        logger.info(f"Pre-computing labels for {num_neurons} neurons...")

        for neuron_idx in range(num_neurons):
            if str(neuron_idx) not in self.labels_cache:
                try:
                    label = self.get_label(neuron_idx)
                    logger.debug(f"Labeled neuron {neuron_idx}: {label}")
                except Exception as e:
                    logger.debug(f"Failed to label neuron {neuron_idx}: {e}")

        logger.info("✅ Pre-computation complete")
