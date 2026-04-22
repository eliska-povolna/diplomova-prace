"""Labeling service for neuron interpretation."""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LabelingService:
    """Provide human-readable labels for neurons."""

    def __init__(
        self,
        labels_json_path: Path,
        interpreter=None,
        config: Optional[Dict] = None,
        data_service=None,
    ):
        self.labels_source_path = Path(labels_json_path)
        self.interpreter = interpreter
        self.config = config or {}
        self.data_service = data_service

        self.labels_by_method: Dict[str, Dict[str, str]] = {}
        self.labels_cache: Dict[str, str] = {}
        self.selected_method: str = "default"

        self._load_cached_labels()
        logger.info(
            "Labeling service ready (%s cached, methods=%s)",
            len(self.labels_cache),
            self.available_methods,
        )

    @property
    def available_methods(self) -> List[str]:
        return sorted(self.labels_by_method.keys())

    def _load_cached_labels(self) -> None:
        source_path = self.labels_source_path

        if source_path.is_dir():
            self._load_from_directory(source_path)
        elif source_path.exists():
            self._load_from_file(source_path)
        else:
            logger.warning("Labels file not found: %s", source_path)

        if not self.labels_by_method:
            self.labels_by_method = {"default": {}}

        self._select_default_method()

    def _load_from_directory(self, labels_dir: Path) -> None:
        loaded = {}

        for label_file in sorted(labels_dir.glob("labels_*.pkl")):
            method_name = label_file.stem[len("labels_") :]
            try:
                with open(label_file, "rb") as f:
                    data = pickle.load(f)
                loaded[method_name] = self._normalize_label_dict(data)
            except Exception as e:
                logger.warning("Failed to load %s: %s", label_file, e)

        if not loaded:
            for fallback_name in ("neuron_labels.json", "labels.json"):
                fallback_path = labels_dir / fallback_name
                if fallback_path.exists():
                    self._load_from_file(fallback_path, method_name="default")
                    return

        self.labels_by_method.update(loaded)

    def _load_from_file(self, labels_file: Path, method_name: str = "default") -> None:
        try:
            if labels_file.suffix.lower() == ".pkl":
                with open(labels_file, "rb") as f:
                    data = pickle.load(f)
            else:
                with open(labels_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            if isinstance(data, dict) and "neuron_labels" in data:
                data = data["neuron_labels"]

            self.labels_by_method[method_name] = self._normalize_label_dict(data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to load labels JSON from %s: %s", labels_file, e)
        except Exception as e:
            logger.warning("Failed to load labels from %s: %s", labels_file, e)

    @staticmethod
    def _normalize_label_dict(data) -> Dict[str, str]:
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _select_default_method(self) -> None:
        preferred_order = ["tag-based", "matrix-based", "llm-based", "default"]
        for method_name in preferred_order:
            if method_name in self.labels_by_method:
                self.selected_method = method_name
                self.labels_cache = dict(self.labels_by_method[method_name])
                return

        self.selected_method = next(iter(self.labels_by_method.keys()))
        self.labels_cache = dict(self.labels_by_method[self.selected_method])

    def set_method(self, method_name: str) -> None:
        if method_name not in self.labels_by_method:
            logger.warning(
                "Label method '%s' not available; keeping '%s'",
                method_name,
                self.selected_method,
            )
            return

        self.selected_method = method_name
        self.labels_cache = dict(self.labels_by_method[method_name])

    def _persist_path(self) -> Path:
        if self.labels_source_path.is_dir():
            return self.labels_source_path / "neuron_labels.json"
        return self.labels_source_path

    def get_label(self, neuron_idx: int) -> str:
        cached_key = str(neuron_idx)
        if cached_key in self.labels_cache:
            return self.labels_cache[cached_key]

        if self.interpreter:
            try:
                label = self._generate_label_via_llm(neuron_idx)
                self.labels_cache[cached_key] = label
                self._save_label(label)
                return label
            except Exception as e:
                logger.debug("LLM label generation failed for %s: %s", neuron_idx, e)

        fallback = f"Feature {neuron_idx}"
        self.labels_cache[cached_key] = fallback
        return fallback

    def _generate_label_via_llm(self, neuron_idx: int) -> str:
        if not self.interpreter:
            raise ValueError("No interpreter available")
        return self.interpreter.label_neuron(neuron_idx)

    def get_pois_for_neuron(self, neuron_idx: int, top_k: int = 10) -> List[Dict]:
        logger.debug(
            "POI retrieval for neuron %s not yet implemented (placeholder)", neuron_idx
        )
        return []

    def _save_label(self, label: str) -> None:
        try:
            persist_path = self._persist_path()
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(persist_path, "w", encoding="utf-8") as f:
                json.dump(self.labels_cache, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save label: %s", e)

    def precompute_all_labels(self, num_neurons: int) -> None:
        if not self.interpreter:
            logger.warning("No interpreter available for batch labeling")
            return

        for neuron_idx in range(num_neurons):
            if str(neuron_idx) not in self.labels_cache:
                try:
                    self.get_label(neuron_idx)
                except Exception as e:
                    logger.debug("Failed to label neuron %s: %s", neuron_idx, e)
                try:
                    label = self.get_label(neuron_idx)
                    logger.debug(f"Labeled neuron {neuron_idx}: {label}")
                except Exception as e:
                    logger.debug(f"Failed to label neuron {neuron_idx}: {e}")

        logger.info("✅ Pre-computation complete")
