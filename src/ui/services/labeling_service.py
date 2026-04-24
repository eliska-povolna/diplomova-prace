"""Labeling service for reading persisted neuron interpretation artifacts."""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LabelingService:
    """Provide human-readable labels for neurons."""

    def __init__(
        self,
        labels_json_path: Path,
        config: Optional[Dict] = None,
        data_service=None,
    ):
        self.labels_source_path = Path(labels_json_path)
        self.config = config or {}
        self.data_service = data_service

        self.labels_by_method: Dict[str, Dict[str, str]] = {}
        self.labels_cache: Dict[str, str] = {}
        self.selected_method: str = "weighted-category"
        self.method_descriptions: Dict[str, str] = {}
        self.method_aliases: Dict[str, str] = {}
        self.comparison_rows: List[Dict] = []
        self.superfeatures: Dict[str, Dict[str, Any]] = {}
        self.concept_mapping: Dict[str, Any] = {}

        self._load_cached_labels()
        logger.info(
            "Labeling service ready (%s cached, methods=%s)",
            len(self.labels_cache),
            self.available_methods,
        )

    @property
    def available_methods(self) -> List[str]:
        preferred_order = [
            "llm-review-based",
            "llm-based",
            "matrix-based",
            "weighted-category",
        ]
        seen = set()
        ordered = []
        for method_name in preferred_order:
            if method_name in self.labels_by_method and method_name not in seen:
                ordered.append(method_name)
                seen.add(method_name)
        for method_name in sorted(self.labels_by_method.keys()):
            if method_name not in seen:
                ordered.append(method_name)
        return ordered

    def _load_cached_labels(self) -> None:
        source_path = self.labels_source_path

        if source_path.is_dir():
            self._load_from_directory(source_path)
        elif source_path.exists():
            self._load_from_file(source_path)
        else:
            logger.warning("Labels file not found: %s", source_path)

        self._select_default_method()

    def _load_from_directory(self, labels_dir: Path) -> None:
        loaded = {}
        metadata_path = labels_dir / "neuron_labels.json"
        if metadata_path.exists():
            self._load_from_file(metadata_path, method_name="weighted-category")

        for label_file in sorted(labels_dir.glob("labels_*.pkl")):
            method_name = label_file.stem[len("labels_") :]
            try:
                with open(label_file, "rb") as f:
                    data = pickle.load(f)
                method_name = self._normalize_method_name(method_name)
                loaded[method_name] = self._normalize_label_dict(data)
            except Exception as e:
                logger.warning("Failed to load %s: %s", label_file, e)

        if not loaded:
            self._load_extra_artifacts(labels_dir)
            for fallback_name in ("neuron_labels.json", "labels.json"):
                fallback_path = labels_dir / fallback_name
                if fallback_path.exists():
                    self._load_from_file(fallback_path, method_name="weighted-category")
                    return

        self.labels_by_method.update(loaded)
        self._load_extra_artifacts(labels_dir)

    def _load_from_file(self, labels_file: Path, method_name: str = "weighted-category") -> None:
        try:
            if labels_file.suffix.lower() == ".pkl":
                with open(labels_file, "rb") as f:
                    data = pickle.load(f)
            else:
                with open(labels_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

            if isinstance(data, dict) and "methods" in data:
                self.method_descriptions = {
                    str(k): str(v)
                    for k, v in (data.get("method_descriptions") or {}).items()
                }
                self.method_aliases = {
                    str(k): str(v)
                    for k, v in (data.get("method_aliases") or {}).items()
                }
                self.comparison_rows = list(data.get("comparison") or [])
                self.superfeatures = {
                    str(k): v for k, v in (data.get("superfeatures") or {}).items()
                }
                self.concept_mapping = dict(data.get("concept_mapping") or {})
                for raw_method, labels in (data.get("methods") or {}).items():
                    normalized_method = self._normalize_method_name(str(raw_method))
                    self.labels_by_method[normalized_method] = self._normalize_label_dict(
                        labels
                    )
                selected = data.get("selected_method")
                if selected:
                    self.selected_method = self._normalize_method_name(str(selected))
                return
            elif isinstance(data, dict) and "neuron_labels" in data:
                data = data["neuron_labels"]

            normalized_method = self._normalize_method_name(method_name)
            self.labels_by_method[normalized_method] = self._normalize_label_dict(data)
        except json.JSONDecodeError as e:
            logger.warning("Failed to load labels JSON from %s: %s", labels_file, e)
        except Exception as e:
            logger.warning("Failed to load labels from %s: %s", labels_file, e)

    def _load_extra_artifacts(self, labels_dir: Path) -> None:
        superfeatures_path = labels_dir / "superfeatures.pkl"
        if superfeatures_path.exists():
            try:
                with open(superfeatures_path, "rb") as f:
                    data = pickle.load(f)
                self.superfeatures = {str(k): v for k, v in (data or {}).items()}
            except Exception as e:
                logger.warning("Failed to load %s: %s", superfeatures_path, e)

        concept_mapping_path = labels_dir / "concept_mapping.pkl"
        if concept_mapping_path.exists():
            try:
                with open(concept_mapping_path, "rb") as f:
                    self.concept_mapping = pickle.load(f) or {}
            except Exception as e:
                logger.warning("Failed to load %s: %s", concept_mapping_path, e)

    def _normalize_method_name(self, method_name: str) -> str:
        if method_name in {"tag-based", "default"}:
            return "weighted-category"
        return self.method_aliases.get(method_name, method_name)

    @staticmethod
    def _normalize_label_dict(data) -> Dict[str, str]:
        if not isinstance(data, dict):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _select_default_method(self) -> None:
        preferred_order = [
            self.selected_method,
            "llm-review-based",
            "llm-based",
            "matrix-based",
            "weighted-category",
        ]
        for method_name in preferred_order:
            if method_name in self.labels_by_method:
                self.selected_method = method_name
                self.labels_cache = dict(self.labels_by_method[method_name])
                return

        if self.labels_by_method:
            self.selected_method = next(iter(self.labels_by_method.keys()))
            self.labels_cache = dict(self.labels_by_method[self.selected_method])
        else:
            self.selected_method = "weighted-category"
            self.labels_cache = {}

    def set_method(self, method_name: str) -> None:
        method_name = self._normalize_method_name(method_name)
        if method_name not in self.labels_by_method:
            logger.warning(
                "Label method '%s' not available; keeping '%s'",
                method_name,
                self.selected_method,
            )
            return

        self.selected_method = method_name
        self.labels_cache = dict(self.labels_by_method[method_name])

    def get_superfeatures(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.superfeatures)

    def get_superfeature(self, superfeature_id: str) -> Optional[Dict[str, Any]]:
        return self.superfeatures.get(str(superfeature_id))

    def get_concept_mapping(self) -> Dict[str, Any]:
        return dict(self.concept_mapping)

    def resolve_superfeature_to_neurons(self, superfeature_id: str) -> Dict[int, float]:
        superfeature = self.get_superfeature(superfeature_id)
        if not superfeature:
            return {}

        neurons = [int(n) for n in superfeature.get("neurons", [])]
        if not neurons:
            return {}
        return {neuron_idx: 1.0 for neuron_idx in neurons}

    def resolve_concept_to_neurons(
        self, concept_id: str, top_k: int = 8
    ) -> Dict[int, float]:
        concepts = self.concept_mapping.get("concepts", [])
        for concept in concepts:
            if concept.get("concept_id") == concept_id:
                top_neurons = concept.get("top_neurons", [])[:top_k]
                return {
                    int(entry["neuron_idx"]): float(entry["score"])
                    for entry in top_neurons
                    if float(entry.get("score", 0.0)) > 0
                }
        return {}

    def _persist_path(self) -> Path:
        if self.labels_source_path.is_dir():
            return self.labels_source_path / "neuron_labels.json"
        return self.labels_source_path

    def get_label(self, neuron_idx: int) -> str:
        cached_key = str(neuron_idx)
        if cached_key in self.labels_cache:
            return self.labels_cache[cached_key]

        fallback = f"Feature {neuron_idx}"
        self.labels_cache[cached_key] = fallback
        return fallback

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
        for neuron_idx in range(num_neurons):
            if str(neuron_idx) not in self.labels_cache:
                label = self.get_label(neuron_idx)
                logger.debug(f"Labeled neuron {neuron_idx}: {label}")

        logger.info("✅ Pre-computation complete")
