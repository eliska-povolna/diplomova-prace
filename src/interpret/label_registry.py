"""Structured persistence helper for multi-method neuron labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LabelRegistry:
    """Store per-method labels plus a selected/default method."""

    methods: Dict[str, Dict[int, str]] = field(default_factory=dict)
    selected_method: str = "weighted-category"
    method_descriptions: Dict[str, str] = field(default_factory=dict)
    method_aliases: Dict[str, str] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict:
        methods_str = {
            method: {str(neuron_id): str(label) for neuron_id, label in labels.items()}
            for method, labels in self.methods.items()
        }
        selected_labels = methods_str.get(self.selected_method, {})
        return {
            "selected_method": self.selected_method,
            "neuron_labels": selected_labels,
            "methods": methods_str,
            "method_descriptions": dict(self.method_descriptions),
            "method_aliases": dict(self.method_aliases),
            "comparison": self.comparison_rows(),
            **self.extras,
        }

    def comparison_rows(self) -> List[dict]:
        neuron_ids = sorted(
            {
                int(neuron_id)
                for labels in self.methods.values()
                for neuron_id in labels.keys()
            }
        )
        rows = []
        for neuron_id in neuron_ids:
            method_labels = {
                method: str(labels.get(neuron_id, ""))
                for method, labels in self.methods.items()
            }
            rows.append(
                {
                    "neuron_id": neuron_id,
                    "selected_method": self.selected_method,
                    "selected_label": method_labels.get(self.selected_method, ""),
                    "labels": method_labels,
                }
            )
        return rows
