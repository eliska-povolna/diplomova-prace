"""Shared steering state utilities for the Streamlit UI."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional


def demo_state_key(selected_user: str, suffix: str) -> str:
    return f"live_demo::{selected_user}::{suffix}"


def normalize_neuron_values(
    neuron_values: Optional[Mapping[Any, Any]],
) -> Dict[int, float]:
    normalized: Dict[int, float] = {}
    for raw_idx, raw_value in (neuron_values or {}).items():
        try:
            neuron_idx = int(raw_idx)
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if abs(value) < 1e-9:
            continue
        normalized[neuron_idx] = value
    return normalized


def build_steering_config(
    *,
    neuron_values: Optional[Mapping[Any, Any]],
    alpha: float = 0.3,
    source: str = "neuron",
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    normalized = normalize_neuron_values(neuron_values)
    if not normalized:
        return None

    return {
        "type": "neuron",
        "source": source,
        "alpha": float(alpha),
        "neuron_values": normalized,
        "provenance": dict(provenance or {}),
    }


def get_steering_config(session_state, selected_user: str) -> Optional[Dict[str, Any]]:
    config = session_state.get(demo_state_key(selected_user, "active_steering_config"))
    if not config:
        return None

    return build_steering_config(
        neuron_values=config.get("neuron_values"),
        alpha=float(config.get("alpha", 0.3)),
        source=str(config.get("source", "neuron")),
        provenance=config.get("provenance") or {},
    )


def set_steering_config(
    session_state,
    selected_user: str,
    *,
    neuron_values: Optional[Mapping[Any, Any]],
    alpha: float = 0.3,
    source: str = "neuron",
    provenance: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    config = build_steering_config(
        neuron_values=neuron_values,
        alpha=alpha,
        source=source,
        provenance=provenance,
    )
    session_state[demo_state_key(selected_user, "active_steering_config")] = config
    return config


def steering_config_hash(config: Optional[Mapping[str, Any]]) -> str:
    if not config:
        return ""

    serializable = {
        "type": "neuron",
        "source": str(config.get("source", "neuron")),
        "alpha": float(config.get("alpha", 0.3)),
        "neuron_values": normalize_neuron_values(config.get("neuron_values")),
    }
    payload = json.dumps(serializable, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_inference_config(
    config: Optional[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not config:
        return None
    return {
        "type": "neuron",
        "alpha": float(config.get("alpha", 0.3)),
        "neuron_values": normalize_neuron_values(config.get("neuron_values")),
    }
