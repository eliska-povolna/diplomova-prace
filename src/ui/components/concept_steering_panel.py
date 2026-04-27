"""Concept and superfeature steering UI for Streamlit.

This panel reads saved interpretability artifacts and prepares draft steering
updates. It does not retrain models or regenerate labels.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import streamlit as st

from src.ui.steering_state import get_steering_config

logger = logging.getLogger(__name__)


def _format_similarity_explanation(similarity: float) -> str:
    """Return short interpretation text for cosine similarity."""
    if similarity >= 0.75:
        return "Very strong semantic match"
    if similarity >= 0.55:
        return "Strong semantic match"
    if similarity >= 0.35:
        return "Moderate semantic match"
    if similarity >= 0.15:
        return "Weak semantic match"
    if similarity >= 0.0:
        return "Very weak semantic match"
    return "Opposite semantic direction"


def _build_adjusted_weights(
    base_weights: Dict[int, float],
    key_prefix: str,
) -> Dict[int, float]:
    """Build neuron steering weights from per-neuron UI controls."""
    if not base_weights:
        return {}

    global_scale = st.slider(
        "Overall concept strength",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.05,
        key=f"{key_prefix}::global_scale",
        help="Scales all neurons from this concept before adding them to draft.",
    )

    visible_neurons = st.slider(
        "Max features to show",
        min_value=1,
        max_value=16,
        value=8,
        step=1,
        key=f"{key_prefix}::visible_neurons",
        help="Shows top neurons by absolute weight for manual control.",
    )

    sorted_items = sorted(
        ((int(idx), float(weight)) for idx, weight in base_weights.items()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    selected_items = sorted_items[:visible_neurons]

    adjusted = dict(base_weights)
    with st.expander("Fine-tune individual neuron influence", expanded=False):
        st.caption(
            "Set multiplier per neuron. 1.0 keeps original concept weight, "
            "0.0 removes neuron from this concept patch."
        )
        for neuron_idx, base_weight in selected_items:
            multiplier = st.slider(
                f"Neuron {neuron_idx} (base {base_weight:+.3f})",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.05,
                key=f"{key_prefix}::neuron_scale::{neuron_idx}",
            )
            adjusted[neuron_idx] = float(base_weight) * float(multiplier)

    output = {}
    for neuron_idx, weight in adjusted.items():
        scaled = float(weight) * float(global_scale)
        if abs(scaled) >= 1e-9:
            output[int(neuron_idx)] = scaled

    return output


def _build_search_index(labels_service, selected_method: str) -> Dict[str, str]:
    if selected_method == "matrix-based":
        concepts = labels_service.get_concept_mapping().get("concepts", [])
        return {
            str(concept["concept_id"]): str(concept.get("display_name") or concept["concept_id"])
            for concept in concepts
        }

    if selected_method.startswith("llm"):
        superfeatures = labels_service.get_superfeatures()
        search_index = {}
        if superfeatures:
            search_index.update({
                f"superfeature:{sf_id}": str(sf_data.get("super_label", f"Superfeature {sf_id}"))
                for sf_id, sf_data in superfeatures.items()
            })
        hidden_dim = getattr(getattr(st.session_state.get("inference"), "sae", None), "hidden_dim", 0)
        search_index.update(
            {str(idx): labels_service.get_label(idx) for idx in range(hidden_dim)}
        )
        return search_index

    hidden_dim = getattr(getattr(st.session_state.get("inference"), "sae", None), "hidden_dim", 0)
    return {str(idx): labels_service.get_label(idx) for idx in range(hidden_dim)}


def _resolve_result(
    selected_method: str,
    entity_id: str,
    score: float,
    labels_service,
) -> Tuple[str, Dict[int, float], str]:
    if selected_method == "matrix-based":
        neuron_weights = labels_service.resolve_concept_to_neurons(entity_id)
        label = next(
            (
                concept.get("display_name", entity_id)
                for concept in labels_service.get_concept_mapping().get("concepts", [])
                if concept.get("concept_id") == entity_id
            ),
            entity_id,
        )
        return label, neuron_weights, "concept"

    if selected_method.startswith("llm") and entity_id.startswith("superfeature:"):
        superfeature_id = entity_id.split(":", 1)[1]
        superfeature = labels_service.get_superfeature(superfeature_id) or {}
        return (
            str(superfeature.get("super_label", f"Superfeature {superfeature_id}")),
            labels_service.resolve_superfeature_to_neurons(superfeature_id),
            "superfeature",
        )

    return (
        labels_service.get_label(int(entity_id)),
        {int(entity_id): max(0.1, float(score))},
        "neuron",
    )


def render_concept_steering_panel(
    inference_service,
    config: dict,
    session_state,
    selected_user: str | None = None,
) -> Optional[Dict[int, float]]:
    """Render concept steering selector and return draft neuron updates."""
    labels_service = session_state.get("labels")
    selected_user = selected_user or session_state.get("current_user_id")

    if not labels_service:
        st.error("Labeling service not initialized.")
        return None
    if not selected_user:
        st.info("Select a user first to try concept steering.")
        return None

    try:
        from src.models.concept_steering import ConceptSteering
    except ImportError:
        st.error("Concept steering search dependencies are not installed.")
        return None

    st.markdown("## Concept Steering")
    st.caption(
        "Use saved interpretability artifacts to steer recommendations semantically. "
        "Matrix-based labels search saved concept mappings, while LLM-based labels can search "
        "saved superfeatures or individual neuron labels."
    )
    st.markdown(
        """
Search for a concept in plain language, inspect the matching concepts or grouped features,
and apply steering through the same hidden-space mechanism used for direct neuron steering.
"""
    )
    st.caption(
        "Similarity is cosine similarity between your query embedding and concept label "
        "embedding, usually in range -1 to 1, where higher means closer semantics."
    )

    selected_method = labels_service.selected_method
    search_index = _build_search_index(labels_service, selected_method)
    if not search_index:
        st.info("No concepts or labels available for this method yet.")
        return None

    search_cache_key = f"concept_search::{selected_method}"
    cached = session_state.get(search_cache_key)
    if not cached or cached.get("labels") != search_index:
        session_state[search_cache_key] = {
            "labels": dict(search_index),
            "engine": ConceptSteering(search_index),
        }
    search_engine = session_state[search_cache_key]["engine"]

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search concepts",
            placeholder="e.g. japanese food, quiet cafes, nightlife",
            key=f"concept_query_{selected_method}",
        )
    with col2:
        top_k = st.slider(
            "Top-K",
            min_value=1,
            max_value=15,
            value=8,
            key=f"concept_top_k_{selected_method}",
        )

    if not query:
        st.info(
            "Enter a concept to search the saved concept mappings, superfeatures, or neuron labels."
        )
        return None

    results = search_engine.find_related_neurons(query, top_k=top_k)
    if not results:
        st.warning("No matching concepts found.")
        return None

    chosen_entity = None
    chosen_weights: Dict[int, float] = {}
    chosen_type = "concept"
    chosen_similarity = 0.0
    existing_config = get_steering_config(session_state, selected_user) or {}
    active_alpha = float(existing_config.get("alpha", 0.3))

    for rank, (entity_id, label, similarity) in enumerate(results, 1):
        resolved_label, neuron_weights, entity_type = _resolve_result(
            selected_method, str(entity_id), similarity, labels_service
        )
        if not neuron_weights:
            continue

        columns = st.columns([1, 4, 2, 1])
        with columns[0]:
            st.write(f"{rank}.")
        with columns[1]:
            st.write(f"**{resolved_label}**")
            st.caption(f"{entity_type.title()} · {len(neuron_weights)} neurons")
        with columns[2]:
            st.write(f"Similarity: {similarity:.3f}")
            st.caption(_format_similarity_explanation(float(similarity)))
        with columns[3]:
            if st.checkbox(
                "Select concept",
                key=f"concept_pick_{selected_method}_{entity_id}",
                label_visibility="collapsed",
            ):
                chosen_entity = str(entity_id)
                chosen_weights = neuron_weights
                chosen_type = entity_type
                chosen_similarity = float(similarity)

    st.caption(f"Using global steering alpha: {active_alpha:.2f}")

    if not chosen_weights:
        st.info("Select one result to add concept weights into steering draft.")
        return None

    adjusted_weights = _build_adjusted_weights(
        chosen_weights,
        key_prefix=f"concept_adjust::{selected_method}::{chosen_entity}",
    )

    auto_scale_similarity = st.checkbox(
        "Auto-scale by similarity",
        value=True,
        key=f"concept_auto_scale_similarity::{selected_method}",
        help="If enabled, concept neuron weights are multiplied by the selected match similarity.",
    )
    if auto_scale_similarity:
        adjusted_weights = {
            int(neuron_idx): float(weight) * float(chosen_similarity)
            for neuron_idx, weight in adjusted_weights.items()
            if abs(float(weight) * float(chosen_similarity)) >= 1e-9
        }
        st.caption(
            f"Similarity scaling applied: x {chosen_similarity:.3f} to all selected concept weights."
        )

    st.caption(
        f"Prepared draft patch: {len(adjusted_weights)} neurons after scaling controls."
    )

    if st.button(
        "Add Selected Concept to Draft", key=f"apply_concept_{selected_method}"
    ):
        st.success(
            f"Added {chosen_type} steering draft with {len(adjusted_weights)} neuron targets. "
            "Click Apply Steering below tabs to recompute recommendations."
        )
        return dict(adjusted_weights)

    return None
