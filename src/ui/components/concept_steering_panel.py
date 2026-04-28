"""Concept and superfeature steering UI for Streamlit.

This panel reads saved interpretability artifacts and prepares draft steering
updates. It does not retrain models or regenerate labels.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import streamlit as st

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


def _render_inline_concept_strength(
    neuron_weights: Dict[int, float],
    similarity_score: float,
    key_prefix: str,
) -> Tuple[Dict[int, float], bool]:
    """Render slider inline for strength adjustment.
    
    Returns:
        (selected_weights dict, whether user configured it)
    """
    if not neuron_weights:
        return {}, False

    strength_col, explain_col = st.columns([2, 3])
    with strength_col:
        strength_value = st.slider(
            "Strength",
            min_value=0.0,
            max_value=2.0,
            value=float(similarity_score),
            step=0.05,
            key=f"{key_prefix}::strength",
        )
    with explain_col:
        st.caption(f"💡 Automatically set to similarity score ({similarity_score:.3f})")

    # All selected neurons get the slider value
    return {int(idx): float(strength_value) for idx in neuron_weights.keys()}, True


def _build_search_index(labels_service, selected_method: str) -> Dict[str, str]:
    if selected_method == "matrix-based":
        concepts = labels_service.get_concept_mapping().get("concepts", [])
        return {
            str(concept["concept_id"]): str(
                concept.get("display_name") or concept["concept_id"]
            )
            for concept in concepts
        }

    if selected_method.startswith("llm"):
        superfeatures = labels_service.get_superfeatures()
        search_index = {}
        if superfeatures:
            search_index.update(
                {
                    f"superfeature:{sf_id}": str(
                        sf_data.get("super_label", f"Superfeature {sf_id}")
                    )
                    for sf_id, sf_data in superfeatures.items()
                }
            )
        hidden_dim = getattr(
            getattr(st.session_state.get("inference"), "sae", None), "hidden_dim", 0
        )
        search_index.update(
            {str(idx): labels_service.get_label(idx) for idx in range(hidden_dim)}
        )
        return search_index

    hidden_dim = getattr(
        getattr(st.session_state.get("inference"), "sae", None), "hidden_dim", 0
    )
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
    """Render concept steering with checkbox selection and inline strength sliders.
    
    Each search result shows a checkbox. When checked, a strength slider appears inline.
    All selected concepts are combined and returned as draft neuron values.
    """
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

    st.markdown("### Select concepts to draft")

    all_selected_weights = {}

    for rank, (entity_id, label, similarity) in enumerate(results, 1):
        resolved_label, neuron_weights, entity_type = _resolve_result(
            selected_method, str(entity_id), similarity, labels_service
        )
        if not neuron_weights:
            continue

        # Result row: checkbox + label + type + similarity
        col_checkbox, col_label, col_type, col_similarity = st.columns([0.8, 3, 1.2, 1])

        with col_checkbox:
            is_selected = st.checkbox(
                label="",
                value=False,
                key=f"concept_select_{rank}",
            )

        with col_label:
            st.write(f"**#{rank} {resolved_label}**")

        with col_type:
            st.caption(f"{entity_type.title()}")

        with col_similarity:
            st.write(f"**{similarity:.3f}**")

        st.caption(f"{entity_type.title()} · {len(neuron_weights)} neurons")

        # Show inline strength slider and explanation when selected
        if is_selected:
            selected_weights, _ = _render_inline_concept_strength(
                neuron_weights,
                similarity_score=float(similarity),
                key_prefix=f"concept_result_{rank}",
            )
            # Add all neurons from this concept with the selected strength
            for neuron_idx, weight in selected_weights.items():
                all_selected_weights[int(neuron_idx)] = float(weight)
            st.divider()

    return all_selected_weights if all_selected_weights else None
