"""Concept and superfeature steering UI for Streamlit.

This panel reads saved interpretability artifacts and applies steering at inference time.
It does not retrain models or regenerate labels.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import streamlit as st

logger = logging.getLogger(__name__)


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
):
    """Render concept/superfeature steering and apply it live."""
    labels_service = session_state.get("labels")
    data_service = session_state.get("data")
    selected_user = selected_user or session_state.get("current_user_id")

    if not labels_service:
        st.error("Labeling service not initialized.")
        return
    if not selected_user:
        st.info("Select a user first to try concept steering.")
        return

    try:
        from src.models.concept_steering import ConceptSteering
    except ImportError:
        st.error("Concept steering search dependencies are not installed.")
        return

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

    selected_method = labels_service.selected_method
    search_index = _build_search_index(labels_service, selected_method)
    if not search_index:
        st.info("No concepts or labels available for this method yet.")
        return

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
        return

    results = search_engine.find_related_neurons(query, top_k=top_k)
    if not results:
        st.warning("No matching concepts found.")
        return

    chosen_entity = None
    chosen_weights = {}
    chosen_type = "concept"

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
            st.write(f"{similarity:.3f}")
        with columns[3]:
            if st.checkbox("", key=f"concept_pick_{selected_method}_{entity_id}"):
                chosen_entity = str(entity_id)
                chosen_weights = neuron_weights
                chosen_type = entity_type

    alpha = st.slider(
        "Steering strength",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        key=f"concept_alpha_{selected_method}",
    )

    if not chosen_weights:
        st.info("Select one result to apply steering.")
        return

    if st.button("Apply Concept Steering", key=f"apply_concept_{selected_method}"):
        steering_config = {
            "type": chosen_type,
            "alpha": alpha,
            "neuron_weights": chosen_weights,
        }
        if chosen_type == "concept":
            steering_config["concept_id"] = chosen_entity
        elif chosen_type == "superfeature":
            steering_config["superfeature_id"] = chosen_entity.split(":", 1)[1]
        else:
            steering_config["neuron_values"] = chosen_weights

        valid_item_ids = data_service.get_valid_item_ids() if data_service else None
        recommendations = inference_service.get_recommendations_with_delta(
            selected_user,
            steering_config=steering_config,
            top_k=20,
            valid_item_ids=valid_item_ids,
        )
        if data_service:
            poi_indices = [r.get("item_id") or r.get("poi_idx") for r in recommendations]
            session_state[f"poi_details_map_{selected_user}"] = data_service.get_poi_details_batch(
                poi_indices
            )
        session_state[f"live_demo::{selected_user}::steered_recommendations"] = recommendations
        session_state[f"live_demo::{selected_user}::active_steering_config"] = steering_config
        session_state[f"live_demo::{selected_user}::displayed_recommendations"] = recommendations
        session_state[f"live_demo::{selected_user}::feature_chart_original"] = None
        session_state[f"live_demo::{selected_user}::feature_chart_steered"] = None
        session_state.current_recommendations = recommendations
        session_state.steering_modified = True
        session_state.current_steering_config = steering_config
        st.success(
            f"Applied {chosen_type} steering using {len(chosen_weights)} saved neuron targets."
        )
