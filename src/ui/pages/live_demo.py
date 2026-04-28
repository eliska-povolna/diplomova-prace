"""Live demo page � Interactive steering (main interactive page)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from io import BytesIO
from typing import Dict, List, Optional
from scipy.sparse import csr_matrix
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from src.ui.steering_state import (
    get_steering_config,
    set_steering_config,
    steering_config_hash,
    to_inference_config,
)
from src.ui.utils import info_section
from src.ui.utils.formatting import format_feature_id, format_features_list
from src.ui.components.concept_steering_panel import render_concept_steering_panel
from src.utils.evaluation import ndcg_at_k

try:
    import folium
    from streamlit_folium import st_folium

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

logger = logging.getLogger(__name__)


def _compute_data_hash(
    recommendations: List[Dict], past_visits: Optional[List[Dict]] = None
) -> str:
    """
    Compute a stable hash of recommendation and past visit data.

    Used to detect when the map data has actually changed (not just Streamlit reruns).
    Returns the same hash if data content is identical, even across reruns.

    Args:
        recommendations: List of recommendation dicts
        past_visits: Optional list of past visit POI dicts

    Returns:
        SHA256 hex digest of the data
    """
    try:
        # Serialize data to JSON (excludes non-serializable fields, provides stable format)
        data_for_hash = {
            "recommendations": [
                {
                    "item_id": r.get("item_id"),
                        # "rank_after": r.get("rank_after"),
                        # "score": round(
                        #     r.get("score", 0), 3
                    # ),  # Round to avoid float precision issues
                }
                for r in (recommendations or [])
            ],
            "past_visits_count": len(past_visits or []),
        }

        json_str = json.dumps(data_for_hash, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    except Exception as e:
        logger.debug(f"Failed to compute hash: {e}, returning empty")
        return ""


def _demo_state_key(selected_user: str, suffix: str) -> str:
    return f"live_demo::{selected_user}::{suffix}"


def _normalize_terms(text: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", str(text).lower())
        if len(token) > 1
    }


def _recommendation_item_ids(recommendations: List[Dict]) -> List[int]:
    item_ids: List[int] = []
    for reco in recommendations or []:
        item_id = reco.get("item_id") or reco.get("poi_idx")
        if item_id is None:
            continue
        try:
            item_ids.append(int(item_id))
        except (TypeError, ValueError):
            continue
    return item_ids


def _compute_live_ndcg(inference, selected_user: str, recommendations: List[Dict]) -> float | None:
    if not recommendations:
        return None

    try:
        y_true = inference.get_user_ground_truth(selected_user)
    except Exception:
        return None

    if y_true is None or len(y_true) == 0:
        return None

    item_ids = _recommendation_item_ids(recommendations)
    if not item_ids:
        return None

    argsorted = np.asarray(item_ids, dtype=int)
    k = min(len(argsorted), len(y_true))
    if k <= 0:
        return None

    return float(ndcg_at_k(np.asarray(y_true), argsorted, k))


def _format_ndcg_delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "N/A"
    if before <= 0:
        if after <= 0:
            return "0.0%"
        return f"+{after * 100:.1f}%"
    return f"{((after - before) / before) * 100.0:+.1f}%"


def _compute_neuron_cpr(
    label_method: str,
    label_text: str,
    recommendations: List[Dict],
    poi_details_map: Dict,
) -> float | None:
    if label_method not in {"matrix-based", "weighted-category"}:
        return None

    label_terms = _normalize_terms(label_text)
    if not label_terms or not recommendations:
        return None

    matches = 0
    counted = 0
    for reco in recommendations:
        item_id = reco.get("item_id") or reco.get("poi_idx")
        if item_id is None:
            continue
        counted += 1
        details = poi_details_map.get(item_id, {}) if poi_details_map else {}
        category_terms = _normalize_terms(details.get("category", ""))
        if label_terms & category_terms:
            matches += 1

    if counted == 0:
        return None
    return float(matches) / float(counted)


def merge_steering_vectors(
    neuron_values: Dict[int, float],
    concept_neuron_values: Dict[int, float],
) -> Dict[int, float]:
    """Combine steering sources by summing neuron-level contributions."""
    merged = dict(neuron_values or {})
    for raw_idx, raw_value in (concept_neuron_values or {}).items():
        try:
            neuron_idx = int(raw_idx)
            weight = float(raw_value)
        except (TypeError, ValueError):
            continue

        merged[neuron_idx] = float(merged.get(neuron_idx, 0.0)) + weight

        if abs(merged[neuron_idx]) < 1e-9:
            merged.pop(neuron_idx, None)

    return merged


def _set_demo_recommendation_state(
    selected_user: str,
    *,
    base_recommendations: Optional[List[Dict]] = None,
    steered_recommendations: Optional[List[Dict]] = None,
    active_steering_config: Optional[Dict] = None,
    poi_details_map: Optional[Dict] = None,
) -> None:
    if base_recommendations is not None:
        st.session_state[_demo_state_key(selected_user, "base_recommendations")] = (
            base_recommendations
        )
    if steered_recommendations is not None:
        st.session_state[_demo_state_key(selected_user, "steered_recommendations")] = (
            steered_recommendations
        )

    active_config = active_steering_config
    if active_config is None:
        active_config = st.session_state.get(
            _demo_state_key(selected_user, "active_steering_config")
        )

    st.session_state[_demo_state_key(selected_user, "active_steering_config")] = (
        active_config
    )

    displayed = (
        steered_recommendations
        if active_config and steered_recommendations is not None
        else (
            st.session_state.get(
                _demo_state_key(selected_user, "steered_recommendations")
            )
            if active_config
            else (
                base_recommendations
                if base_recommendations is not None
                else st.session_state.get(
                    _demo_state_key(selected_user, "base_recommendations"), []
                )
            )
        )
    )
    st.session_state[_demo_state_key(selected_user, "displayed_recommendations")] = (
        displayed or []
    )
    st.session_state.current_recommendations = displayed or []
    st.session_state.steering_modified = bool(active_config)

    if poi_details_map is not None:
        st.session_state[f"poi_details_map_{selected_user}"] = poi_details_map


def _clear_demo_steering_state(selected_user: str) -> None:
    base = st.session_state.get(
        _demo_state_key(selected_user, "base_recommendations"), []
    )
    st.session_state[_demo_state_key(selected_user, "steered_recommendations")] = []
    set_steering_config(st.session_state, selected_user, neuron_values={})
    st.session_state[_demo_state_key(selected_user, "active_steering_config")] = None
    st.session_state[_demo_state_key(selected_user, "displayed_recommendations")] = base
    st.session_state[_demo_state_key(selected_user, "feature_chart_original")] = None
    st.session_state[_demo_state_key(selected_user, "feature_chart_steered")] = None
    st.session_state[_demo_state_key(selected_user, "steering_hash")] = ""
    st.session_state[_demo_state_key(selected_user, "draft_neuron_values")] = {}
    st.session_state[_demo_state_key(selected_user, "draft_concept_neuron_values")] = {}
    st.session_state[_demo_state_key(selected_user, "draft_source")] = None
    st.session_state[_demo_state_key(selected_user, "draft_provenance")] = {}
    st.session_state[_demo_state_key(selected_user, "sync_sliders_from_config")] = True
    st.session_state.current_recommendations = base
    st.session_state.steering_modified = False
    
    # Reset concept steering panel checkbox states
    keys_to_delete = [k for k in st.session_state.keys() if k.startswith(f"concept_select_") or k.startswith("concept_strength_")]
    for k in keys_to_delete:
        del st.session_state[k]

def _render_action_buttons(selected_user: str):
    action_col_apply, action_col_reset = st.columns(2)

    apply_clicked = action_col_apply.button(
        "Apply Steering",
        key=f"apply_steering_{selected_user}",
        width="stretch",
    )
    reset_clicked = action_col_reset.button(
        "Reset Steering",
        key=f"reset_steering_main_{selected_user}",
        width="stretch",
    )

    return apply_clicked, reset_clicked

@st.fragment
def _render_active_features_section(
    selected_user: str,
    inference,
    num_features: int,
) -> List[Dict]:
    """Display top active features for a user, enabling interactive steering.

    This Streamlit fragment renders independently without triggering full page reruns.
    It retrieves all active neurons and returns them for steering sliders.

    **Behavior**:
    - Retrieves top-k activations from user's latent vector (k=64)
    - Displays only top num_features in chart (default 9, from session_state)
    - Returns ALL activations for steering (not limited by display count)
    - Session state changes to display_num_features don't trigger rerun

    Args:
        selected_user: Yelp user ID
        inference: InferenceService with user latents

    Returns:
        List of top 64 activation dicts with neuron_id, activation, name
    """
    info_section(
        "🧠 Your Active Features",
        "Shows the top active features for this user based on their interaction history. "
        "Higher activation means this feature is more relevant to their preferences. When steering is used, the original values are shown in grey.",
    )

    try:
        user_z = inference.user_latents[selected_user]
        # Get ALL activations (large number to avoid filtering side effects)
        all_activations = inference.get_top_activations(user_z, k=32)

        if all_activations:
            original_chart = st.session_state.get(
                _demo_state_key(selected_user, "feature_chart_original")
            )
            steered_chart = st.session_state.get(
                _demo_state_key(selected_user, "feature_chart_steered")
            )
            if original_chart and steered_chart:
                plot_feature_activations(
                    steered_chart[:num_features],
                    original_activations=original_chart[:num_features],
                    chart_key_suffix=selected_user,
                )
            else:
                plot_feature_activations(
                    all_activations[:num_features],
                    chart_key_suffix=selected_user,
                )
        else:
            st.info("No active features found")

        # Return ALL activations for steering (not filtered by num_features)
        return all_activations

    except Exception as e:
        st.error(f"Failed to get activations: {e}")
        logger.exception("Activation retrieval failed")
        return []


def render_steering_tabs(
    selected_user: str,
    inference,
    activations: List[Dict],
    steering_config: Optional[Dict] = None,
    num_recommendations: int = 12,
) -> None:
    """Render neuron/concept steering tabs that only update draft state."""
    tab_neuron, tab_concept = st.tabs(["Neuron Steering", "Concept Steering"])

    with tab_neuron:
        _render_neuron_steering_draft(
            selected_user=selected_user,
            inference=inference,
            activations=activations,
            steering_config=steering_config,
            num_recommendations=num_recommendations,
        )

    with tab_concept:
        concept_draft = render_concept_steering_panel(
            inference_service=inference,
            config=getattr(inference, "config", {}),
            session_state=st.session_state,
            selected_user=selected_user,
        )
        if concept_draft:
            st.session_state[
                _demo_state_key(selected_user, "draft_concept_neuron_values")
            ] = dict(concept_draft)

def _ensure_cpr_cache(
    selected_user: str,
    inference,
    data,
    num_recommendations: int,
) -> Dict[int, tuple[float | None, float | None]]:
    labels_service = st.session_state.get("labels")
    selected_method = (
        getattr(labels_service, "selected_method", "") if labels_service else ""
    )

    cpr_cache_key = _demo_state_key(selected_user, "cpr_values")
    cpr_meta_key = _demo_state_key(selected_user, "cpr_values_meta")

    steering_config = get_steering_config(st.session_state, selected_user)
    steering_hash = steering_config_hash(steering_config)

    expected_meta = {
        "num_recommendations": int(num_recommendations),
        "steering_hash": steering_hash,
        "label_method": selected_method,
    }

    existing_meta = st.session_state.get(cpr_meta_key)
    if existing_meta == expected_meta:
        return dict(st.session_state.get(cpr_cache_key, {}) or {})

    if selected_method not in {"matrix-based", "weighted-category"}:
        st.session_state[cpr_cache_key] = {}
        st.session_state[cpr_meta_key] = expected_meta
        return {}

    base_recommendations = list(
        st.session_state.get(_demo_state_key(selected_user, "base_recommendations"), [])
        or []
    )[:num_recommendations]

    displayed_recommendations = list(
        st.session_state.get(
            _demo_state_key(selected_user, "displayed_recommendations"), []
        )
        or []
    )[:num_recommendations]

    if not displayed_recommendations:
        displayed_recommendations = base_recommendations

    cached_poi_details = dict(
        st.session_state.get(f"poi_details_map_{selected_user}", {}) or {}
    )

    needed_indices = set(_recommendation_item_ids(base_recommendations))
    needed_indices.update(_recommendation_item_ids(displayed_recommendations))

    missing_indices = [
        idx for idx in needed_indices if idx not in cached_poi_details
    ]

    if missing_indices and data is not None:
        fetched = data.get_poi_details_batch(missing_indices)
        cached_poi_details.update(fetched)
        st.session_state[f"poi_details_map_{selected_user}"] = cached_poi_details

    try:
        user_z = inference.user_latents[selected_user]
        activations = inference.get_top_activations(user_z, k=64)
    except Exception:
        activations = []

    cpr_values = {}

    for feature in activations:
        neuron_idx = feature.get("neuron_idx")
        label = feature.get("label", "")

        if neuron_idx is None:
            continue

        base_cpr = _compute_neuron_cpr(
            selected_method,
            label,
            base_recommendations,
            cached_poi_details,
        )

        current_cpr = _compute_neuron_cpr(
            selected_method,
            label,
            displayed_recommendations,
            cached_poi_details,
        )

        cpr_values[int(neuron_idx)] = (base_cpr, current_cpr)

    st.session_state[cpr_cache_key] = cpr_values
    st.session_state[cpr_meta_key] = expected_meta

    return cpr_values

def _render_neuron_steering_draft(
    selected_user: str,
    inference,
    activations: List[Dict],
    steering_config: Optional[Dict] = None,
    num_recommendations: int = 12,
) -> None:
    labels_service = st.session_state.get("labels")

    info_section(
        "Steer Your Preferences",
        "Set target activations for your top features. The model blends baseline and steered "
        "representations using alpha, then recomputes recommendations. CPR is shown live "
        "for matrix-based and weighted-category labels only.",
    )


    if not activations:
        st.info("No active features to steer")
        return

    st.markdown(
        """
Steer your preferences using this formula in the latent space:
`z_final = (1 - alpha) * z_user + alpha * z_steered`

- You can set your preference for these concepts using the sliders below - if you move the slider to the right, you will see more, if you move it to the left, you will see less places activating this feature.
- You can see the current and steered activation value under each slider.
"""
    )
    if labels_service and getattr(labels_service, "selected_method", "") not in {"matrix-based", "weighted-category"}:
        info_section(
            "🎯 Steering Metrics: NDCG available, CPR unavailable for LLM labeling methods",
            "NDCG and CPR are computed on the currently visible recommendations. "
            "NDCG compares the visible ranking against the held-out test split, and the delta shows the effect of steering versus the baseline ordering. You can see it on the top of the recommendation list."
            "CPR reports how many visible recommendations match the selected label/category concept. You can see it under each label.",
            "LLM-based labels show N/A as their evaluation would require token usage."
        )
    else:
        info_section(
            "🎯 Steering Metrics available (CPR and NDCG)",
            "NDCG and CPR are computed on the currently visible recommendations. "
            "NDCG compares the visible ranking against the held-out test split, and the delta shows the effect of steering versus the baseline ordering. You can see it on the top of the recommendation list."
            "CPR reports how many visible recommendations match the selected label/category concept. You can see it under each label.",
        )

    active_config = get_steering_config(st.session_state, selected_user) or {}
    existing_neuron_values = dict(active_config.get("neuron_values") or {})

    try:
        current_activations = inference.get_user_steering(selected_user)
    except ValueError as e:
        logger.warning(f"Could not get current activations: {e}")
        current_activations = {}

    num_features_to_display = st.session_state.get("display_num_features", 9)
    top_features = activations[:num_features_to_display]
    cols_per_row = 3

    draft_key = _demo_state_key(selected_user, "draft_neuron_values")
    existing_draft_values = dict(st.session_state.get(draft_key, {}) or {})
    merged_draft_values = dict(existing_draft_values)

    sliders_sync_key = _demo_state_key(selected_user, "sync_sliders_from_config")
    sync_sliders = bool(st.session_state.get(sliders_sync_key, False))

    selected_method = (
        getattr(labels_service, "selected_method", "") if labels_service else ""
    )

    cpr_values = _ensure_cpr_cache(
        selected_user=selected_user,
        inference=inference,
        data=st.session_state.get("data"),
        num_recommendations=num_recommendations,
    )

    for row_idx in range(0, len(top_features), cols_per_row):
        cols = st.columns(cols_per_row)

        for col_idx, col in enumerate(cols):
            feature_idx = row_idx + col_idx
            if feature_idx >= len(top_features):
                continue

            feature = top_features[feature_idx]
            neuron_idx = feature["neuron_idx"]
            label = feature["label"]

            current_val = float(
                current_activations.get(neuron_idx, feature.get("activation", 0.0))
            )

            slider_key = f"slider_{neuron_idx}_{selected_user}"

            active_or_current_value = float(
                existing_neuron_values.get(neuron_idx, current_val)
            )

            if sync_sliders or slider_key not in st.session_state:
                st.session_state[slider_key] = active_or_current_value

            with col:
                formatted_label = format_feature_id(neuron_idx, label)
                st.markdown(f"**{formatted_label}**")

                slider_col, learn_col = st.columns([5, 1])

                with slider_col:
                    user_profile = st.slider(
                        "Set your preference:",
                        min_value=-1.0,
                        max_value=2.0,
                        step=0.1,
                        key=slider_key,
                        help="If you want to see more of this feature, move the slider to the right. If less, move it to the left.",
                    )

                with learn_col:
                    if st.button(
                        "📚",
                        key=f"learn_{neuron_idx}_{selected_user}",
                        help="Learn more about this feature",
                    ):
                        st.session_state["selected_feature_id"] = neuron_idx
                        interpretability_page = st.session_state.get(
                            "_interpretability_page"
                        )
                        if interpretability_page:
                            st.switch_page(interpretability_page)

                if abs(float(user_profile) - active_or_current_value) > 1e-9:
                    merged_draft_values[neuron_idx] = float(user_profile)
                else:
                    merged_draft_values.pop(neuron_idx, None)

                if existing_neuron_values:
                    blended_activation = current_val
                    try:
                        blended_activation = inference.get_steered_neuron_activation(
                            selected_user,
                            neuron_idx,
                            float(user_profile),
                        )
                    except Exception:
                        logger.error("Could not compute blended activation.")

                    st.caption(
                        f"📊 Activation: {current_val:.2f} → steered to: {blended_activation:.2f}"
                    )
                else:
                    st.caption(f"📊 Activation: {current_val:.2f}")

                base_cpr, current_cpr = cpr_values.get(neuron_idx, (None, None))

                if selected_method in {"matrix-based", "weighted-category"}:
                    if base_cpr is not None and current_cpr is not None:
                        if active_config:
                            st.caption(f"🎯 CPR: {base_cpr:.1%} → {current_cpr:.1%}")
                        else:
                            st.caption(f"🎯 CPR: {base_cpr:.1%}")
                    elif base_cpr is not None:
                        st.caption(f"🎯 CPR: {base_cpr:.1%}")
                    else:
                        st.caption("🎯 CPR: N/A")
                else:
                    st.caption("🎯 CPR: N/A")

    st.session_state[draft_key] = merged_draft_values
    st.session_state[sliders_sync_key] = False

    if active_config:
        try:
            original_activations = inference.get_user_steering(selected_user)

            original_features = [
                {
                    **feat,
                    "activation": original_activations.get(feat["neuron_idx"], 0.0),
                }
                for feat in top_features
            ]

            steered_activations = inference.get_steered_activations(
                selected_user,
                active_config.get("neuron_values", {}),
                k=len(top_features),
            )

            st.session_state[
                _demo_state_key(selected_user, "feature_chart_original")
            ] = original_features

            st.session_state[
                _demo_state_key(selected_user, "feature_chart_steered")
            ] = steered_activations

        except Exception as e:
            logger.debug(f"Could not compute steered activations chart: {e}")
    else:
        st.session_state[_demo_state_key(selected_user, "feature_chart_original")] = None
        st.session_state[_demo_state_key(selected_user, "feature_chart_steered")] = None

def _recompute_recommendations_shared(
    selected_user: str,
    inference,
    data,
    *,
    render_diagnostics: bool = True,
) -> None:
    try:
        steering_config = get_steering_config(st.session_state, selected_user)
        steering_hash_key = _demo_state_key(selected_user, "steering_hash")
        current_hash = steering_config_hash(steering_config)
        previous_hash = st.session_state.get(steering_hash_key, "")
        steering_changed = current_hash != previous_hash

        force = st.session_state.get(
            _demo_state_key(selected_user, "force_recompute"), False
        )

        base_cache_key = _demo_state_key(selected_user, "base_recommendations")
        full_cache_key = _demo_state_key(selected_user, "full_recommendations")

        current_base = st.session_state.get(base_cache_key)
        valid_item_ids = data.get_valid_item_ids()

        num_recommendations = st.session_state.get(
            "display_num_recommendations", 12
        )

        TOP_K = 50
        baseline_recommendations = current_base

        if baseline_recommendations is None:
            baseline_recommendations = inference.get_recommendations_with_delta(
                selected_user,
                steering_config=None,
                top_k=TOP_K,
                valid_item_ids=valid_item_ids,
            )
            st.session_state[base_cache_key] = baseline_recommendations

        inference_steering = to_inference_config(steering_config)
        needs_model_recompute = steering_changed or force or current_base is None

        if needs_model_recompute:
            recommendations_with_delta = baseline_recommendations

            if inference_steering:
                recommendations_with_delta = inference.get_recommendations_with_delta(
                    selected_user,
                    steering_config=inference_steering,
                    top_k=TOP_K,
                    valid_item_ids=valid_item_ids,
                )

            full_recommendations = list(
                (recommendations_with_delta or baseline_recommendations)[:TOP_K]
            )
        else:
            full_recommendations = list(
                st.session_state.get(full_cache_key, []) or []
            )
            if not full_recommendations:
                full_recommendations = list(baseline_recommendations[:TOP_K])

        cached_poi_details = dict(
            st.session_state.get(f"poi_details_map_{selected_user}", {}) or {}
        )

        visible_indices = set(
            _recommendation_item_ids(full_recommendations[:num_recommendations])
        )
        visible_indices.update(
            _recommendation_item_ids(baseline_recommendations[:num_recommendations])
        )

        missing_indices = [
            idx for idx in visible_indices if idx not in cached_poi_details
        ]

        if missing_indices:
            fetched = data.get_poi_details_batch(missing_indices)
            cached_poi_details.update(fetched)

        valid_baseline = [
            reco
            for reco in baseline_recommendations
            if (reco.get("item_id") or reco.get("poi_idx")) is not None
        ]

        _set_demo_recommendation_state(
            selected_user,
            base_recommendations=valid_baseline,
            steered_recommendations=(
                full_recommendations if inference_steering else []
            ),
            active_steering_config=steering_config,
            poi_details_map=cached_poi_details,
        )

        st.session_state[full_cache_key] = full_recommendations
        st.session_state[steering_hash_key] = current_hash

        _ensure_cpr_cache(
            selected_user=selected_user,
            inference=inference,
            data=data,
            num_recommendations=num_recommendations,
        )

        if render_diagnostics and needs_model_recompute and inference_steering:
            up = sum(
                1 for r in full_recommendations if r.get("position_delta", 0) < 0
            )
            down = sum(
                1 for r in full_recommendations if r.get("position_delta", 0) > 0
            )
            flat = sum(
                1 for r in full_recommendations if r.get("position_delta", 0) == 0
            )

            base_score_by_item = {
                int(r.get("item_id") or r.get("poi_idx")): float(r.get("score", 0.0))
                for r in valid_baseline
                if (r.get("item_id") or r.get("poi_idx")) is not None
            }

            score_deltas = []
            for r in full_recommendations:
                item_idx = r.get("item_id") or r.get("poi_idx")
                if item_idx in base_score_by_item:
                    score_deltas.append(
                        float(r.get("score", 0.0)) - base_score_by_item[item_idx]
                    )

            avg_delta = float(np.mean(score_deltas)) if score_deltas else 0.0

            st.markdown("#### Steering Diagnostics")

            provenance = dict((steering_config or {}).get("provenance") or {})
            last_neuron_updates = int(
                provenance.get("last_apply_neuron_updates", 0)
            )
            last_concept_updates = int(
                provenance.get("last_apply_concept_updates", 0)
            )

            source_breakdown_part = ""
            if last_neuron_updates or last_concept_updates:
                source_breakdown_part = (
                    "Source breakdown: "
                    f"neuron {last_neuron_updates}, concept {last_concept_updates} | "
                )

            st.caption(
                f"Changed features: {len(inference_steering.get('neuron_values', {}))} | "
                f"{source_breakdown_part}"
                f"Rank delta: up {up}, down {down}, unchanged {flat} | "
                f"Avg score delta: {avg_delta:+.4f}"
            )

            if up == 0 and down == 0:
                st.info(
                    "No rank change in the visible top-K. This usually means target activations are close to baseline "
                    "or current alpha is too low for rank reordering."
                )

        st.session_state[_demo_state_key(selected_user, "force_recompute")] = False

    except Exception as e:
        st.error(f"Failed to generate recommendations: {e}")
        logger.exception("Recommendation generation failed")
        return


@st.fragment
def _render_poi_cards_section(
    data,
    available_width: int,
    card_width_px: int,
    show_scores: bool,
    photo_height_px: int,
    filtered_recommendations: list,
    poi_details_map: dict = None,
    selected_user: str | None = None,
    inference=None,
) -> None:
    """Isolated POI cards display fragment.

    Reruns only when recommendations or display settings change.
    Does not trigger rerun when steering or other factors change.

    Args:
        poi_details_map: Pre-fetched POI details dict (optional, for performance).
                        If provided, uses cached details instead of fetching.
    """
    if filtered_recommendations:
        visible_ndcg_k = len(filtered_recommendations)
        title_col, metric_col, info_col = st.columns([3, 1.1, 0.25])
        with title_col:
            st.subheader("🏆 Recommended for You")
        with metric_col:
            if selected_user and inference is not None:
                base_recommendations = list(
                    st.session_state.get(
                        _demo_state_key(selected_user, "base_recommendations"), []
                    )
                    or []
                )
                base_recommendations = base_recommendations[:visible_ndcg_k]
                current_ndcg = _compute_live_ndcg(
                    inference,
                    selected_user,
                    filtered_recommendations,
                )
                baseline_ndcg = _compute_live_ndcg(
                    inference,
                    selected_user,
                    base_recommendations,
                )
                if current_ndcg is not None:
                    if st.session_state.get("steering_modified"):
                        st.metric(
                            f"NDCG@{visible_ndcg_k}",
                            f"{current_ndcg:.3f}",
                            delta=_format_ndcg_delta(baseline_ndcg, current_ndcg),
                        )
                    else:
                        st.metric(f"NDCG@{visible_ndcg_k}", f"{current_ndcg:.3f}")
                else:
                    st.caption(f"NDCG@{visible_ndcg_k}: N/A")
        with info_col:
            st.button(
                "ℹ️",
                key=f"rec_info_{selected_user or 'global'}",
                help=(
                    "NDCG@k compares the shown recommendations against the held-out test split, "
                    "where k equals the number of visible recommendations. It is computed live for the currently selected user."
                ),
            )

        responsive_cards_per_row = max(1, available_width // card_width_px)
        cols = st.columns(responsive_cards_per_row)

        cached_poi_details = dict(
            poi_details_map or st.session_state.get(f"poi_details_map_{selected_user}", {}) or {}
        )
        missing_indices = [
            idx
            for idx in [r.get("item_id") or r.get("poi_idx") for r in filtered_recommendations]
            if idx is not None and idx not in cached_poi_details
        ]
        if missing_indices and data is not None:
            fetched = data.get_poi_details_batch(missing_indices)
            cached_poi_details.update(fetched)
            if selected_user:
                st.session_state[f"poi_details_map_{selected_user}"] = cached_poi_details

        valid_card_rows = []
        for reco in filtered_recommendations:
            poi_idx = reco.get("item_id") or reco.get("poi_idx")
            if poi_idx not in cached_poi_details:
                logger.debug(
                    "Skipping POI card for idx=%s (missing in batch POI details map)",
                    poi_idx,
                )
                continue

            poi_details = cached_poi_details.get(poi_idx)
            if not poi_details:
                continue
            valid_card_rows.append((reco, poi_details))

        for display_idx, (reco, poi_details) in enumerate(valid_card_rows):
            with cols[display_idx % responsive_cards_per_row]:
                try:
                    draw_poi_card(
                        poi_details,
                        reco,
                        show_scores,
                        card_width_px,
                        photo_height_px,
                    )
                except Exception as e:
                    logger.exception(
                        f"POI card error for {poi_details.get('name', 'Unknown')}: {e}"
                    )
                    continue


@st.fragment
def _render_history_section(
    selected_user: str,
    data,
    inference,
    available_width: int,
    card_width_px: int,
    photo_height_px: int,
) -> None:
    """Isolated history display fragment.

    Reruns only when display settings or history cache changes.
    Uses cached POI details - zero DB lookups on reruns.
    """
    st.subheader("📜 Your Past Visits")

    history_cache_key = f"past_visits_{selected_user}"
    history_pois_cache_key = f"past_visits_pois_{selected_user}"

    # Try to get cached POI details first (zero DB lookups)
    if history_pois_cache_key in st.session_state:
        history_pois = st.session_state.get(history_pois_cache_key, [])
        logger.debug(
            f"✅ Using cached POI details for {len(history_pois)} past visits (zero DB lookups)"
        )
    else:
        # If POI details not cached, fetch from indices (shouldn't happen if map section ran first)
        history_indices = st.session_state.get(history_cache_key, [])
        if history_indices:
            history_pois = [data.get_poi_details(idx) for idx in history_indices]
            history_pois = [p for p in history_pois if p]
            logger.debug(
                f"Loaded {len(history_pois)} valid POI details from {len(history_indices)} indices"
            )
        else:
            history_pois = []

    if history_pois:
        try:
            if history_pois:
                with st.expander(
                    f"Show {len(history_pois)} past visits", expanded=False
                ):
                    hist_cols = st.columns(max(1, available_width // card_width_px))
                    displayed_count = 0

                    for poi in history_pois:
                        try:
                            with hist_cols[displayed_count % len(hist_cols)]:
                                draw_poi_card(
                                    poi,
                                    {},
                                    show_scores=False,
                                    card_width_px=card_width_px,
                                    photo_height_px=photo_height_px,
                                )
                                displayed_count += 1
                        except Exception as e:
                            logger.exception(
                                f"History POI card error for {poi.get('name', 'Unknown')}: {e}"
                            )
                            continue
            else:
                st.info("No valid past visits found")
        except Exception as e:
            st.error(f"❌ Error displaying past visits: {e}")
            logger.exception(f"Past visits display error: {e}")


@st.fragment
def _render_map_section(
    selected_user: str,
    data,
    inference,
    filtered_recommendations: list,
    show_past_visits: bool = False,
) -> None:
    """Display interactive Folium map of recommended POIs with optional visit history.

    This Streamlit fragment isolates map rendering to prevent full page reruns when users
    interact with the map (pan, zoom, click markers).

    **Key Optimization**:
    - Maps cached using SHA256 hash of recommendation data
    - Only rebuilds when actual data changes (recommendations update)
    - Panning/zooming/clicking markers does NOT trigger rebuild
    - Map interactions are local to Folium, not full page reruns

    **Map Design**:
    - Center: Calculated from recommendation coordinates
    - Markers: Blue for recommendations, Red for past visits
    - Popups: Click marker for name and rating

    Args:
        selected_user: Yelp user ID
        data: DataService for POI details
        inference: InferenceService (context only)
        filtered_recommendations: Pre-filtered recommendations list
        show_past_visits: Whether to show past visit history (bool)

    Returns:
        None
    """
    # Read from session_state so sidebar changes don't trigger map rerun
    show_history = st.session_state.get("show_history_checkbox", False)
    if filtered_recommendations:
        info_section(
            "📍 Recommended Locations",
            "Interactive map showing recommended POI locations (colored markers). "
            "Each marker represents a place based on your interests.",
        )

        if HAS_FOLIUM:
            try:
                # STEP 1: Check if we need to load past visits
                past_visits_for_map = None
                history_cache_key = f"past_visits_{selected_user}"
                history_pois_cache_key = f"past_visits_pois_{selected_user}"

                if show_history:
                    if history_pois_cache_key in st.session_state:
                        past_visits_for_map = st.session_state.get(
                            history_pois_cache_key
                        )
                        logger.debug(
                            f"✅ Using cached POI details for {len(past_visits_for_map)} past visits (instant, no DB lookups)"
                        )
                    else:
                        logger.debug(
                            f"Past visits checkbox is ON but not cached - loading now..."
                        )
                        try:
                            past_visits_indices = inference.get_user_history(
                                selected_user
                            )
                            logger.debug(
                                f"Fetching POI details for {len(past_visits_indices)} past visits (using batch method)..."
                            )

                            # Use batch method for efficient bulk fetching instead of loop
                            past_visits_pois_batch = data.get_poi_details_batch(
                                past_visits_indices
                            )
                            past_visits_pois = [
                                past_visits_pois_batch[idx]
                                for idx in past_visits_indices
                                if idx in past_visits_pois_batch
                            ]
                            past_visits_pois = [
                                p
                                for p in past_visits_pois
                                if p and p.get("lat") and p.get("lon")
                            ]

                            st.session_state[history_cache_key] = past_visits_indices
                            st.session_state[history_pois_cache_key] = past_visits_pois
                            past_visits_for_map = past_visits_pois

                            logger.debug(
                                f"Loaded and cached {len(past_visits_pois)}/{len(past_visits_indices)} valid past visits"
                            )
                        except Exception as e:
                            logger.warning(f"Could not load past visits: {e}")

                # STEP 2: Build the map using already-filtered recommendations
                # Use filtered_recommendations (passed as parameter - rerun only on steering change)
                recommendations = filtered_recommendations or []

                # OPTIMIZATION: Cache the map object based on data content hash
                # This prevents map rebuilds when just panning/zooming
                data_hash = _compute_data_hash(recommendations, past_visits_for_map)
                map_cache_key = f"cached_folium_map_{selected_user}_{data_hash}"
                # cleanup old cached maps for this user
                cache_prefix = f"cached_folium_map_{selected_user}_"
                existing_keys = [k for k in st.session_state if k.startswith(cache_prefix)]

                if len(existing_keys) > 3:
                    for k in existing_keys[:-3]:
                        del st.session_state[k]
                if map_cache_key in st.session_state:
                    # ✅ Map data hasn't changed - use cached map (instant render!)
                    map_obj = st.session_state[map_cache_key]
                    logger.debug(
                        f"⚡ Using cached map object (no rebuild, instant pan/zoom)"
                    )
                else:
                    # Map data changed - rebuild map
                    logger.debug(f"Map data changed, rebuilding map...")
                    map_obj = build_folium_map(
                        recommendations,
                        data,
                        past_visits=past_visits_for_map,
                    )

                    if map_obj is not None:
                        # Store in cache for next pan/zoom
                        st.session_state[map_cache_key] = map_obj
                        logger.debug(f"✅ Map cached for instant rerenders on pan/zoom")

                if map_obj is None:
                    st.warning(
                        "⚠️ Could not load map. Check that recommendations have valid locations in the database."
                    )
                    logger.warning(
                        "Map object returned None - recommendations may lack coordinates"
                    )
                else:
                    st_folium(
                        map_obj,
                        width=None,
                        height=500,
                        key=f"folium_{selected_user}_{data_hash[:12]}",
                        returned_objects=[],
                    )
            except Exception as e:
                logger.exception("Map rendering failed")
                st.error("❌ Map rendering failed. Check logs for details.")
        else:
            st.info(
                "ℹ️ Install streamlit-folium for map visualization: `pip install streamlit-folium`"
            )


def show():
    """Display live demo page with interactive steering."""

    inference = st.session_state.get("inference")
    data = st.session_state.get("data")
    labels = st.session_state.get("labels")

    if not all([inference, data, labels]):
        st.error("Services not initialized")
        return

    st.title("🎛️ Interactive Steering Demo")

    st.markdown(
        """
    Adjust feature sliders to steer recommendations in real-time.
    See how each neuron influences the model's predictions.
    """
    )
    # =====================================================================
    # SIDEBAR: Controls
    # =====================================================================
    with st.sidebar:
        st.header("🎛️ Controls")

        # Initialize sidebar state
        if "sidebar_expanded" not in st.session_state:
            st.session_state.sidebar_expanded = True

        # User selection - with caching to avoid empty user list bugs
        st.subheader("Select User")

        # Load test users with caching
        test_users = None

        # Try to get from session state first (fast)
        if "cached_test_users" in st.session_state:
            test_users = st.session_state["cached_test_users"]
            logger.debug(f"✅ Using cached test users: {len(test_users)} users")

        # If not in cache, try to load
        if not test_users:
            try:
                test_users = data.get_test_users(limit=50)
                if test_users:
                    # Cache for future renders
                    st.session_state["cached_test_users"] = test_users
                    logger.debug(
                        f"Loaded {len(test_users)} test users from data service"
                    )
            except Exception as e:
                logger.error(f"❌ Failed to load test users: {e}", exc_info=True)

        if not test_users:
            st.error("❌ No test users available - check database connection")
            return

        user_options = {
            u["id"]: f"{u['id'][:8]}... ({u['interactions']} items)" for u in test_users
        }

        selected_user = st.selectbox(
            "User ID",
            options=list(user_options.keys()),
            format_func=lambda x: user_options[x],
            key="user_selectbox",
        )

        st.divider()

        # Display options
        st.subheader("Display Options")

        show_latent = st.checkbox("Show activations", value=True)

        # Checkbox with callback to load past visits when toggled ON
        def _on_history_checkbox_change():
            """When user checks 'Show past visits', load them into session state."""
            is_checked = st.session_state.get("show_history_checkbox", False)
            if is_checked:
                history_cache_key = (
                    f"past_visits_{st.session_state.get('selected_user', '')}"
                )
                if history_cache_key not in st.session_state:
                    # Trigger loading in background - next rerun will see it in session state
                    logger.debug(
                        f"Past visits checkbox enabled - will load on next render"
                    )
            else:
                # Checkbox was unchecked - no need to do anything
                logger.debug(f"Past visits checkbox disabled")

        show_history = st.checkbox(
            "Show past visits",
            value=False,
            key="show_history_checkbox",
            on_change=_on_history_checkbox_change,
        )
        show_scores = st.checkbox("Show scores", value=True)

        st.divider()

        # Output parameters
        st.subheader("Output Parameters")

        # Responsive card layout: user sets cards per row, system calculates width
        # Use 960px width for larger cards (20% increase from 800px)
        available_width = 960
        recs_per_row = st.slider(
            "Cards per row", min_value=1, max_value=5, value=3, step=1
        )

        # Calculate card width based on available width and cards per row
        card_width_px = available_width // recs_per_row

        # Calculate photo dimensions based on card width (maintain aspect ratio)
        photo_height_px = int(card_width_px * 0.733)

        st.caption(
            f"📐 Width: {card_width_px}px | Photo: {card_width_px}×{photo_height_px}px"
        )

        max_features = 32  # it is difficult to retrieve the actual number of activated neurons, but most neurons activate under 20 features
        num_features = st.slider(
            "Features to display",
            min_value=5,
            max_value=max(5, max_features),
            value=min(9, max_features),
            step=1,
            key="display_num_features_slider",
        )
        num_recommendations = st.slider("Recommendations", 5, 50, 12)

        # Store in session_state so fragments can read without rerunning
        st.session_state.display_num_features = num_features
        st.session_state.display_num_recommendations = num_recommendations

        st.divider()

    # =====================================================================
    # MAIN AREA
    # =====================================================================
    if selected_user:
        # Get/create user encoding
        user_already_encoded = (
            st.session_state.get("current_user_id") == selected_user
            and selected_user in inference.user_latents
        )

        if not user_already_encoded:
            try:
                # Check if we've already encoded this user on a previous run
                cached_csr_key = f"cached_csr_{selected_user}"
                if cached_csr_key in st.session_state:
                    logger.debug(f"Using cached CSR matrix for user {selected_user}")
                    user_interactions_csr = st.session_state[cached_csr_key]
                else:
                    # STEP 1: Try to load precomputed matrix (cloud or local pickle)
                    user_interactions_csr = data.get_precomputed_user_matrix(
                        selected_user
                    )

                    if user_interactions_csr is not None:
                        logger.info(
                            f"✅ Loaded precomputed CSR matrix for user {selected_user}"
                        )
                    else:
                        # STEP 2: Fallback - build matrix from interaction history
                        # Encode user from interaction history
                        poi_indices = data.get_user_interactions(selected_user)

                        if not poi_indices:
                            st.warning(
                                f"No interaction history found for user {selected_user}"
                            )
                            return

                        # Validate inference service is properly initialized
                        if inference.n_items is None:
                            error_msg = "Inference service not properly initialized: n_items is None"
                            logger.error(error_msg + " - Model loading failed")
                            st.error(f"❌ {error_msg}")
                            return

                        logger.debug(
                            f"Retrieved {len(poi_indices)} interactions for user {selected_user}"
                        )

                        # Create sparse CSR matrix from POI indices (1 row, n_items columns)
                        row = np.zeros(len(poi_indices), dtype=int)  # All row 0
                        col = np.array(poi_indices, dtype=int)  # POI indices as columns
                        data_vals = np.ones(len(poi_indices), dtype=np.float32)

                        # VALIDATION: Check for index out of bounds (indicates item2index mismatch)
                        if len(col) > 0:
                            max_col = col.max()
                            if max_col >= inference.n_items:
                                error_msg = (
                                    f"❌ User encoding failed: Index mismatch detected\\n\\n"
                                    f"**Issue**: The item2index mapping contains indices up to {max_col}, "
                                    f"but the inference model only has {inference.n_items} items.\\n\\n"
                                    f"**Cause**: The item2index mapping (from training) doesn't match the current model version.\\n\\n"
                                    f"**Solution**: "
                                    f"\\n1. Delete old mappings: `rm -rf outputs/*/mappings/business2index_universal.pkl`"
                                    f"\\n2. Retrain the model to generate a fresh mapping"
                                    f"\\n3. Restart the Streamlit app"
                                )
                                logger.error(error_msg.replace("\\n", " "))
                                st.error(error_msg)
                                return

                        user_interactions_csr = csr_matrix(
                            (data_vals, (row, col)), shape=(1, inference.n_items)
                        )
                        logger.debug(
                            f"Built interaction matrix: shape={user_interactions_csr.shape}, nnz={user_interactions_csr.nnz}"
                        )

                        # Validate CSR matrix was created
                        if user_interactions_csr is None:
                            raise RuntimeError(
                                f"Failed to create CSR matrix for user {selected_user}"
                            )

                        if not hasattr(user_interactions_csr, "toarray"):
                            raise TypeError(
                                f"CSR matrix missing toarray method. Type: {type(user_interactions_csr)}"
                            )

                    # Cache it for future reruns
                    st.session_state[cached_csr_key] = user_interactions_csr

                # Verify CSR matrix before encoding
                if user_interactions_csr is None:
                    raise ValueError(f"CSR matrix is None for user {selected_user}")

                # Encode user
                logger.debug(f"Encoding user {selected_user}...")
                inference.encode_user(selected_user, user_interactions_csr)
                logger.debug(f"User {selected_user} encoded successfully")
                st.session_state.current_user_id = selected_user

            except Exception as e:
                logger.exception("User encoding failed")
                st.error("❌ Failed to encode user. Check logs for details.")
                return

        # ===================================================================
        # Section 1: Active Features (isolated fragment)
        # ===================================================================
        if show_latent:
            activations = _render_active_features_section(
                selected_user=selected_user,
                inference=inference,
                num_features = st.session_state["display_num_features_slider"]
            )
        else:
            user_z = inference.user_latents[selected_user]
            activations = inference.get_top_activations(user_z, k=64)

        active_config = get_steering_config(st.session_state, selected_user) or {}
        active_alpha = float(active_config.get("alpha", 0.3))
        draft_key = _demo_state_key(selected_user, "draft_neuron_values")
        if draft_key not in st.session_state:
            st.session_state[draft_key] = {}

        draft_alpha_key = _demo_state_key(selected_user, "draft_alpha")
        if draft_alpha_key not in st.session_state:
            st.session_state[draft_alpha_key] = active_alpha

        global_alpha = st.slider(
            "Steering alpha:  controls how strongly steering influences the final ranking (if it is set to 0, no steering will be applied.)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state[draft_alpha_key],
            step=0.05,
            help="Shared steering strength used by both neuron and concept steering.",
        )
        st.session_state[draft_alpha_key] = global_alpha
        # ===================================================================
        # Section 2: Steering Inputs (draft only)
        # ===================================================================
        render_steering_tabs(
            selected_user=selected_user,
            inference=inference,
            activations=activations,
            steering_config=active_config,
            num_recommendations=num_recommendations,
        )

        draft_neuron_values = dict(st.session_state.get(draft_key, {}) or {})
        concept_draft_key = _demo_state_key(selected_user, "draft_concept_neuron_values")
        draft_concept_neuron_values = dict(
            st.session_state.get(concept_draft_key, {}) or {}
        )
        # Only show pending draft if values differ from active config (not just non-empty)
        active_neuron_values = dict(active_config.get("neuron_values") or {})
        different_neurons = {
            idx: val
            for idx, val in draft_neuron_values.items()
            if abs(float(val) - float(active_neuron_values.get(idx, 0.0))) > 1e-9
        }
        has_pending_neuron_draft = bool(different_neurons)
        has_pending_concept_draft = bool(draft_concept_neuron_values)
        has_pending_alpha = (
            bool(active_config) and abs(global_alpha - active_alpha) > 1e-9
        )

        if has_pending_neuron_draft or has_pending_concept_draft:
            pending_merged = merge_steering_vectors(
                different_neurons,
                draft_concept_neuron_values,
            )
            st.caption(
                "Pending steering draft: "
                f"{len(pending_merged)} unique feature update(s) "
                f"(neuron: {len(different_neurons)}, concept: {len(draft_concept_neuron_values)})."
            )
            with st.expander("ℹ️ What do these counts mean?"):
                st.write(
                    "**Unique** = the number of final feature IDs after merging neuron and concept drafts.\n\n"
                    "**Source counts** (in parentheses) = how many updates come from each tab before merge.\n\n"
                    "If both tabs update the same feature, source counts can be higher than unique."
                )
        if has_pending_alpha:
            st.caption(
                f"Pending alpha change: current {active_alpha:.2f} → new {global_alpha:.2f}."
            )


        apply_clicked, reset_clicked = _render_action_buttons(selected_user)
        if reset_clicked:
            _clear_demo_steering_state(selected_user)
            st.rerun()

        if apply_clicked:
            active_neuron_values = dict(active_config.get("neuron_values") or {})
            neuron_contributions: Dict[int, float] = {}
            for raw_idx, raw_value in draft_neuron_values.items():
                try:
                    neuron_idx = int(raw_idx)
                    draft_value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                baseline_value = float(active_neuron_values.get(neuron_idx, 0.0))
                neuron_contributions[neuron_idx] = draft_value - baseline_value

            merged_after_neuron = merge_steering_vectors(
                active_neuron_values,
                neuron_contributions,
            )
            # Neuron and concept steering are combined by summing their contributions
            # into a single steering vector applied before recommendation recomputation.
            merged_neuron_values = merge_steering_vectors(
                merged_after_neuron,
                draft_concept_neuron_values,
            )

            draft_source = st.session_state.get(
                _demo_state_key(selected_user, "draft_source")
            )
            if has_pending_neuron_draft and has_pending_concept_draft:
                source = "combined"
            elif has_pending_concept_draft:
                source = "concept"
            elif has_pending_neuron_draft:
                source = "neuron"
            else:
                source = str(draft_source or active_config.get("source", "neuron"))

            provenance = dict(active_config.get("provenance") or {})
            provenance.update(
                dict(
                    st.session_state.get(
                        _demo_state_key(selected_user, "draft_provenance"), {}
                    )
                    or {}
                )
            )
            if has_pending_neuron_draft:
                provenance["edited_in"] = "neuron_panel"
            provenance["last_apply_neuron_updates"] = int(len(different_neurons))
            provenance["last_apply_concept_updates"] = int(
                len(draft_concept_neuron_values)
            )

            set_steering_config(
                st.session_state,
                selected_user,
                neuron_values=merged_neuron_values,
                alpha=st.session_state.get(draft_alpha_key, active_alpha),
                source=source,
                provenance=provenance,
            )
            st.session_state[_demo_state_key(selected_user, "force_recompute")] = True
            st.session_state[draft_key] = {}
            st.session_state[concept_draft_key] = {}
            st.session_state[_demo_state_key(selected_user, "draft_source")] = None
            st.session_state[_demo_state_key(selected_user, "draft_provenance")] = {}
            st.session_state[_demo_state_key(selected_user, "sync_sliders_from_config")] = True
            st.rerun()

        # ===================================================================
        # Section 3: Shared Recommendations Compute Block
        # ===================================================================
        # Check if recommendations need recomputation (steering changed, or initial load)
        steering_config = get_steering_config(st.session_state, selected_user)
        steering_hash_key = _demo_state_key(selected_user, "steering_hash")
        current_steering_hash = steering_config_hash(steering_config)
        previous_steering_hash = st.session_state.get(steering_hash_key, "")
        steering_changed_check = current_steering_hash != previous_steering_hash
        
        # Only recompute if steering changed, initial load, or Apply was clicked
        if steering_changed_check or not st.session_state.get(_demo_state_key(selected_user, "displayed_recommendations")):
            _recompute_recommendations_shared(
                selected_user=selected_user,
                inference=inference,
                data=data,
            )

        displayed_recommendations = st.session_state.get(
            _demo_state_key(selected_user, "displayed_recommendations"),
            [],
        )
        filtered_recommendations = displayed_recommendations[:num_recommendations]

        # Section 4: Map Visualization (ALWAYS renders instantly with recommendations)
        # ===================================================================
        map_placeholder = st.empty()
        with map_placeholder.container():
            _render_map_section(
                selected_user=selected_user,
                data=data,
                inference=inference,
                filtered_recommendations=filtered_recommendations,
                show_past_visits=show_history,
            )

        # ===================================================================
        # Section 5: POI Cards (Recommendations)
        # ===================================================================
        _render_poi_cards_section(
            data=data,
            available_width=available_width,
            card_width_px=card_width_px,
            show_scores=show_scores,
            photo_height_px=photo_height_px,
            filtered_recommendations=filtered_recommendations,
            poi_details_map=st.session_state.get(
                f"poi_details_map_{selected_user}", {}
            ),
            selected_user=selected_user,
            inference=inference,
        )

        # ===================================================================
        # Section 6: User History Display (uses cached data from pre-load)
        # ===================================================================
        if show_history:
            _render_history_section(
                selected_user=selected_user,
                data=data,
                inference=inference,
                available_width=available_width,
                card_width_px=card_width_px,
                photo_height_px=photo_height_px,
            )


# =============================================================================
# Helper Functions
# =============================================================================


def plot_feature_activations(
    activations: List[Dict],
    original_activations: List[Dict] = None,
    chart_key_suffix: str = "global",
):
    """Plot horizontal bar chart of top feature activations with optional original comparison.

    Args:
        activations: List of dicts with neuron_idx, label, activation
        original_activations: Optional list of original activations to show as greyed-out bars
    """

    # Format labels with neuron index as string in description
    labels = [format_feature_id(a["neuron_idx"], a["label"]) for a in activations]
    values = [a["activation"] for a in activations]

    # Keep order with largest values at top (Plotly renders top-to-bottom)

    fig = go.Figure()

    # Generate colors for each bar (colorful spectrum - Viridis)
    # Use activation values for color mapping
    color_scale = np.linspace(0, 1, len(values))
    colors = [
        f"hsl({180 + h*180}, 100%, {40 + l*20}%)"
        for h, l in zip(color_scale, np.linspace(0, 1, len(values)))
    ]

    # Add steered activations as main bars with colorful bars
    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            name="Steered",
            marker=dict(
                color=values,  # Use values for color intensity
                colorscale="Viridis",
                showscale=False,  # Hide the colorbar legend
            ),
            text=[f"{v:.3f}" for v in values],  # Show value on bar
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Activation: %{x:.3f}<extra></extra>",
            showlegend=False,
        )
    )

    # Add original activations as greyed-out comparison bars if provided
    if original_activations:
        # Map original activations by neuron_idx for lookup
        original_map = {a["neuron_idx"]: a["activation"] for a in original_activations}
        original_values = [original_map.get(a["neuron_idx"], 0.0) for a in activations]

        fig.add_trace(
            go.Bar(
                y=labels,
                x=original_values,
                orientation="h",
                name="Original",
                marker=dict(color="rgba(150, 150, 150, 0.6)"),
                text=[f"{v:.3f}" for v in original_values],  # Show value on bar
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>Original: %{x:.3f}<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        margin=dict(l=250, b=100),  # Increased left margin for full label visibility
        height=max(400, len(labels) * 25),  # Dynamic height based on number of bars
        barmode="overlay",
        showlegend=False,
        xaxis_title="Activation Magnitude",
        yaxis=dict(autorange="reversed"),  # Ensure all labels are visible
    )

    st.plotly_chart(fig, width="stretch", key=f"activation_chart_{chart_key_suffix}")


def build_folium_map(
    recommendations: List[Dict], data_service, past_visits: List[Dict] = None
) -> folium.Map | None:
    """Build interactive Folium map with POI markers (recommendations and past visits).

    Args:
        recommendations: List of recommendation dicts with item_id/poi_idx and score
        data_service: DataService instance for fetching POI details
        past_visits: Optional list of POI detail dicts for past visits (pre-fetched POI details, not indices!)
                     This should be a list of dicts with lat/lon/name/etc, NOT a list of indices
    """

    # DEBUG: Log what we received
    logger.debug(f"build_folium_map() called with:")
    logger.debug(f"  - recommendations: {len(recommendations)} items")
    logger.debug(
        f"  - past_visits: {len(past_visits) if past_visits else 0} POI detail dicts (pre-fetched, no DB lookups needed)"
    )

    if not HAS_FOLIUM:
        logger.warning("Folium not available for map rendering")
        return None

    if not recommendations:
        logger.warning("No recommendations to display on map")
        return None

    try:
        # Get POI details for all recommendations
        # Support both new format (item_id) and old format (poi_idx)
        pois = []
        failed_pois = []
        for i, r in enumerate(recommendations):
            poi_idx = r.get("item_id") or r.get("poi_idx")
            if poi_idx is None:
                logger.warning(f"Recommendation {i}: No item_id or poi_idx found")
                failed_pois.append((i, poi_idx, "No index"))
                continue

            poi = data_service.get_poi_details(poi_idx)
            if poi and poi.get("lat") and poi.get("lon"):
                pois.append(poi)
            else:
                if not poi:
                    reason = "Empty POI (not in mapping or database)"
                elif not poi.get("lat") or not poi.get("lon"):
                    reason = (
                        f"Invalid coords: lat={poi.get('lat')}, lon={poi.get('lon')}"
                    )
                else:
                    reason = "Unknown"
                failed_pois.append((i, poi_idx, reason))

        if failed_pois:
            logger.warning(
                f"Failed to load {len(failed_pois)}/{len(recommendations)} POIs:"
            )
            for rec_idx, poi_idx, reason in failed_pois[:5]:  # Log first 5 failures
                logger.warning(f"  - Rec #{rec_idx}: poi_idx={poi_idx} ({reason})")
            if len(failed_pois) > 5:
                logger.warning(f"  ... and {len(failed_pois) - 5} more")

        if not pois:
            logger.error(
                f"No valid POI data: all {len(recommendations)} recommendations failed to load."
            )
            logger.error(
                f"Check: (1) index2item mapping exists, (2) database connectivity, (3) POI data has coordinates"
            )
            return None

        # Get past visit POIs (already pre-fetched, just use them)
        past_visit_pois = []
        if past_visits:
            # past_visits is already a list of POI detail dicts (pre-fetched from cache)
            # No need to call get_poi_details again!
            past_visit_pois = [
                p for p in past_visits if p and p.get("lat") and p.get("lon")
            ]
            logger.debug(
                f"Using {len(past_visit_pois)} pre-fetched past visit POI details (no DB lookups)"
            )
        else:
            logger.debug(f"No past visits provided to map (past_visits={past_visits})")

        # DEBUG: Log data service state
        if failed_pois:
            logger.warning(
                f"Data service state: index2item={'present' if hasattr(data_service, 'index2item') and data_service.index2item else 'missing'}, backend={getattr(data_service, 'backend_type', 'unknown')}"
            )

        # Calculate center from RECOMMENDATIONS ONLY (ignore past visits for map focus)
        lats = [p["lat"] for p in pois if p.get("lat")]
        lons = [p["lon"] for p in pois if p.get("lon")]

        if not lats or not lons:
            logger.warning("No valid coordinates for map center")
            return None

        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        logger.debug(
            f"Creating map centered at ({center_lat:.4f}, {center_lon:.4f}) with {len(pois)} recommendations and {len(past_visit_pois)} past visits"
        )

        # Create map (bounds are fitted after markers are added).
        m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap")

        # Define color palette for rank badges (hex colors)
        hex_colors = [
            "#E74C3C",  # red
            "#3498DB",  # blue
            "#2ECC71",  # green
            "#9B59B6",  # purple
            "#E67E22",  # orange
            "#C0392B",  # darkred
            "#1E3A8A",  # darkblue
            "#16A34A",  # darkgreen
            "#0891B2",  # cadetblue
            "#6B21A8",  # darkpurple
        ]

        # Add past visit markers (grey with checkmark)
        for poi in past_visit_pois:
            try:
                popup_text = f"<b>{poi.get('name', 'Unknown')}</b><br>"
                popup_text += "📜 <i>Past Visit</i><br>"
                if poi.get("category"):
                    popup_text += f"{poi['category']}<br>"
                popup_text += (
                    f"⭐ {poi.get('rating', 0)} ({poi.get('review_count', 0)} reviews)"
                )

                # Create checkmark icon using HTML (much smaller than recommendations)
                checkmark_html = """
                <div style="font-size: 12px; color: white; background-color: #888; 
                            border-radius: 50%; width: 16px; height: 16px; 
                            display: flex; align-items: center; justify-content: center;
                            font-weight: bold; border: 1px solid white; box-shadow: 0 1px 2px rgba(0,0,0,0.3);">✓</div>
                """
                icon = folium.DivIcon(html=checkmark_html)

                folium.Marker(
                    location=[poi["lat"], poi["lon"]],
                    popup=folium.Popup(popup_text, max_width=250),
                    tooltip=f"✅ Visited: {poi.get('name', 'Unknown')}",
                    icon=icon,
                ).add_to(m)
            except Exception as e:
                logger.debug(
                    f"Failed to add past visit marker for {poi.get('name', 'Unknown')}: {e}"
                )
                continue

        # Add recommendation markers (colored with rank numbers)
        # Add in REVERSE order so rank 1 appears on top (last added = highest z-index)
        # When we enumerate(reversed(pois)), we iterate: worst→...→best
        # This means rank 1 (best) is added LAST, so it appears on top ✓
        logger.info(
            f"Adding {len(pois)} recommendation markers"
        )
        logger.debug(
            f"First POI in list (best/rank1): {pois[0].get('name', 'unknown')}"
        )
        logger.debug(
            f"Last POI in list (worst/rank{len(pois)}): {pois[-1].get('name', 'unknown')}"
        )

        for rank_in_reversed, poi in enumerate(reversed(pois), 1):
            try:
                # Convert reversed enumeration index to actual rank
                # rank_in_reversed goes from 1 to len(pois)
                # actual_rank = len(pois) - rank_in_reversed + 1 converts it back to 1 to len(pois)
                actual_rank = len(pois) - rank_in_reversed + 1
                hex_color = hex_colors[(actual_rank - 1) % len(hex_colors)]

                # Build popup with info
                popup_text = f"<b>{poi.get('name', 'Unknown')}</b><br>"
                popup_text += f"🎯 <i>Rank #{actual_rank}</i><br>"
                if poi.get("category"):
                    popup_text += f"{poi['category']}<br>"
                popup_text += (
                    f"⭐ {poi.get('rating', 0)} ({poi.get('review_count', 0)} reviews)"
                )

                # Create numbered badge using HTML (colored circles with rank number)
                rank_html = f"""
                <div style="font-size: 14px; color: white; background-color: {hex_color}; 
                            border-radius: 50%; width: 34px; height: 34px; 
                            display: flex; align-items: center; justify-content: center;
                            font-weight: bold; border: 2px solid white; box-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    {actual_rank}
                </div>
                """
                icon = folium.DivIcon(html=rank_html)

                folium.Marker(
                    location=[poi["lat"], poi["lon"]],
                    popup=folium.Popup(popup_text, max_width=250),
                    tooltip=f"Rank {actual_rank}: {poi.get('name', 'Unknown')}",
                    icon=icon,
                ).add_to(m)

            except Exception as e:
                logger.debug(
                    f"Marker for rank {actual_rank} ({poi.get('name', 'Unknown')}) failed: {e}"
                )
                continue

        # Fit bounds to recommendation points so all recommendation markers are visible.
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        if len(lats) > 1 and len(lons) > 1:
            lat_pad = max(0.0025, (max_lat - min_lat) * 0.15)
            lon_pad = max(0.0025, (max_lon - min_lon) * 0.15)
            m.fit_bounds(
                [
                    [min_lat - lat_pad, min_lon - lon_pad],
                    [max_lat + lat_pad, max_lon + lon_pad],
                ]
            )
        else:
            m.location = [center_lat, center_lon]
            m.zoom_start = 14

        logger.debug(
            f"Map created with {len(pois)} recommendations and {len(past_visit_pois)} past visits"
        )
        return m

    except Exception as e:
        logger.error(f"Failed to build map: {e}", exc_info=True)
        return None


def _create_placeholder_image(width: int = 300, height: int = 300) -> Image.Image:
    """Create a placeholder image when no photo is available."""
    img = Image.new("RGB", (width, height), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)

    # Draw centered text
    text = "📷 No Photo"
    text_bbox = draw.textbbox((0, 0), text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=(100, 100, 100))
    return img


def _crop_image_to_aspect_ratio(
    image_url: str, target_ratio: float = 1.0, max_width: int = 300
) -> Image.Image | None:
    """Placeholder function - not used, Streamlit handles image display."""
    return None


def _crop_image_to_square(image_path: str, size: int = 280) -> Optional[bytes]:
    """Crop image to 1:1 square and return as bytes, or None on error."""
    try:
        from PIL import Image

        img = Image.open(image_path)
        # Crop to square (crop from center)
        if img.width != img.height:
            min_dim = min(img.width, img.height)
            left = (img.width - min_dim) // 2
            top = (img.height - min_dim) // 2
            img = img.crop((left, top, left + min_dim, top + min_dim))
        # Resize to target size
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        # Convert to bytes
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return buffer.getvalue()
    except Exception as e:
        logger.debug(f"Failed to crop image: {e}")
        return None


def _image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string for HTML embedding."""
    try:
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        logger.debug(f"Failed to convert image to base64: {e}")
        return ""


def draw_poi_card(
    poi: Dict,
    recommendation: Dict,
    show_scores: bool = False,
    card_width_px: int = 300,
    photo_height_px: int = 87,
):
    """
    Draw a single POI recommendation card with photo - fixed height with scrollable content.

    Card layout (FIXED HEIGHT = 650px):
    - Header: Name (2 lines max) + position delta (~50px, flex-shrink: 0)
    - Photo: Fixed height photo_height_px (~174px, flex-shrink: 0)
    - Content: Scrollable area filling remaining space (flex: 1, overflow-y: auto)

    This ensures all cards in a row have identical heights with scrollable overflow content.
    Names naturally wrap to 2 lines without padding/truncation.

    Skips invalid POIs silently (already filtered upstream).
    """
    # Skip empty POI dicts (return silently, not an error)
    if not poi or "name" not in poi or not poi.get("name"):
        logger.debug(f"Skipping empty/invalid POI: {poi}")
        return

    try:
        # Fixed card dimensions proportional to photo height (20% smaller, but photo less tall for more text)
        # For a compact layout, use: header + photo + flexible content + footer
        header_height_px = 32
        footer_height_px = 26  # Fixed footer for Yelp button
        content_min_height_px = 225  # Adjusted for footer + more text space
        photo_height_px_scaled = int(
            photo_height_px * 0.6
        )  # 40% smaller height (less tall, not narrower)
        card_height_px = (
            header_height_px
            + photo_height_px_scaled
            + content_min_height_px
            + footer_height_px
        )

        # Inject CSS for card sizing (one-time per page)
        st.markdown(
            f"""
        <style>
        /* POI Card: Fixed height with scrollable content */
        .poi-card-wrapper {{
            height: {card_height_px}px;
            display: flex;
            flex-direction: column;
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            overflow: hidden;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .poi-card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 10px 12px 6px 12px;
            flex-shrink: 0;
            gap: 8px;
            border-bottom: 1px solid #f0f0f0;
        }}
        
        .poi-card-name {{
            flex: 1;
            font-weight: 600;
            font-size: 16px;
            line-height: 1.3;
            word-wrap: break-word;
            overflow-wrap: break-word;
            max-height: 52px;
            overflow: hidden;
        }}
        
        .poi-card-rank {{
            flex-shrink: 0;
            font-size: 18px;
            font-weight: bold;
            white-space: nowrap;
            padding-top: 2px;
        }}
        
        .poi-card-photo {{
            width: 100%;
            height: {photo_height_px_scaled}px;
            flex-shrink: 0;
            background-color: #e8e8e8;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .poi-card-photo img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }}
        
        .poi-card-content {{
            flex: 1;
            overflow-y: auto;
            padding: 10px 12px;
            font-size: 13px;
            min-height: 0;
        }}
        
        .poi-card-content > * {{
            margin: 4px 0;
        }}
        
        .poi-card-footer {{
            flex-shrink: 0;
            padding: 8px 12px;
            border-top: 1px solid #f0f0f0;
            background: white;
        }}
        
        .poi-card-footer {{
            display: flex;
            justify-content: flex-end;
        }}
        
        .poi-card-footer a {{
            display: inline-block;
            padding: 6px 12px;
            background: #f5f5f5;
            color: #0066cc;
            text-decoration: none;
            font-size: 13px;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            transition: all 0.2s;
        }}
        
        .poi-card-footer a:hover {{
            background: #efefef;
            border-color: #0066cc;
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )

        # Build photo HTML (cloud first, fallback to local/placeholder)
        photo_html = ""
        photo_loaded = False

        if poi.get("primary_photo"):
            try:
                photo_path = poi["primary_photo"]
                # Try cloud-hosted image first (HTTP/HTTPS URLs)
                if photo_path.startswith(("http://", "https://")):
                    # External image - embed directly
                    photo_html = f'<img src="{photo_path}" alt="photo"/>'
                    photo_loaded = True
                else:
                    # Local file - convert to base64
                    img_bytes = _crop_image_to_square(photo_path, size=card_width_px)
                    if img_bytes:
                        b64_image = _image_to_base64(img_bytes)
                        photo_html = f'<img src="data:image/jpeg;base64,{b64_image}" alt="photo"/>'
                        photo_loaded = True
            except Exception as e:
                logger.debug(f"Failed to load photo {poi.get('primary_photo')}: {e}")

        if not photo_loaded:
            # Show placeholder
            try:
                placeholder = _create_placeholder_image(
                    card_width_px, photo_height_px_scaled
                )
                buffer = BytesIO()
                placeholder.save(buffer, format="PNG")
                b64_placeholder = base64.b64encode(buffer.getvalue()).decode("utf-8")
                photo_html = f'<img src="data:image/png;base64,{b64_placeholder}" alt="placeholder"/>'
            except Exception as e:
                logger.debug(f"Failed to create placeholder: {e}")
                photo_html = '<div style="font-size: 48px; text-align: center; line-height: 100%;">📷</div>'

        # Build rank display (always visible in card header)
        rank = recommendation.get("rank_after", 0)
        rank_html = f'<span style="font-size: 18px; font-weight: bold; color: #1f77b4;">#{rank + 1}</span>'

        # Build delta display for footer (if steering was applied)
        delta_footer_html = ""
        if recommendation.get("show_delta"):
            arrow = recommendation.get("arrow", "→")
            arrow_value = recommendation.get("arrow_value", 0)
            arrow_color = recommendation.get("arrow_color", "gray")

            if arrow_color == "green":
                delta_html = f'<span style="color:green; font-weight: bold;">{arrow}{arrow_value}</span>'
            elif arrow_color == "red":
                delta_html = f'<span style="color:red; font-weight: bold;">{arrow}{arrow_value}</span>'
            else:
                delta_html = f'<span style="color:gray;">{arrow}</span>'
            delta_footer_html = f"{delta_html}  positions changed  "

        # Build content section (rating, category, why, score, link)
        content_parts = []

        # Rating + reviews with visual stars
        rating = poi.get("rating", 0.0)
        reviews = poi.get("review_count", 0)
        # Create filled and empty stars (no half stars for clarity)
        filled_stars = int(round(rating))  # Round to nearest whole star
        empty_stars = max(0, 5 - filled_stars)
        star_display = "★" * filled_stars + "☆" * empty_stars
        content_parts.append(
            f"<div>{star_display} {rating:.1f} • {reviews:,} reviews</div>"
        )

        # Category
        category = poi.get("category", "")
        if category:
            content_parts.append(f"<div>📂 {category}</div>")

        # Why (contributing neurons) - show top 3 each on own line with contribution score
        if recommendation.get("contributing_neurons"):
            features = recommendation["contributing_neurons"][:3]  # Top 3
            content_parts.append(
                f"<div><b>Why This Recommendation:</b></div>"
                f"<div style='font-size: 11px; color: #666; margin-bottom: 4px;'>"
                f"Top latent features that contributed to this recommendation:</div>"
            )
            for feat in features:
                neuron_idx = feat.get("neuron_idx")
                feature_label = feat.get("label") or f"Feature {neuron_idx}"
                contribution = feat.get("contribution", 0)
                # Format each on own line: "  #42 Italian: 0.85"
                formatted = f"  #{neuron_idx} {feature_label}"
                if contribution:
                    formatted += f": {contribution:.2f}"
                content_parts.append(
                    f"<div style='font-size: 12px; margin: 2px 0;'>{formatted}</div>"
                )

        # Score with info
        if show_scores and recommendation.get("score"):
            content_parts.append(
                f"<div><b>Recommendation Score</b></div>"
                f"<div style='font-size: 11px; color: #666; margin-bottom: 4px;'>"
                f"Predicted relevance score (0-1 scale, higher is better):</div>"
                f"<div style='font-size: 14px; color: #1f77b4; font-weight: bold;'>"
                f"{recommendation['score']:.3f}</div>"
            )

        content_html = "".join(content_parts)

        # Yelp link (for footer, separate from scrollable content) with position delta
        url = poi.get("url", "")
        footer_html = ""
        if url:
            footer_html = f'{delta_footer_html}   <a href="{url}" target="_blank">View on Yelp →</a>'

        # Build complete card as HTML
        business_name = poi.get("name", "Unknown")
        card_html = f"""
        <div class="poi-card-wrapper">
            <div class="poi-card-header">
                <div class="poi-card-name">{business_name}</div>
                <div class="poi-card-rank">{rank_html}</div>
            </div>
            <div class="poi-card-photo">{photo_html}</div>
            <div class="poi-card-content">{content_html}</div>
            <div class="poi-card-footer">{footer_html}</div>
        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)

    except Exception as e:
        logger.debug(f"POI card error for {poi.get('name', 'Unknown')}: {e}")


def get_feature_color(index: int) -> str:
    """Map feature index to Folium color."""
    colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "darkblue",
        "darkgreen",
        "cadetblue",
        "darkpurple",
    ]
    return colors[index % len(colors)]
