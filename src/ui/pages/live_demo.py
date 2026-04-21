"""Live demo page — Interactive steering (main interactive page)."""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from src.ui.utils import info_section
from src.ui.utils.formatting import format_feature_id, format_features_list
from src.ui.components.concept_steering_panel import render_concept_steering_panel

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
                    "rank_after": r.get("rank_after"),
                    "score": round(
                        r.get("score", 0), 3
                    ),  # Round to avoid float precision issues
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


@st.fragment
def _render_active_features_section(
    selected_user: str,
    inference,
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
        "Higher activation means this feature is more relevant to their preferences.",
    )

    try:
        user_z = inference.user_latents[selected_user]
        # Get ALL activations (large number to avoid filtering side effects)
        all_activations = inference.get_top_activations(user_z, k=64)

        # Display only top num_features (read from session_state, doesn't trigger rerun)
        num_features = st.session_state.get("display_num_features", 9)
        if all_activations:
            plot_feature_activations(all_activations[:num_features])
        else:
            st.info("No active features found")

        # Return ALL activations for steering (not filtered by num_features)
        return all_activations

    except Exception as e:
        st.error(f"Failed to get activations: {e}")
        logger.exception("Activation retrieval failed")
        return []


@st.fragment
def render_steering_tabs(
    selected_user: str,
    inference,
    data,
    activations: List[Dict],
) -> None:
    """Render neuron steering and concept steering in separate tabs."""
    tab_neuron, tab_concept = st.tabs(["Neuron Steering", "Concept Steering"])

    with tab_neuron:
        _render_steering_and_recommendations(
            selected_user=selected_user,
            inference=inference,
            data=data,
            activations=activations,
        )

    with tab_concept:
        render_concept_steering_panel(
            inference_service=inference,
            config=getattr(inference, "config", {}),
            session_state=st.session_state,
        )


@st.fragment
def _render_steering_and_recommendations(
    selected_user: str,
    inference,
    data,
    activations: List[Dict],
) -> None:
    """Generate recommendations with interactive feature steering controls.

    Core steering experience: users adjust feature importance via sliders,
    recommendations regenerate only when steering changes (cached for performance).

    **Steering Mechanism**:
    - Sliders: -1 (suppress), 0 (no change), +2 (emphasize)
    - Alpha=0.3: 30% blend between baseline and steered recommendations
    - Changes regenerate recommendations; cached for other page changes

    **Performance**:
    - First steering change: ~1-3s (inference compute)
    - Cached: <100ms (reused recommendations)
    - Slider interactions: <100ms (no recompute)

    Args:
        selected_user: Yelp user ID
        inference: InferenceService with recommendations
        data: DataService for item validation
        activations: List of active features from _render_active_features_section
    """
    info_section(
        "🎚️ Adjust Your Preferences",
        "Use sliders to steer recommendations by adjusting feature activation. "
        "Left (-1) = Avoid, Center (0) = No change, Right (+2) = Strongly prefer",
    )

    if activations and len(activations) > 0:
        st.markdown(
            """
        Adjust feature importance to steer recommendations:
        - **Left (-1)**: Strongly avoid this feature
        - **Center (0)**: Use current feature importance
        - **Right (+2)**: Strongly prefer this feature
        """
        )

        steering_updates = {}

        try:
            current_activations = inference.get_user_steering(selected_user)
        except ValueError as e:
            logger.warning(f"Could not get current activations: {e}")
            current_activations = {}

        # Display sliders for top num_features only (read from session_state, doesn't trigger rerun)
        num_features_to_display = st.session_state.get("display_num_features", 9)
        top_features = activations[:num_features_to_display]
        cols_per_row = 3

        for row_idx in range(0, len(top_features), cols_per_row):
            cols = st.columns(cols_per_row)

            for col_idx, col in enumerate(cols):
                feature_idx = row_idx + col_idx

                if feature_idx < len(top_features):
                    feature = top_features[feature_idx]
                    neuron_idx = feature["neuron_idx"]
                    label = feature["label"]

                    current_val = current_activations.get(neuron_idx, 0.5)

                    with col:
                        formatted_label = format_feature_id(neuron_idx, label)

                        # Feature slider with learn more button
                        slider_col, learn_col = st.columns([4, 1])
                        with slider_col:
                            steering_value = st.slider(
                                f"{formatted_label}",
                                min_value=-1.0,
                                max_value=2.0,
                                value=0.0,
                                step=0.1,
                                key=f"slider_{neuron_idx}_{selected_user}",
                                label_visibility="collapsed",
                            )
                        with learn_col:
                            # Use st.switch_page with the stored page object and query params
                            if st.button(
                                "📚",
                                key=f"learn_{neuron_idx}_{selected_user}",
                                help="Learn more about this feature",
                            ):
                                # Store feature ID in session_state for interpretability page to read
                                st.session_state["selected_feature_id"] = neuron_idx
                                interpretability_page = st.session_state.get(
                                    "_interpretability_page"
                                )
                                if interpretability_page:
                                    st.switch_page(interpretability_page)

                        # Add back the label on its own line
                        st.caption(f"**{formatted_label}**")

                        if steering_value != 0.0:
                            try:
                                steered_activation = (
                                    inference.get_steered_neuron_activation(
                                        selected_user, neuron_idx, steering_value
                                    )
                                )
                                st.caption(
                                    f"📊 Current: {current_val:.2f} → After steering: {steered_activation:.2f}"
                                )
                                steering_updates[neuron_idx] = steering_value
                            except Exception as e:
                                logger.debug(
                                    f"Could not compute steered activation: {e}"
                                )
                                st.caption(
                                    f"📊 Current: {current_val:.2f} → Steered to: {steering_value:.2f}"
                                )
                                steering_updates[neuron_idx] = steering_value
                        else:
                            st.caption(f"📊 Current: {current_val:.2f}")

        st.divider()

        if steering_updates:
            try:
                st.subheader("📊 Features: Original vs After Steering")
                original_activations = inference.get_user_steering(selected_user)
                original_features = [
                    {
                        **feat,
                        "activation": original_activations.get(feat["neuron_idx"], 0.0),
                    }
                    for feat in top_features
                ]
                steered_activations = inference.get_steered_activations(
                    selected_user, steering_updates, k=len(top_features)
                )
                if steered_activations:
                    plot_feature_activations(
                        steered_activations, original_activations=original_features
                    )
            except Exception as e:
                logger.debug(f"Could not compute steered activations chart: {e}")

        # COMPUTE RECOMMENDATIONS - only on steering change or first load
        try:
            steering_cache_key = f"steering_{selected_user}"
            last_steering = st.session_state.get(steering_cache_key, {})
            steering_changed = steering_updates != last_steering

            reco_cache_key = f"recommendations_all_{selected_user}"

            if steering_changed or reco_cache_key not in st.session_state:
                logger.info(
                    "Computing recommendations (steering changed or first load)"
                )

                if selected_user not in inference.baseline_recommendations:
                    logger.debug(f"Computing baseline for {selected_user}")
                    inference.get_baseline_recommendations(selected_user, 50)

                steering_config = None
                if steering_updates:
                    st.info(f"🎨 Steering applied: {len(steering_updates)} features")
                    steering_config = {
                        "type": "neuron",
                        "neuron_values": steering_updates,
                        "alpha": 0.3,
                    }

                logger.debug(f"Fetching recommendations for {selected_user}")

                # Get valid item IDs from data service (filtered by current state)
                valid_item_ids = data.get_valid_item_ids()
                logger.debug(
                    f"State filtering: {len(valid_item_ids)} valid items available"
                )

                # Request top_k recommendations with state filter applied
                # Inference service handles filtering internally
                recommendations_with_delta = inference.get_recommendations_with_delta(
                    selected_user,
                    steering_config=steering_config,
                    top_k=20,
                    valid_item_ids=valid_item_ids,
                )
                logger.debug(
                    f"Inference returned {len(recommendations_with_delta)} recommendations"
                )

                # OPTIMIZATION: Batch lookup for POI details (single query instead of N queries)
                # Extract all POI indices first
                poi_indices = [
                    r.get("item_id") or r.get("poi_idx")
                    for r in recommendations_with_delta
                ]
                # Batch lookup (returns dict of {poi_idx: details})
                poi_details_map = data.get_poi_details_batch(poi_indices)

                # Cache the POI details map to avoid re-fetching in card renderer
                st.session_state[f"poi_details_map_{selected_user}"] = poi_details_map

                # Filter recommendations to only valid POIs (those with resolved details)
                valid_recommendations = []
                for reco in recommendations_with_delta:
                    poi_idx = reco.get("item_id") or reco.get("poi_idx")
                    if poi_idx in poi_details_map:
                        valid_recommendations.append(reco)

                # Cache ALL valid recommendations
                st.session_state[reco_cache_key] = valid_recommendations
                st.session_state[steering_cache_key] = steering_updates

                filtered_count = len(recommendations_with_delta) - len(
                    valid_recommendations
                )
                logger.info(
                    f"Filtered recommendations: {len(recommendations_with_delta)} total → "
                    f"{filtered_count} invalid → {len(valid_recommendations)} valid (cached)"
                )
            else:
                logger.debug("Using cached recommendations (steering unchanged)")

            # Update session state with cached recommendations
            reco_cache = st.session_state.get(reco_cache_key, [])
            st.session_state.current_recommendations = reco_cache
            st.session_state.steering_modified = len(steering_updates) > 0

        except Exception as e:
            st.error(f"Failed to generate recommendations: {e}")
            logger.exception("Recommendation generation failed")
            return

    else:
        st.info("No active features to steer")


@st.fragment
def _render_poi_cards_section(
    data,
    available_width: int,
    card_width_px: int,
    show_scores: bool,
    photo_height_px: int,
    filtered_recommendations: list,
    poi_details_map: dict = None,
) -> None:
    """Isolated POI cards display fragment.

    Reruns only when recommendations or display settings change.
    Does not trigger rerun when steering or other factors change.

    Args:
        poi_details_map: Pre-fetched POI details dict (optional, for performance).
                        If provided, uses cached details instead of fetching.
    """
    if filtered_recommendations:
        st.subheader("🏆 Recommended for You")

        responsive_cards_per_row = max(1, available_width // card_width_px)
        cols = st.columns(responsive_cards_per_row)

        for display_idx, reco in enumerate(filtered_recommendations):
            poi_idx = reco.get("item_id") or reco.get("poi_idx")

            # Use cached details if available (already fetched in batch)
            # Otherwise fetch individually (fallback for edge cases)
            if poi_details_map and poi_idx in poi_details_map:
                poi_details = poi_details_map[poi_idx]
            else:
                poi_details = data.get_poi_details(poi_idx)

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
    if st.session_state.get("current_recommendations"):
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
                    # Stable key for st_folium - changes only when recommendations count changes
                    # (not on every rerun, which would rebuild map)
                    poi_count = len(recommendations)
                    st_folium(
                        map_obj,
                        width=None,
                        height=500,
                        key=f"folium_{selected_user}_{poi_count}",
                    )
            except Exception as e:
                logger.exception("Map rendering failed")
                st.error("❌ Map rendering failed. Check logs for details.")
        else:
            st.info(
                "📦 Install streamlit-folium for map visualization: `pip install streamlit-folium`"
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
        # Use conservative 800px width (accounts for sidebar variability)
        available_width = 800
        recs_per_row = st.slider(
            "Cards per row", min_value=1, max_value=5, value=3, step=1
        )

        # Calculate card width based on available width and cards per row
        card_width_px = available_width // recs_per_row

        # Calculate photo dimensions based on card width (maintain aspect ratio)
        photo_height_px = int(card_width_px * 0.733)

        st.caption(
            f"📐 {recs_per_row} cards | Width: {card_width_px}px | Photo: {card_width_px}×{photo_height_px}px"
        )

        # Get actual max features from inference service hidden dimension
        max_features = inference.sae.hidden_dim if inference and inference.sae else 64

        num_features = st.slider(
            "Features to display",
            min_value=5,
            max_value=max_features,
            value=min(9, max_features),
            step=1,
        )
        num_recommendations = st.slider("Recommendations", 5, 50, 20)

        # Store in session_state so fragments can read without rerunning
        st.session_state.display_num_features = num_features
        st.session_state.display_num_recommendations = num_recommendations

        st.divider()

        # Actions
        if st.button("🔄 Reset Steering", use_container_width=True):
            st.session_state.steering_modified = False
            st.session_state.current_recommendations = []
            st.session_state.baseline_recommendations = None
            st.rerun()

    # =====================================================================
    # MAIN AREA
    # =====================================================================
    if selected_user:
        # Get/create user encoding
        user_already_encoded = (
            st.session_state.get("current_user_id") == selected_user
            and st.session_state.get("current_recommendations")
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
                        import numpy as np
                        from scipy.sparse import csr_matrix

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
            )
        else:
            activations = []

        # ===================================================================
        # Section 2: Steering Sliders & Recommendations (isolated fragment)
        # ===================================================================
        render_steering_tabs(
            selected_user=selected_user,
            inference=inference,
            data=data,
            activations=activations,
        )

        # Now slice cached recommendations for display
        all_recommendations = st.session_state.get(
            "recommendations_all_{0}".format(selected_user), []
        )
        filtered_recommendations = all_recommendations[:num_recommendations]

        # Only update session_state if recommendations actually changed
        # This prevents unnecessary fragment reruns when only display params changed
        old_filtered = st.session_state.get("filtered_recommendations_to_display", [])
        if len(old_filtered) != len(filtered_recommendations) or (
            filtered_recommendations
            and old_filtered
            and old_filtered[0].get("item_id")
            != filtered_recommendations[0].get("item_id")
        ):
            st.session_state[
                "filtered_recommendations_to_display"
            ] = filtered_recommendations

        # Section 3: Map Visualization (ALWAYS renders instantly with recommendations)
        # ===================================================================
        _render_map_section(
            selected_user=selected_user,
            data=data,
            inference=inference,
            filtered_recommendations=st.session_state.get(
                "filtered_recommendations_to_display", []
            ),
            show_past_visits=show_history,
        )

        # ===================================================================
        # Section 4: POI Cards (Recommendations)
        # ===================================================================
        _render_poi_cards_section(
            data=data,
            available_width=available_width,
            card_width_px=card_width_px,
            show_scores=show_scores,
            photo_height_px=photo_height_px,
            filtered_recommendations=st.session_state.get(
                "filtered_recommendations_to_display", []
            ),
            poi_details_map=st.session_state.get(
                f"poi_details_map_{selected_user}", {}
            ),
        )

        # ===================================================================
        # Section 5: User History Display (uses cached data from pre-load)
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
    activations: List[Dict], original_activations: List[Dict] = None
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

    st.plotly_chart(fig, use_container_width=True)


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

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
        )

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
            f"Adding {len(pois)} recommendation markers in REVERSE order (so rank 1 is added last = on top)"
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

        logger.info(
            f"✅ Map created with {len(pois)} recommendations and {len(past_visit_pois)} past visits"
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
        # Fixed card dimensions proportional to photo height
        # For a compact layout, use: header + photo + flexible content + footer
        header_height_px = 40
        footer_height_px = 30  # Fixed footer for Yelp button
        content_min_height_px = 231  # Adjusted for footer
        card_height_px = (
            header_height_px
            + photo_height_px
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
            height: {photo_height_px}px;
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
                placeholder = _create_placeholder_image(card_width_px, photo_height_px)
                buffer = BytesIO()
                placeholder.save(buffer, format="PNG")
                b64_placeholder = base64.b64encode(buffer.getvalue()).decode("utf-8")
                photo_html = f'<img src="data:image/png;base64,{b64_placeholder}" alt="placeholder"/>'
            except Exception as e:
                logger.debug(f"Failed to create placeholder: {e}")
                photo_html = '<div style="font-size: 48px; text-align: center; line-height: 100%;">📷</div>'

        # Build rank/delta display
        rank_html = ""
        if recommendation.get("show_delta"):
            arrow = recommendation.get("arrow", "→")
            arrow_value = recommendation.get("arrow_value", 0)
            arrow_color = recommendation.get("arrow_color", "gray")

            if arrow_color == "green":
                rank_html = f'<span style="color:green;">{arrow}{arrow_value}</span>'
            elif arrow_color == "red":
                rank_html = f'<span style="color:red;">{arrow}{arrow_value}</span>'
            else:
                rank_html = f'<span style="color:gray;">#{recommendation.get("rank_after", 0) + 1}</span>'
        else:
            rank = recommendation.get("rank_after", 0)
            rank_html = f'<span style="color:blue;">#{rank + 1}</span>'

        # Build content section (rating, category, why, score, link)
        content_parts = []

        # Rating + reviews
        rating = poi.get("rating", 0.0)
        reviews = poi.get("review_count", 0)
        content_parts.append(f"<div>⭐ {rating:.1f} • {reviews:,} reviews</div>")

        # Category
        category = poi.get("category", "")
        if category:
            content_parts.append(f"<div>📂 {category}</div>")

        # Why (contributing neurons) - make features clickable
        if recommendation.get("contributing_neurons"):
            features = recommendation["contributing_neurons"][:1]
            for feat in features:
                neuron_idx = feat.get("idx")
                feature_label = feat.get("label") or f"Feature {neuron_idx}"
                activation = feat.get("activation")

                if activation is not None:
                    formatted_feature = format_feature_id(
                        neuron_idx, feature_label, activation
                    )
                else:
                    formatted_feature = format_feature_id(neuron_idx, feature_label)

                # Display contributing feature (use steering button to learn more)
                clickable_feature = formatted_feature
                content_parts.append(f"<div>🧠 Why: {clickable_feature}</div>")

        # Score
        if show_scores and recommendation.get("score"):
            content_parts.append(f"<div>Score: {recommendation['score']:.3f}</div>")

        content_html = "".join(content_parts)

        # Yelp link (for footer, separate from scrollable content)
        url = poi.get("url", "")
        footer_html = ""
        if url:
            footer_html = f'<a href="{url}" target="_blank">View on Yelp →</a>'

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

        # (Removed separate button - features are now clickable as inline links in "Why" section)

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
