"""Live Demo page — Interactive steering (main interactive page)."""

import logging
from typing import List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

try:
    import folium
    from streamlit_folium import st_folium

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

logger = logging.getLogger(__name__)


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

    # Debug section (collapsible)
    with st.expander("🐛 Debug Info", expanded=False):
        debug_cols = st.columns(3)
        with debug_cols[0]:
            st.write(f"**Inference loaded:** {inference is not None}")
            st.write(f"**n_items:** {inference.n_items if inference else 'N/A'}")
        with debug_cols[1]:
            st.write(
                f"**Current user ID:** {st.session_state.get('current_user_id', 'None')}"
            )
            st.write(
                f"**Has recommendations:** {bool(st.session_state.get('current_recommendations'))}"
            )
        with debug_cols[2]:
            st.write(
                f"**Encoded users:** {list(inference.user_latents.keys()) if inference else 'N/A'}"
            )

    # =====================================================================
    # SIDEBAR: Controls
    # =====================================================================
    with st.sidebar:
        st.header("🎛️ Controls")

        # Initialize sidebar state
        if "sidebar_expanded" not in st.session_state:
            st.session_state.sidebar_expanded = True

        # User selection
        st.subheader("Select User")
        test_users = data.get_test_users(limit=50)

        if not test_users:
            st.error("No test users available")
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
        show_history = st.checkbox("Show past visits", value=False)
        show_scores = st.checkbox("Show scores", value=False)

        st.divider()

        # Output parameters
        st.subheader("Output Parameters")

        recs_per_row = st.slider("Cards per row", 1, 10, 5)
        num_features = st.slider("Features to display", 5, 64, 10)
        num_recommendations = st.slider("Recommendations", 5, 50, 20)

        st.divider()

        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset"):
                st.session_state.steering_modified = False
                st.session_state.current_recommendations = []
                st.rerun()

        with col2:
            if st.button("🏠 Home"):
                st.switch_page("pages:Home")

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
                    # Encode user from interaction history
                    poi_indices = data.get_user_interactions(selected_user)
                    logger.debug(
                        f"Retrieved {len(poi_indices)} POI indices for user {selected_user}"
                    )

                    if not poi_indices:
                        st.warning(
                            f"No interaction history found for user {selected_user}"
                        )
                        return

                    # Validate inference service is properly initialized
                    if inference.n_items is None:
                        st.error(
                            "❌ Inference service not properly initialized: n_items is None"
                        )
                        logger.error(
                            f"Inference service n_items is None! This indicates model loading failed."
                        )
                        return

                    logger.debug(
                        f"Creating CSR matrix with shape (1, {inference.n_items})"
                    )

                    # Create sparse CSR matrix from POI indices (1 row, n_items columns)
                    from scipy.sparse import csr_matrix
                    import numpy as np

                    # Validate POI indices are within bounds
                    max_poi_idx = max(poi_indices) if poi_indices else 0
                    if max_poi_idx >= inference.n_items:
                        st.error(
                            f"❌ POI index {max_poi_idx} exceeds n_items={inference.n_items}"
                        )
                        logger.error(
                            f"POI index out of bounds: {max_poi_idx} >= {inference.n_items}"
                        )
                        return

                    row = np.zeros(len(poi_indices), dtype=int)  # All row 0
                    col = np.array(poi_indices, dtype=int)  # POI indices as columns
                    data_vals = np.ones(len(poi_indices), dtype=np.float32)

                    logger.debug(
                        f"CSR matrix data: row={row[:5]}..., col={col[:5]}..., data_vals={data_vals[:5]}..."
                    )

                    user_interactions_csr = csr_matrix(
                        (data_vals, (row, col)), shape=(1, inference.n_items)
                    )
                    logger.debug(
                        f"Created CSR matrix: shape={user_interactions_csr.shape}, nnz={user_interactions_csr.nnz}"
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
                logger.debug(
                    f"CSR matrix type: {type(user_interactions_csr)}, shape: {user_interactions_csr.shape}"
                )
                inference.encode_user(selected_user, user_interactions_csr)
                logger.debug(f"User {selected_user} encoded successfully")
                st.session_state.current_user_id = selected_user

            except Exception as e:
                st.error(f"Failed to encode user: {e}")
                logger.exception("User encoding failed")
                return

        # ===================================================================
        # Section 1: Active Features
        # ===================================================================

        if show_latent:
            st.subheader("🧠 Your Active Features")

            try:
                # Get top activations
                user_z = inference.user_latents[selected_user]
                activations = inference.get_top_activations(user_z, k=num_features)

                if activations:
                    # Plot feature bars
                    plot_feature_activations(activations)
                else:
                    st.info("No active features found")

            except Exception as e:
                st.error(f"Failed to get activations: {e}")
                logger.exception("Activation retrieval failed")
                activations = []
        else:
            activations = []

        # ===================================================================
        # Section 2: Steering Sliders
        # ===================================================================

        st.subheader("🎚️ Adjust Your Preferences")

        if activations and len(activations) > 0:
            st.markdown(
                """
            Each slider affects how much that feature influences recommendations.
            - **Left (-1)**: Strongly avoid
            - **Center (0)**: No change
            - **Right (+2)**: Strongly prefer
            """
            )

            # Create sliders in columns
            steering_updates = {}

            # Show sliders for top features
            top_features = activations[: min(15, num_features)]

            # Create 3-column layout for sliders
            cols_per_row = 3

            for row_idx in range(0, len(top_features), cols_per_row):
                cols = st.columns(cols_per_row)

                for col_idx, col in enumerate(cols):
                    feature_idx = row_idx + col_idx

                    if feature_idx < len(top_features):
                        feature = top_features[feature_idx]
                        neuron_idx = feature["neuron_idx"]
                        label = feature["label"][:20]

                        with col:
                            slider_value = st.slider(
                                label,
                                min_value=-1.0,
                                max_value=2.0,
                                value=0.0,
                                step=0.1,
                                key=f"slider_{neuron_idx}_{selected_user}",
                            )

                            if slider_value != 0.0:
                                steering_updates[neuron_idx] = slider_value

            st.divider()

            # Generate recommendations with steering
            try:
                if steering_updates:
                    st.info(f"🎨 Steering applied: {len(steering_updates)} features")

                result = inference.steer_and_recommend(
                    selected_user, steering_updates, top_k=num_recommendations
                )

                st.session_state.current_recommendations = result["recommendations"]
                st.session_state.steering_modified = len(steering_updates) > 0

                # Display inference latency
                if "latency_ms" in result:
                    col_latency, _ = st.columns([1, 4])
                    with col_latency:
                        st.metric("⚡ Inference Time", f"{result['latency_ms']:.0f}ms")

            except Exception as e:
                st.error(f"Failed to generate recommendations: {e}")
                logger.exception("Recommendation generation failed")
                return

        else:
            st.info("No active features to steer")

        # ===================================================================
        # Section 3: Map Visualization
        # ===================================================================

        if st.session_state.get("current_recommendations"):
            st.subheader("📍 Recommended Locations")

            if HAS_FOLIUM:
                try:
                    map_html = build_folium_map(
                        st.session_state.current_recommendations, data
                    )
                    st_folium(map_html, width=None, height=500)
                except Exception as e:
                    st.warning(f"Map rendering failed: {e}")
                    logger.debug(f"Folium error: {e}")
            else:
                st.info("Install streamlit-folium for map visualization")

        # ===================================================================
        # Section 4: POI Cards
        # ===================================================================

        if st.session_state.get("current_recommendations"):
            st.subheader("🏆 Recommended for You")

            recommendations = st.session_state.current_recommendations

            # Display POI cards
            cols = st.columns(recs_per_row)

            for idx, reco in enumerate(recommendations):
                with cols[idx % recs_per_row]:
                    try:
                        poi_details = data.get_poi_details(reco["poi_idx"])
                        draw_poi_card(poi_details, reco, show_scores)
                    except Exception as e:
                        st.error(f"Failed to display POI: {e}")
                        logger.debug(f"POI card error: {e}")

        # ===================================================================
        # Section 5: User History (optional)
        # ===================================================================

        if show_history:
            st.subheader("📜 Your Past Visits")

            try:
                history = inference.get_user_history(selected_user)

                if history:
                    history_pois = [data.get_poi_details(idx) for idx in history]

                    with st.expander(
                        f"Show {len(history)} past visits", expanded=False
                    ):
                        hist_cols = st.columns(recs_per_row)

                        for idx, poi in enumerate(history_pois):
                            with hist_cols[idx % recs_per_row]:
                                draw_poi_card(poi, {}, show_scores=False)

                else:
                    st.info("No interaction history found")

            except Exception as e:
                st.warning(f"Could not load history: {e}")


# =============================================================================
# Helper Functions
# =============================================================================


def plot_feature_activations(activations: List[Dict]):
    """Plot horizontal bar chart of top feature activations."""

    labels = [a["label"] for a in activations]
    values = [a["activation"] for a in activations]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=values, colorscale="Viridis", showscale=True),
        )
    )

    fig.update_layout(
        margin=dict(l=200),
        height=300,
        showlegend=False,
        xaxis_title="Activation Magnitude",
    )

    st.plotly_chart(fig, use_container_width=True)


def build_folium_map(recommendations: List[Dict], data_service) -> folium.Map:
    """Build interactive Folium map with POI markers."""

    if not HAS_FOLIUM or not recommendations:
        return None

    try:
        # Get POI details for all recommendations
        pois = [data_service.get_poi_details(r["poi_idx"]) for r in recommendations]

        pois = [p for p in pois if p]  # Filter out invalid

        if not pois:
            return None

        # Calculate center
        lats = [p["lat"] for p in pois]
        lons = [p["lon"] for p in pois]

        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap"
        )

        # Add markers
        for i, poi in enumerate(pois):
            color = get_feature_color(i % 10)

            # Build popup with photo
            popup_text = f"<b>{poi['name']}</b><br>"
            popup_text += f"{poi['category']}<br>"
            popup_text += f"⭐ {poi['rating']} ({poi['review_count']} reviews)"

            if poi.get("primary_photo"):
                popup_text += f"<br><img src='{poi['primary_photo']}' width='200'>"

            folium.Marker(
                location=[poi["lat"], poi["lon"]],
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=poi["name"],
                icon=folium.Icon(color=color, icon="info-sign"),
            ).add_to(m)

        return m

    except Exception as e:
        logger.error(f"Failed to build map: {e}")
        return None


def draw_poi_card(poi: Dict, recommendation: Dict, show_scores: bool = False):
    """Draw a single POI recommendation card with photo."""

    try:
        # Display photo if available
        if poi.get("primary_photo"):
            try:
                st.image(poi["primary_photo"], use_column_width=True)
            except Exception as e:
                st.caption(f"📸 Photo unavailable")
        else:
            st.info("📸 No photos available")

        # POI info
        st.markdown(f"### {poi['name']}")

        # Rating + review count
        st.markdown(f"⭐ {poi['rating']:.1f} " f"({poi['review_count']:,} reviews)")

        # Category
        st.caption(poi["category"])

        # Photo count
        if poi.get("photo_count", 0) > 1:
            st.caption(f"📷 +{poi['photo_count']-1} photos on Yelp")

        # Recommendation explanation
        if recommendation.get("contributing_neurons"):
            features = recommendation["contributing_neurons"][:2]
            explanation = "**Why recommended**: "
            for feat in features:
                feature_label = feat.get("label", f'Feature {feat.get("idx")}')
                explanation += f"_{feature_label}_"
            st.caption(explanation)

        # Score (optional)
        if show_scores and recommendation.get("score"):
            st.metric("Score", f"{recommendation['score']:.3f}")

        # Yelp link
        st.markdown(f"[View on Yelp]({poi['url']})", unsafe_allow_html=False)

    except Exception as e:
        st.error(f"Error rendering POI: {e}")
        logger.debug(f"POI card error: {e}")


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
