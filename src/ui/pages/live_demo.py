"""Live Demo page — Interactive steering (main interactive page)."""

import logging
import base64
from typing import List, Dict
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from PIL import Image, ImageDraw

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
                st.switch_page("🏠 Home")

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
                    map_obj = build_folium_map(
                        st.session_state.current_recommendations, data
                    )
                    if map_obj is None:
                        st.info("📍 No valid POI locations to display on map")
                        logger.warning("Map object returned None")
                    else:
                        st_folium(map_obj, width=None, height=500)
                except Exception as e:
                    st.error(f"❌ Map rendering failed: {e}")
                    logger.exception(f"Folium map error: {e}")
            else:
                st.info("📦 Install streamlit-folium for map visualization: `pip install streamlit-folium`")

        # ===================================================================
        # Section 4: POI Cards
        # ===================================================================

        if st.session_state.get("current_recommendations"):
            st.subheader("🏆 Recommended for You")

            recommendations = st.session_state.current_recommendations

            # Display POI cards
            cols = st.columns(recs_per_row)
            displayed_count = 0

            for idx, reco in enumerate(recommendations):
                poi_details = data.get_poi_details(reco["poi_idx"])
                
                # Skip empty/invalid POIs (already validated in get_poi_details)
                if not poi_details:
                    logger.debug(f"Skipping invalid POI at index {reco['poi_idx']}")
                    continue
                
                with cols[displayed_count % recs_per_row]:
                    try:
                        draw_poi_card(poi_details, reco, show_scores)
                        displayed_count += 1
                    except Exception as e:
                        logger.exception(f"POI card error for {poi_details.get('name', 'Unknown')}: {e}")
                        # Skip this POI and continue to next
                        continue

        # ===================================================================
        # Section 5: User History (optional)
        # ===================================================================

        if show_history:
            st.subheader("📜 Your Past Visits")

            try:
                history = inference.get_user_history(selected_user)

                if history:
                    history_pois = [data.get_poi_details(idx) for idx in history]
                    # Filter out empty/invalid POIs
                    history_pois = [p for p in history_pois if p]

                    if history_pois:
                        with st.expander(
                            f"Show {len(history_pois)} past visits", expanded=False
                        ):
                            hist_cols = st.columns(recs_per_row)
                            displayed_count = 0

                            for poi in history_pois:
                                try:
                                    with hist_cols[displayed_count % recs_per_row]:
                                        draw_poi_card(poi, {}, show_scores=False)
                                        displayed_count += 1
                                except Exception as e:
                                    logger.exception(f"History POI card error for {poi.get('name', 'Unknown')}: {e}")
                                    continue
                    else:
                        st.info("Your past visits had no valid location data")

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

    st.plotly_chart(fig, width='stretch')


def build_folium_map(recommendations: List[Dict], data_service) -> folium.Map | None:
    """Build interactive Folium map with POI markers."""

    if not HAS_FOLIUM:
        logger.warning("Folium not available for map rendering")
        return None
    
    if not recommendations:
        logger.warning("No recommendations to display on map")
        return None

    try:
        # Get POI details for all recommendations
        pois = [data_service.get_poi_details(r["poi_idx"]) for r in recommendations]
        pois = [p for p in pois if p and p.get("lat") and p.get("lon")]

        if not pois:
            logger.warning("No valid POI data with coordinates for map")
            return None

        # Calculate center
        lats = [p["lat"] for p in pois if p.get("lat")]
        lons = [p["lon"] for p in pois if p.get("lon")]
        
        if not lats or not lons:
            logger.warning("No valid coordinates for map center")
            return None

        center_lat = np.mean(lats)
        center_lon = np.mean(lons)

        logger.debug(f"Creating map centered at ({center_lat:.4f}, {center_lon:.4f}) with {len(pois)} POIs")

        # Create map
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=13, 
            tiles="OpenStreetMap"
        )

        # Add markers
        for i, poi in enumerate(pois):
            try:
                color = get_feature_color(i % 10)
                
                # Build popup with info
                popup_text = f"<b>{poi.get('name', 'Unknown')}</b><br>"
                if poi.get("category"):
                    popup_text += f"{poi['category']}<br>"
                popup_text += f"⭐ {poi.get('rating', 0)} ({poi.get('review_count', 0)} reviews)"

                folium.Marker(
                    location=[poi["lat"], poi["lon"]],
                    popup=folium.Popup(popup_text, max_width=250),
                    tooltip=poi.get("name", f"POI {i}"),
                    icon=folium.Icon(color=color, icon="info-sign"),
                ).add_to(m)
            except Exception as e:
                logger.debug(f"Failed to add marker for POI {i}: {e}")
                continue

        logger.info(f"✅ Map created with {len(pois)} markers")
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


def _crop_image_to_aspect_ratio(image_url: str, target_ratio: float = 1.0, max_width: int = 300) -> Image.Image | None:
    """Placeholder function - not used, Streamlit handles image display."""
    return None


def _crop_image_to_square(image_path: str, size: int = 280) -> bytes:
    """Crop image to 1:1 square and return as bytes."""
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
    import base64
    try:
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        logger.debug(f"Failed to convert image to base64: {e}")
        return ""


def draw_poi_card(poi: Dict, recommendation: Dict, show_scores: bool = False):
    """
    Draw a single POI recommendation card with photo - fixed pixel-based sizing.
    
    Card layout:
    - Photo: 280x280 px (CROPPED with object-fit: cover)
    - Content: Flexible height with scrolling if needed
    - Total card: 750px fixed height for consistent grid alignment
    
    Skips invalid POIs silently (already filtered upstream).
    """
    # Skip empty POI dicts (return silently, not an error)
    if not poi or "name" not in poi or not poi.get("name"):
        logger.debug(f"Skipping empty/invalid POI: {poi}")
        return
    
    try:
        # Inject CSS once per page for card sizing
        st.markdown("""
        <style>
        /* POI Card fixed sizing - creates uniform grid */
        .poi-card-wrapper {
            height: 750px;
            display: flex;
            flex-direction: column;
            border: 1px solid #d0d0d0;
            border-radius: 8px;
            overflow: hidden;
        }
        .poi-card-photo-container {
            width: 100%;
            height: 280px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #e8e8e8;
            overflow: hidden;
        }
        .poi-card-photo-container img {
            width: 280px;
            height: 280px;
            object-fit: cover;
            display: block;
        }
        .poi-card-content {
            flex: 1;
            overflow-y: auto;
            padding: 12px 16px;
            min-height: 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Use container with border
        with st.container(border=True):
            # PHOTO SECTION - Exactly 280x280 px with 1:1 cropping
            photo_loaded = False
            if poi.get("primary_photo"):
                try:
                    photo_path = poi["primary_photo"]
                    # Crop to square and convert to base64
                    img_bytes = _crop_image_to_square(photo_path, size=280)
                    if img_bytes:
                        b64_image = _image_to_base64(img_bytes)
                        st.markdown(
                            f'<div class="poi-card-photo-container"><img src="data:image/jpeg;base64,{b64_image}" alt="photo"/></div>',
                            unsafe_allow_html=True
                        )
                        photo_loaded = True
                except Exception as e:
                    logger.debug(f"Failed to load photo {poi.get('primary_photo')}: {e}")
            
            if not photo_loaded:
                # Show placeholder
                try:
                    placeholder = _create_placeholder_image(280, 280)
                    buffer = BytesIO()
                    placeholder.save(buffer, format="PNG")
                    b64_placeholder = base64.b64encode(buffer.getvalue()).decode("utf-8")
                    st.markdown(
                        f'<div class="poi-card-photo-container"><img src="data:image/png;base64,{b64_placeholder}" alt="placeholder"/></div>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    logger.debug(f"Failed to create placeholder: {e}")
                    st.markdown(
                        '<div class="poi-card-photo-container" style="font-size: 48px;">📷</div>',
                        unsafe_allow_html=True
                    )
            
            # CONTENT SECTION - Scrollable
            st.divider()
            
            # NAME - No truncation
            st.markdown(f"**{poi.get('name', 'Unknown')}**")
            
            # RATING + REVIEWS
            rating = poi.get("rating", 0.0)
            reviews = poi.get("review_count", 0)
            st.caption(f"⭐ {rating:.1f} • {reviews:,} reviews")

            # CATEGORY
            category = poi.get("category", "")
            if category:
                st.caption(f"📂 {category}")

            # PHOTO COUNT
            if poi.get("photo_count", 0) > 1:
                st.caption(f"📷 +{poi['photo_count']-1} more")

            # RECOMMENDATION
            if recommendation.get("contributing_neurons"):
                features = recommendation["contributing_neurons"][:1]
                for feat in features:
                    feature_label = feat.get("label", f"Feature {feat.get('idx')}")
                    st.caption(f"🧠 _{feature_label}_")

            # SCORE
            if show_scores and recommendation.get("score"):
                st.caption(f"Score: {recommendation['score']:.3f}")

            # YELP LINK
            url = poi.get("url", "")
            if url:
                st.markdown(f"[View on Yelp]({url})", unsafe_allow_html=False)

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
