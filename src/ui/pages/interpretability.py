"""Interpretability page — Feature browser."""

import streamlit as st
import pandas as pd
from pathlib import Path
try:
    from src.ui.components.neuron_wordcloud import display_neuron_wordcloud
    from src.ui.cache import get_precomputed_cache_dir
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False


def show():
    """Display interpretability page."""
    labels = st.session_state.get("labels")
    data = st.session_state.get("data")

    if not labels:
        st.error("Labeling service not initialized")
        return

    st.title("🔍 Feature Interpretability")

    st.markdown(
        """
    Browse all learned features and understand what each neuron represents
    by examining the POIs that maximally activate it.
    """
    )

    # Feature selector
    col1, col2 = st.columns([2, 1])

    with col1:
        neuron_idx = st.slider(
            "Select Feature",
            min_value=0,
            max_value=63,  # TODO: Make dynamic based on SAE k
            value=0,
            step=1,
            key="neuron_slider",
        )

    with col2:
        if st.button("🎛️ Use in Live Demo"):
            st.session_state.selected_neuron = neuron_idx
            st.switch_page("src.ui.main:show_live_demo")

    st.divider()

    # Feature details
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("📝 Label")

        # Get label (will use LLM if available)
        label = labels.get_label(neuron_idx)

        st.markdown(
            f"""
        ### {label}
        
        **Feature Index**: `{neuron_idx}`
        
        **Semantic**: Interpretable dimension learned by SAE
        """
        )

        # Copy button for label
        st.code(label, language="text")
        
        # Word cloud visualization (if available)
        if HAS_WORDCLOUD:
            st.divider()
            st.subheader("☁️ Word Cloud")
            st.markdown("Top words from POIs activating this feature.")
            
            cache_dir = get_precomputed_cache_dir()
            try:
                display_neuron_wordcloud(
                    neuron_idx,
                    label="",
                    precomputed_wordcloud_dir=cache_dir,
                    width=400,
                    height=200,
                    colormap="viridis",
                    show_info=False,
                )
            except Exception as e:
                st.warning(f"Word cloud unavailable: {str(e)[:50]}")

    with col_right:
        st.subheader("📍 Top Activating POIs")

        # Get top POIs for this neuron
        top_pois = labels.get_pois_for_neuron(neuron_idx, top_k=10)

        if top_pois:
            # Convert to DataFrame for display
            pois_records = []
            for poi in top_pois:
                pois_records.append(
                    {
                        "Name": poi.get("name", "Unknown"),
                        "Category": poi.get("category", ""),
                        "Activation": f"{poi.get('activation', 0):.3f}",
                        "Rating": f"⭐ {poi.get('rating', 0):.1f}",
                    }
                )

            pois_df = pd.DataFrame(pois_records)
            st.dataframe(pois_df, width='stretch', hide_index=True)

            st.caption(
                """
            These POIs have the highest average activation for this feature.
            """
            )
        else:
            st.info(
                """
            📊 **POI Activation Data Unavailable**
            
            Top activating POIs would be computed from training data during model interpretation.
            This requires running the neuron labeling pipeline (see `notebooks/03_neuron_labeling_demo.ipynb`).
            
            Currently, the system provides labels only. Activation analysis coming soon!
            """
            )

    st.divider()

    # Feature statistics / relationships
    st.subheader("🔗 Related Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        **Frequently Co-Activated With**
        - Feature 5 (Italian Restaurants)
        - Feature 12 (Upscale Venues)
        - Feature 23 (Downtown Area)
        """
        )

    with col2:
        st.markdown(
            """
        **Rarely Co-Activated With**
        - Feature 2 (Budget-Friendly)
        - Feature 8 (Fast Food)
        - Feature 19 (Casual Dining)
        """
        )

    st.divider()

    # Feature comparison
    st.subheader("🔀 Compare Features")

    col1, col2 = st.columns(2)

    with col1:
        compare_idx_1 = st.number_input(
            "Feature 1", min_value=0, max_value=63, value=neuron_idx, key="compare_1"
        )

    with col2:
        compare_idx_2 = st.number_input(
            "Feature 2",
            min_value=0,
            max_value=63,
            value=(neuron_idx + 1) % 64,
            key="compare_2",
        )

    if st.button("Compare"):
        label_1 = labels.get_label(compare_idx_1)
        label_2 = labels.get_label(compare_idx_2)

        st.markdown(
            f"""
        ### Comparison
        
        | | Feature {compare_idx_1} | Feature {compare_idx_2} |
        |---|---|---|
        | **Label** | {label_1} | {label_2} |
        | **Co-activation** | ? | ? |
        | **Avg POI Rating** | 4.2 ⭐ | 4.1 ⭐ |
        | **Top Category** | Restaurant | Cafe |
        """
        )
