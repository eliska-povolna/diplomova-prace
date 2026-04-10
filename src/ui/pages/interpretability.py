"""Interpretability page — Feature browser with labels and wordclouds."""

import streamlit as st
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def show():
    """Display interpretability page with neuron labels and wordclouds."""
    
    st.title("🔍 Feature Interpretability")

    st.markdown(
        """
    Browse all learned features and understand what each neuron represents
    through human-readable labels and visual wordclouds of activating business categories.
    """
    )

    # Initialize services
    labels_service = st.session_state.get("labels")
    wordcloud_service = st.session_state.get("wordcloud")
    
    if not labels_service:
        st.error("❌ Labeling service not initialized")
        return
    
    if not wordcloud_service:
        st.warning("⚠️ Wordcloud service not available - labels will display without visualizations")

    # Feature selector
    col1, col2 = st.columns([2, 1])

    with col1:
        # Get maximum neuron count
        max_neuron = 63  # Default for SAE k=32 or similar
        
        neuron_idx = st.slider(
            "Select Feature",
            min_value=0,
            max_value=max_neuron,
            value=0,
            step=1,
            key="neuron_slider",
        )

    with col2:
        if st.button("🎛️ Use in Live Demo"):
            st.session_state.selected_neuron = neuron_idx
            st.switch_page("🎛️ Live Demo")

    st.divider()

    # Feature details - Two column layout
    col_left, col_right = st.columns([1, 1.5])

    with col_left:
        # Label section
        st.subheader("📝 Label")
        
        # Get label from service
        try:
            label = labels_service.get_label(neuron_idx)
        except Exception as e:
            label = f"Feature {neuron_idx}"
            logger.warning(f"Failed to get label: {e}")

        st.markdown(
            f"""
        ### {label}
        
        **Feature Index**: `{neuron_idx}`  
        **Type**: Interpretable dimension (SAE neuron)
        """
        )

        # Copy-friendly code block
        st.code(label, language="text")
        
        # Get categories if available
        if wordcloud_service:
            try:
                categories = wordcloud_service.get_categories_for_neuron(neuron_idx)
                if categories:
                    st.subheader("📂 Top Categories")
                    # Display as pills
                    for i in range(0, len(categories), 2):
                        cols = st.columns(2)
                        with cols[0]:
                            st.caption(f"• {categories[i]}")
                        if i + 1 < len(categories):
                            with cols[1]:
                                st.caption(f"• {categories[i+1]}")
            except Exception as e:
                logger.debug(f"Could not display categories: {e}")

    with col_right:
        st.subheader("☁️ Category Wordcloud")
        st.markdown("Frequencies of business categories that activate this feature.")
        
        if wordcloud_service:
            try:
                # Generate wordcloud
                fig = wordcloud_service.generate_wordcloud_fig(
                    neuron_idx,
                    figsize=(7, 4),
                    width=600,
                    height=400,
                    colormap='tab20'
                )
                
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("📊 No wordcloud data available for this feature")
            except Exception as e:
                st.warning(f"⚠️ Wordcloud generation failed: {str(e)[:80]}")
                logger.error(f"Wordcloud generation error: {e}")
        else:
            st.info("📊 Wordcloud service not available")

    with col_right:
        st.subheader("📍 Top Activating POIs")

        # Get top POIs for this neuron
        top_pois = labels_service.get_pois_for_neuron(neuron_idx, top_k=10) if labels_service else []

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
            st.dataframe(pois_df, use_container_width=True, hide_index=True)

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
