"""Interpretability page — Feature browser with labels and wordclouds."""

import logging

import streamlit as st

from src.ui.utils import info_section

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
        st.warning(
            "⚠️ Wordcloud service not available - labels will display without visualizations"
        )

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
        info_section(
            "📝 Label",
            "Human-readable interpretation of what this feature detects. "
            "Features are automatically labeled based on the categories and businesses that activate them.",
        )

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
                    info_section(
                        "📂 Top Categories",
                        "Business categories ranked by frequency (how many items have activated this feature). "
                        "Higher frequency = more businesses in that category activate this feature.",
                    )
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
        info_section(
            "☁️ Category Wordcloud",
            "Visual word cloud where larger words represent categories that activate this feature more frequently. "
            "This provides an intuitive at-a-glance view of the feature's category preferences.",
        )

        if wordcloud_service:
            try:
                # Generate wordcloud
                fig = wordcloud_service.generate_wordcloud_fig(
                    neuron_idx, figsize=(6, 4), width=600, height=400, colormap="tab20"
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
        info_section(
            "🔥 Top Activating Categories",
            "Categories ranked by average activation strength (σ values). "
            "Higher σ values indicate stronger activations. "
            "'n' shows how many times the category was activated for this feature.",
        )

        # Get categories sorted by activation strength (from wordcloud service)
        if wordcloud_service:
            try:
                top_categories = wordcloud_service.get_top_activating_categories(
                    neuron_idx, top_k=10
                )

                if top_categories:
                    # Display categories with activation strength bars
                    st.markdown("**Categories by average activation strength:**")

                    # Find max activation for normalization
                    max_activation = (
                        max([c["avg_activation"] for c in top_categories])
                        if top_categories
                        else 1.0
                    )

                    # Create a simple bar-like visualization using progress bars and text
                    for item in top_categories:
                        category = item["category"]
                        avg_activation = item["avg_activation"]
                        frequency = item["frequency"]

                        # Normalize for display (assuming activations are typically 0-1)
                        strength_pct = min(100, max(0, int(avg_activation * 100)))
                        normalized_strength = (
                            avg_activation / max_activation if max_activation > 0 else 0
                        )

                        # Multi-line display with activation strength and frequency
                        col_name, col_strength = st.columns([2, 1])

                        with col_name:
                            st.caption(f"**{category}**")

                        with col_strength:
                            st.caption(f"σ={avg_activation:.2f} (n={frequency})")

                        # Progress bar for visual strength
                        st.progress(
                            min(1.0, normalized_strength), text=f"{strength_pct}%"
                        )
                else:
                    st.info(
                        """
                    📊 **Category Data Unavailable**
                    
                    This feature would display business categories extracted during model training,
                    sorted by average activation strength.
                    Run the training pipeline to generate category metadata.
                    """
                    )
            except Exception as e:
                st.warning(f"Could not load activation categories: {e}")
        else:
            st.info("⚠️ Wordcloud service not available")

    st.divider()

    # Top activating businesses section
    if wordcloud_service:
        info_section(
            "🏢 Top Activating Businesses",
            "Specific businesses/places that most strongly activate this feature. "
            "σ values show activation strength. This data arrives once the notebook exports business activation data.",
        )
        try:
            top_items = wordcloud_service.get_top_items(neuron_idx, top_k=5)
            if top_items:
                st.markdown("**Places that most strongly activate this feature:**")
                for i, item in enumerate(top_items, 1):
                    # Display business name/info if available
                    business_name = item.get(
                        "name", item.get("business_id", f"Business {i}")
                    )
                    activation = item.get("activation", item.get("avg_activation", 0))

                    col_rank, col_name, col_stats = st.columns([0.5, 2, 1])
                    with col_rank:
                        st.caption(f"#{i}")
                    with col_name:
                        st.caption(f"**{business_name}**")
                    with col_stats:
                        st.caption(f"σ={activation:.2f}")
            else:
                st.info(
                    """
                🏢 **Top Businesses Not Yet Available**
                
                This section will show top-activating businesses once the notebook exports
                this data (requires re-running notebook 03 with updated export code).
                """
                )
        except Exception as e:
            st.caption(f"⚠️ Could not load top businesses: {e}")

    st.divider()

    # Related Features (co-activation section)
    coactivation_service = st.session_state.get("coactivation")

    if coactivation_service:
        info_section(
            "🔗 Related Features",
            "Features that are frequently or rarely co-activated with this feature (based on correlation analysis). "
            "Frequently co-activated features often detect complementary patterns in the data.",
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Frequently Co-Activated With**")
            highly_coactivated = coactivation_service.get_highly_coactivated(neuron_idx)
            if highly_coactivated:
                for item in highly_coactivated:
                    st.caption(f"• {item['label']} (Feature {item['neuron_id']})")
            else:
                st.caption("*No positive co-activation data found*")

        with col2:
            st.markdown("**Rarely Co-Activated With**")
            rarely_coactivated = coactivation_service.get_rarely_coactivated(neuron_idx)
            if rarely_coactivated:
                for item in rarely_coactivated:
                    st.caption(f"• {item['label']} (Feature {item['neuron_id']})")
            else:
                st.caption("*No negative correlations found*")

        st.divider()

    # Feature comparison
    info_section(
        "🔀 Compare Features",
        "Side-by-side comparison of two features' labels and categories. "
        "Use this to understand how different features differ in their semantic meanings.",
    )

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
        label_1 = labels_service.get_label(compare_idx_1)
        label_2 = labels_service.get_label(compare_idx_2)

        # Get categories if available
        categories_1 = []
        categories_2 = []
        if wordcloud_service:
            try:
                categories_1 = wordcloud_service.get_categories_for_neuron(
                    compare_idx_1
                )
                categories_2 = wordcloud_service.get_categories_for_neuron(
                    compare_idx_2
                )
            except Exception as e:
                logger.debug(f"Could not get categories: {e}")

        st.markdown(
            f"""
        ### Comparison
        
        | | Feature {compare_idx_1} | Feature {compare_idx_2} |
        |---|---|---|
        | **Label** | {label_1} | {label_2} |
        | **Categories** | {', '.join(categories_1) if categories_1 else 'N/A'} | {', '.join(categories_2) if categories_2 else 'N/A'} |
        """
        )
