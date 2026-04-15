"""Home page — Welcome & quick stats."""

import streamlit as st


def show():
    """Display home page."""
    inference = st.session_state.get("inference")
    data = st.session_state.get("data")

    if not inference or not data:
        st.error("Services not initialized")
        return

    # Debug section at top to help diagnose caching issues
    with st.expander("🔧 Debug Info", expanded=False):
        debug_cols = st.columns([3, 1])
        with debug_cols[0]:
            st.write(f"**Model n_items**: {inference.n_items}")
            st.write(f"**Data POIs**: {data.num_pois}")
            item2index_size = getattr(data, 'item2index', None)
            if item2index_size:
                st.write(f"**item2index mapping size**: {len(item2index_size)}")
            else:
                st.write(f"**item2index**: Not loaded")
            test_users = data.get_test_users(limit=50)
            st.write(f"**Test users loaded**: {len(test_users)}")
            if test_users:
                st.write(f"  First user: {test_users[0]['id']}")
            if not test_users:
                st.warning("⚠️ No test users found! Check cache at data/ui_cache/")
        with debug_cols[1]:
            if st.button("🔄 Clear Cache", key="clear_cache_btn"):
                # Clear service cache to force reload on next run
                st.cache_resource.clear()
                st.success("Cache cleared! Reload page in browser.")

    # Title
    st.title("🗺️ Interpretable POI Recommender")
    st.write(
        """
    Discover places tailored to your preferences using **sparse, interpretable neural features**.
    
    This demo showcases a recommendation system that combines:
    - **ELSA**: Collaborative filtering autoencoder for dense embeddings
    - **Sparse Autoencoder (SAE)**: Decomposition into interpretable features
    - **Interactive Steering**: Adjust your preferences in real-time
    """
    )

    # Quick stats
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🧠 Active Features",
            getattr(inference.sae, "k", 64),
            delta=None,
            delta_color="off",
        )

    with col2:
        st.metric("📍 POIs", f"{data.num_pois:,}", delta=None, delta_color="off")

    with col3:
        st.metric(
            "👥 Test Users",
            len(data.get_test_users(limit=50)),
            delta="available",
            delta_color="off",
        )

    with col4:
        st.metric("🏙️ Dataset", "Yelp Academic", delta="California", delta_color="off")

    # How it works
    st.subheader("⚙️ How It Works")

    with st.expander("1️⃣ ELSA — Collaborative Filtering Autoencoder", expanded=False):
        st.markdown(
            """
        **ELSA** learns dense latent representations from user-POI interactions:
        
        - Encodes each user's history into a high-dimensional embedding
        - Captures general preference patterns across all users
        - Reconstructs user interests via a decoder
        - Serves as the foundation for feature interpretation
        """
        )

    with st.expander("2️⃣ Sparse Autoencoder (SAE) — Feature Extraction", expanded=False):
        st.markdown(
            f"""
        **Sparse Autoencoder** decomposes ELSA embeddings into **{getattr(inference.sae, 'k', 64)} interpretable features**:
        
        - Only top-k features activate per user (highly sparse)
        - Each neuron learns a distinct semantic concept
        - Examples: "Italian Restaurants", "Budget-Friendly", "Late-Night Venues"
        - Enables human understanding of model decisions
        """
        )

    with st.expander("3️⃣ Interactive Steering — Adjust Preferences", expanded=False):
        st.markdown(
            """
        Use **steering sliders** to explore how preferences change recommendations:
        
        - Move slider → strength of that feature changes
        - Real-time updates to recommendations
        - See map and cards update instantly
        - Understand feature influence on rankings
        
        **Algorithm**: 
        ```
        z_steered = (1 - α) × z_user + α × steering_vector
        ```
        where α = 0.3 (30% influence)
        """
        )

    with st.expander("4️⃣ Evaluation Metrics — Quality Assessment", expanded=False):
        st.markdown(
            """
        Evaluate recommendation quality across:
        
        - **Recall@20**: What % of liked items appear in top-20?
        - **NDCG@100**: How well are items ranked?
        - **Model Size**: Trade-off between sparsity and accuracy
        - **Inference Speed**: <2s recommendations for interactive use
        """
        )

    # Navigation buttons
    st.divider()
    st.subheader("🚀 Get Started")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 View Evaluation Results", width='stretch'):
            st.switch_page("📊 Results")

    with col2:
        if st.button("🎛️ Try Interactive Steering", width='stretch'):
            st.switch_page("🎛️ Live Demo")

    with col3:
        if st.button("🔍 Browse Features", width='stretch'):
            st.switch_page("🔍 Interpretability")

    # Footer
    st.divider()
    st.markdown(
        """
    ---
    **Citation**: 
    Sparse autoencoders for recommendation systems interpretation.
    *WWW 2026* (Demo Track)
    
    **Data**: Yelp Academic Dataset (California businesses)
    """
    )
