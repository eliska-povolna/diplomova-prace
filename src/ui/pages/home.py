"""Home page: overview and system explanation."""

from typing import Any, Dict, Optional

import streamlit as st


def _safe_get(d: Optional[Dict[str, Any]], *path, default=None):
    current: Any = d
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _dataset_label(config: Optional[Dict[str, Any]]) -> str:
    if not isinstance(config, dict):
        return "Yelp (all)"
    state_filter = config.get("state_filter")
    if state_filter and str(state_filter).strip().lower() not in {"all", "*"}:
        return f"Yelp ({str(state_filter).strip()})"
    return "Yelp (all)"


def show():
    """Display home page."""
    inference = st.session_state.get("inference")
    data = st.session_state.get("data")
    training_results = st.session_state.get("training_results") or {}
    config = st.session_state.get("config") or {}

    if not inference or not data:
        st.error("Services not initialized")
        return

    summary_data = _safe_get(training_results, "summary", "data", default={}) or {}
    summary_eval = (
        _safe_get(training_results, "summary", "evaluation", default={}) or {}
    )

    n_users = (
        summary_data.get("n_users")
        or summary_data.get("users")
        or summary_data.get("num_users")
    )
    n_items = (
        summary_data.get("n_items")
        or summary_data.get("items")
        or summary_data.get("num_items")
        or data.num_pois
    )
    n_interactions = (
        summary_data.get("n_interactions")
        or summary_data.get("interactions")
        or summary_data.get("num_interactions")
    )

    test_users_count = summary_eval.get("n_users_test")
    if test_users_count is None:
        test_users_count = len(data.get_test_users(limit=50))

    default_alpha = float(getattr(inference, "alpha", 0.3))

    st.title("🗺️ Interpretable POI Recommender")
    st.write(
        """
Discover places tailored to user preferences with sparse, interpretable neural features.

This app combines:
- ELSA collaborative filtering for dense user representations
- Sparse autoencoder decomposition for interpretable features
- Interactive steering with a shared canonical neuron-level configuration
"""
    )

    st.subheader("📊 Dataset Overview")
    metrics_cols = st.columns(5)
    with metrics_cols[0]:
        if isinstance(n_items, int):
            st.metric("POIs", f"{n_items:,}", delta=None, delta_color="off")
        else:
            st.metric("POIs", str(n_items), delta=None, delta_color="off")
    with metrics_cols[1]:
        st.metric("Dataset", _dataset_label(config), delta=None, delta_color="off")
    with metrics_cols[2]:
        st.metric(
            "Users",
            f"{int(n_users):,}" if isinstance(n_users, (int, float)) else "N/A",
            delta=None,
            delta_color="off",
        )
    with metrics_cols[3]:
        st.metric(
            "Interactions",
            (
                f"{int(n_interactions):,}"
                if isinstance(n_interactions, (int, float))
                else "N/A"
            ),
            delta=None,
            delta_color="off",
        )
    with metrics_cols[4]:
        st.metric(
            "Test users",
            (
                f"{int(test_users_count):,}"
                if isinstance(test_users_count, (int, float))
                else "N/A"
            ),
            delta=None,
            delta_color="off",
        )

    st.subheader("⚙️ How It Works")

    with st.expander("1️⃣ ELSA: Collaborative Filtering"):
        st.markdown(
            """
- Encodes each user's interaction history into a dense latent vector.
- Learns collaborative structure across similar users and venues.
- Produces the base ranking signal before interpretability and steering layers.
- Captures preference patterns that are hard to express with simple hand-crafted rules.
"""
        )

    with st.expander("2️⃣ Sparse Autoencoder: Interpretable Features"):
        st.markdown(
            f"""
- SAE dictionary width (`hidden_dim`) controls total feature capacity.
- Sparsity (`k`) controls how many features are active per user vector.
- Feature meanings come from run-scoped labeling artifacts loaded for the strict best run.
- This enables feature-level explanations and controllable steering in the UI.
"""
        )

    with st.expander("3️⃣ Unified Steering (Concept + Neuron Compatible)"):
        st.markdown(
            f"""
- Concept steering and neuron steering both write to the same canonical `neuron_values` map.
- Updates are merged (patch semantics), so one steering action does not erase unrelated edits.
- A single `alpha` controls interpolation strength for the final latent vector.

Formula used by inference:
```text
z_final = (1 - alpha) * z_user + alpha * z_steered
```
Default `alpha` in this run is `{default_alpha:.2f}`.
"""
        )

    with st.expander("4️⃣ Evaluation Metrics"):
        st.markdown(
            """
- `Recall@20`: how many relevant places appear in top-20 recommendations.
- `NDCG@20`: ranking quality with position-aware gain.
- Additional ranking, diversity, and latency metrics are shown on the Results page.
"""
        )

    st.divider()
    st.subheader("🚀 Get Started")

    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("📊 View Evaluation Results", width="stretch"):
            page = st.session_state.get("_results_page")
            if page:
                st.switch_page(page)
    with nav2:
        if st.button("🎛️ Try Interactive Steering", width="stretch"):
            page = st.session_state.get("_live_demo_page")
            if page:
                st.switch_page(page)
    with nav3:
        if st.button("🔍 Browse Features", width="stretch"):
            page = st.session_state.get("_interpretability_page")
            if page:
                st.switch_page(page)
