"""Concept Steering UI Panel for Streamlit.

Allows users to search for neurons by semantic query and apply concept-based steering.
Integrates with live_demo.py as a separate steering tab.

Reference: IMPLEMENTATION_PLAN.md Task 8.4
"""

import json
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

logger = logging.getLogger(__name__)


def render_concept_steering_panel(
    inference_service,
    config: dict,
    session_state,
):
    """Render Concept Steering panel in Streamlit.

    Parameters
    ----------
    inference_service : InferenceService
        Service for model inference and neuron activation
    config : dict
        Configuration dict with model paths and hyperparameters
    session_state : st.session_state
        Streamlit session state for persistence

    Usage in live_demo.py:
    -------
    tab1, tab2 = st.tabs(["Neuron Steering", "Concept Steering"])
    with tab1:
        render_neuron_steering_panel(inference_service, config, st.session_state)
    with tab2:
        render_concept_steering_panel(inference_service, config, st.session_state)
    """
    try:
        from src.models.concept_steering import ConceptSteering
    except ImportError:
        st.error("❌ ConceptSteering module not found. Install dependencies.")
        return

    st.markdown("## 🔍 Concept Steering: Find Neurons by Semantic Query")
    st.markdown(
        """
    Find neurons that are semantically similar to a concept or user preference.
    
    **Example queries:**
    - "Italian restaurants" → Find neurons tuned to Italian cuisine
    - "quiet coffee shops" → Find neurons for peaceful dining
    - "upscale fine dining" → Find neurons for expensive venues
    """
    )

    # Load neuron labels from the already initialized label service when available.
    try:
        labels_service = session_state.get("labels")
        if labels_service is not None:
            hidden_dim = getattr(
                getattr(inference_service, "sae", None), "hidden_dim", 0
            )
            neuron_labels = {
                idx: labels_service.get_label(idx) for idx in range(hidden_dim)
            }
        else:
            output_dir = Path(config.get("model_checkpoint_dir", "outputs"))
            labels_path = output_dir.parent / "neuron_labels.json"

            if not labels_path.exists():
                st.warning(
                    "⚠️ Neuron labels are not available in the current session. "
                    "Run labeling first or configure GCS-backed labels."
                )
                return

            with open(labels_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "neuron_labels" in data:
                    neuron_labels = {
                        int(k): v for k, v in data["neuron_labels"].items()
                    }
                else:
                    neuron_labels = {int(k): v for k, v in data.items()}

    except Exception as e:
        st.error(f"❌ Failed to load neuron labels: {e}")
        return

    # Initialize ConceptSteering
    if "concept_steering" not in session_state:
        try:
            session_state.concept_steering = ConceptSteering(
                neuron_labels=neuron_labels,
                model_name="all-MiniLM-L6-v2",  # Fast 384-dim model
            )
            st.success(f"✅ Loaded {len(neuron_labels)} neuron labels")
        except ImportError as e:
            st.error(
                f"❌ SentenceTransformer required: pip install sentence-transformers"
            )
            return

    concept = session_state.concept_steering

    # UI Controls
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "🎯 Enter concept query:",
            placeholder="e.g., 'Italian restaurants', 'quiet cafes'",
            key="concept_query",
        )
    with col2:
        top_k = st.slider("Top-K:", min_value=1, max_value=20, value=10, step=1)

    if not query:
        st.info("💡 Enter a query to find related neurons")
        return

    # Find related neurons
    st.markdown("### Related Neurons")
    results = concept.find_related_neurons(query, top_k=top_k)

    if not results:
        st.warning("No neurons found for this query")
        return

    # Display results as table
    col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
    with col1:
        st.markdown("**#**")
    with col2:
        st.markdown("**Neuron Label**")
    with col3:
        st.markdown("**Similarity**")
    with col4:
        st.markdown("**Select**")

    st.divider()

    selected_neurons = []
    for idx, (neuron_id, label, similarity) in enumerate(results, 1):
        col1, col2, col3, col4 = st.columns([1, 3, 2, 1])

        with col1:
            st.write(f"{idx}.")
        with col2:
            st.write(f"**Neuron {neuron_id}:** {label}")
        with col3:
            # Show similarity as bar
            bar_width = int(similarity * 20)
            st.write(f"{'█' * bar_width} {similarity:.3f}")
        with col4:
            is_selected = st.checkbox(f"", key=f"neuron_{neuron_id}")
            if is_selected:
                selected_neurons.append(neuron_id)

    # Steering strength
    st.divider()
    st.markdown("### Steering Control")
    strength = st.slider(
        "Steering Strength (0=no effect, 1=full steering):",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Controls how much the concept affects recommendations",
    )

    if selected_neurons:
        st.markdown(f"✅ **Selected {len(selected_neurons)} neurons for steering**")

        if st.button("🚀 Apply Concept Steering"):
            st.info(
                f"Steering with {len(selected_neurons)} neurons at strength {strength:.2f}. "
                "Results will reflect concept-guided recommendations."
            )

            # TODO: Integrate with recommendation inference
            # concept_vector = concept.compute_concept_vector(query, strength=strength)
            # recommendations = inference_service.steer_with_concept(concept_vector, selected_neurons)

    else:
        st.info("Select neurons to apply steering")
