"""Interpretability page — Feature browser with labels and wordclouds."""

import logging
from collections import defaultdict

import streamlit as st

logger = logging.getLogger(__name__)


def resolve_item_to_business_name(item: dict, data_service, wordcloud_service) -> tuple:
    """Try to resolve an item to its business name.

    Returns (business_name, activation_value)
    """
    # Handle non-dict items (raw IDs)
    if not isinstance(item, dict):
        if data_service and hasattr(data_service, "get_business_name"):
            try:
                name = data_service.get_business_name(str(item))
                return (name, 0) if name else (str(item), 0)
            except Exception:
                pass
        return str(item), 0

    # Extract activation value
    activation = item.get("activation", item.get("avg_activation", 0))
    if isinstance(activation, dict):
        activation = activation.get("avg_activation", 0)

    # Extract business_id
    business_id = (
        item.get("business_id")
        or item.get("item_id")
        or item.get("id")
        or item.get("business")
        or item.get("name")
    )

    # Always try to look up business name from database
    if business_id and data_service and hasattr(data_service, "get_business_name"):
        try:
            name = data_service.get_business_name(str(business_id))
            if name:
                return name, activation
        except Exception:
            pass

    # Fallback: return business_id as-is
    return (
        str(business_id) if business_id else f"Item {item.get('index', '?')}",
        activation,
    )


def _get_superfeature_member_metadata(superfeature_context: dict, wordcloud_service):
    member_metadata = []
    if not superfeature_context or not wordcloud_service:
        return member_metadata

    metadata_source = getattr(wordcloud_service, "category_metadata", {}) or {}
    for member_id in superfeature_context.get("members", []):
        metadata = metadata_source.get(str(member_id), {}) or {}
        member_metadata.append((int(member_id), metadata))
    return member_metadata


def _aggregate_superfeature_categories(superfeature_context: dict, wordcloud_service):
    category_values = defaultdict(list)
    for _member_id, metadata in _get_superfeature_member_metadata(
        superfeature_context, wordcloud_service
    ):
        for category, activations in (
            metadata.get("category_weights", {}) or {}
        ).items():
            cleaned = [
                float(value) for value in activations if isinstance(value, (int, float))
            ]
            if cleaned:
                category_values[category].extend(cleaned)

    results = []
    for category, activations in category_values.items():
        total_activation = float(sum(activations))
        results.append(
            {
                "category": category,
                "total_activation": total_activation,
                "avg_activation": float(total_activation / len(activations)),
                "max_activation": float(max(activations)),
                "min_activation": float(min(activations)),
                "frequency": len(activations),
            }
        )

    results.sort(key=lambda item: item["total_activation"], reverse=True)
    return results


def _aggregate_superfeature_top_items(
    superfeature_context: dict,
    wordcloud_service,
    data_service,
):
    item_values = {}
    for _member_id, metadata in _get_superfeature_member_metadata(
        superfeature_context, wordcloud_service
    ):
        for item in metadata.get("top_items", []) or []:
            if not isinstance(item, dict):
                continue
            raw_item_id = (
                item.get("business_id")
                or item.get("item_id")
                or item.get("id")
                or item.get("business")
                or item.get("name")
            )
            if raw_item_id is None:
                continue

            item_key = str(raw_item_id)
            activation = item.get("activation", item.get("avg_activation", 0))
            try:
                activation_value = float(activation)
            except (TypeError, ValueError):
                continue

            bucket = item_values.setdefault(
                item_key,
                {
                    "sample": dict(item),
                    "activations": [],
                    "frequency": 0,
                },
            )
            bucket["activations"].append(activation_value)
            bucket["frequency"] += 1

    aggregated = []
    for item_key, payload in item_values.items():
        sample = payload["sample"]
        total_activation = float(sum(payload["activations"]))
        avg_activation = (
            total_activation / len(payload["activations"])
            if payload["activations"]
            else 0.0
        )
        business_name, _ = resolve_item_to_business_name(
            sample, data_service, wordcloud_service
        )
        aggregated.append(
            {
                "item_id": item_key,
                "business_name": business_name,
                "sample_item": sample,
                "total_activation": total_activation,
                "avg_activation": float(avg_activation),
                "frequency": payload["frequency"],
            }
        )

    aggregated.sort(key=lambda item: item["total_activation"], reverse=True)
    return aggregated


def _aggregate_superfeature_wordcloud_frequencies(
    superfeature_context: dict, wordcloud_service
):
    frequencies = defaultdict(float)
    for item in _aggregate_superfeature_categories(
        superfeature_context, wordcloud_service
    ):
        frequencies[item["category"]] = item["total_activation"]
    return dict(frequencies)


def render_clickable_feature(feature_id: int, feature_label: str):
    """Render a clickable feature button that navigates to the feature on this page."""
    if st.button(
        f"🔗 {feature_id}: {feature_label}", key=f"related_feature_{feature_id}"
    ):
        # Update session state: set feature ID and use a separate variable for pending search
        # (can't modify feature_search key directly after widget creation)
        logger.info(
            f"🔘 Related feature button clicked: feature_id={feature_id}, label={feature_label}"
        )
        if "selected_superfeature_id" in st.session_state:
            del st.session_state["selected_superfeature_id"]
        if "selected_superfeature_anchor_neuron" in st.session_state:
            del st.session_state["selected_superfeature_anchor_neuron"]
        st.session_state.selected_feature_id = feature_id
        st.session_state._pending_feature_search = str(feature_id)
        logger.info(f"   ✓ Set session_state.selected_feature_id = {feature_id}")
        logger.info(f"   ✓ Set session_state._pending_feature_search = '{feature_id}'")
        logger.info(f"   → Triggering rerun...")
        st.rerun()


def show():
    """Display interpretability page with neuron labels and wordclouds."""

    st.title("🔍 Feature Interpretability")

    st.markdown(
        """
    Browse all learned features and understand what each neuron represents
    through human-readable labels and visual wordclouds of activating business categories.
    """
    )
    st.caption(
        "By default the app follows the best run from the latest experiment (highest NDCG@20). "
        "The weighted-category baseline uses activation-weighted business categories rather than raw category counts, "
        "and deprioritizes very generic parent categories such as Restaurants and Food when more specific categories are available."
    )

    # Initialize services
    labels_service = st.session_state.get("labels")
    wordcloud_service = st.session_state.get("wordcloud")
    data_service = st.session_state.get("data")
    config = st.session_state.get("config", {})

    if not labels_service:
        st.error("❌ Labeling service not initialized")
        return

    available_methods = getattr(labels_service, "available_methods", [])
    if available_methods and hasattr(labels_service, "set_method"):
        global_method = st.session_state.get("global_label_method")
        if global_method in available_methods:
            labels_service.set_method(global_method)

    selected_method = getattr(labels_service, "selected_method", "weighted-category")
    st.caption(f"Using global label source: `{selected_method}`")
    superfeatures = (
        labels_service.get_superfeatures() if selected_method.startswith("llm") else {}
    )

    if not wordcloud_service:
        st.warning(
            "⚠️ Wordcloud service not available - labels will display without visualizations"
        )

    # Get semantic search settings from config
    semantic_threshold = config.get("ui", {}).get("semantic_search_threshold", 0.5)
    semantic_top_k = config.get("ui", {}).get("semantic_search_top_k", 10)

    # Get maximum neuron count from loaded model
    # IMPORTANT: Use hidden_dim (actual neuron count), NOT k (sparsity level)
    # k = how many neurons to keep active per sample
    # hidden_dim = total neurons in the dictionary
    max_neuron = (
        st.session_state.inference.sae.hidden_dim - 1
        if (
            hasattr(st.session_state, "inference")
            and hasattr(st.session_state.inference, "sae")
        )
        else 1023
    )

    # ═══════════════════════════════════════════════════════════════════
    # SIDEBAR: Feature Search and Selection
    # ═══════════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 🔍 Search Features")
        st.caption(
            "Search by feature number (e.g., '5', '42'), label name (e.g., 'Italian', 'Coffee'), or by meaning (e.g., 'Asian' finds semantically similar features)"
        )
        browse_mode = "Neuron"
        if superfeatures:
            browse_mode = st.radio(
                "Browse",
                options=["Neuron", "Superfeature"],
                horizontal=True,
                key="interpretability_browse_mode",
            )

        # If pending feature search was set by button click, update session state directly
        # (value param doesn't override existing keyed widget values in session_state)
        if st.session_state.get("_pending_feature_search"):
            pending_value = st.session_state._pending_feature_search
            logger.info(
                f"📝 Found pending_feature_search in session_state: '{pending_value}'"
            )
            logger.info(
                f"   → Updating session_state.feature_search to '{pending_value}'"
            )
            st.session_state.feature_search = pending_value
            logger.info(f"   ✓ session_state.feature_search updated")
            del st.session_state._pending_feature_search
            logger.info(f"   ✓ Cleared _pending_feature_search")
        else:
            logger.debug("No pending_feature_search found")

        logger.debug(
            f"Current session_state.feature_search = '{st.session_state.get('feature_search', '')}' (before widget creation)"
        )

        search_query = st.text_input(
            "Search features",
            placeholder="e.g., Italian or 5 or Asian",
            key="feature_search",
            label_visibility="collapsed",
        )

        logger.debug(f"Search bar widget created, search_query = '{search_query}'")

        # Check if feature was passed via session_state (from related features button)
        neuron_idx = None
        if st.session_state.get("selected_feature_id") is not None:
            neuron_idx = st.session_state.selected_feature_id
        # Check URL query parameter as fallback
        elif "feature" in st.query_params:
            try:
                neuron_idx = int(st.query_params["feature"])
            except (ValueError, TypeError):
                neuron_idx = None

        # Validate neuron index
        if neuron_idx is not None and not (0 <= neuron_idx <= max_neuron):
            neuron_idx = None

        # Search should stay independent of the currently selected feature.
        matching_features = []
        if browse_mode == "Superfeature":
            matching_superfeatures = []
            for sf_id, sf_data in superfeatures.items():
                super_label = str(sf_data.get("super_label", f"Superfeature {sf_id}"))
                sub_labels = " ".join(sf_data.get("sub_labels", []))
                haystack = f"{super_label} {sub_labels}".lower()
                if not search_query or search_query.lower() in haystack:
                    matching_superfeatures.append((sf_id, super_label))

            if matching_superfeatures:
                selected_sf_idx = st.radio(
                    "Select a superfeature",
                    options=range(len(matching_superfeatures)),
                    format_func=lambda i: f"{matching_superfeatures[i][1]}",
                    key="superfeature_selection_radio",
                    label_visibility="collapsed",
                )
                st.session_state.selected_superfeature_id = matching_superfeatures[
                    selected_sf_idx
                ][0]
            elif search_query:
                st.warning("No superfeatures found. Try a different search.")
        elif search_query:
            # Try to match as a number first
            try:
                search_num = int(search_query)
                if 0 <= search_num <= max_neuron:
                    try:
                        label = labels_service.get_label(search_num)
                        matching_features.append((search_num, label))
                    except Exception:
                        matching_features.append((search_num, f"Feature {search_num}"))
            except ValueError:
                pass

            # If no exact number match, try substring search
            if not matching_features:
                for idx in range(max_neuron + 1):
                    try:
                        label = labels_service.get_label(idx)
                        if search_query.lower() in label.lower():
                            matching_features.append((idx, label))
                    except Exception:
                        pass

            # If still no results, try semantic search (use cached model from cache.py)
            if not matching_features:
                try:
                    from cache import (
                        load_semantic_search_model,
                        cache_all_label_embeddings,
                    )

                    with st.spinner("🔍 Searching with semantic model..."):
                        semantic_model = load_semantic_search_model()
                        logger.info(
                            f"Semantic model loaded: {semantic_model is not None}"
                        )

                        if semantic_model is not None:
                            # Get cached label embeddings (cached in session_state on first call)
                            label_embeddings_dict = cache_all_label_embeddings(
                                labels_service, max_neuron
                            )
                            logger.info(
                                f"Label embeddings cached: {label_embeddings_dict is not None}, count: {len(label_embeddings_dict) if label_embeddings_dict else 0}"
                            )

                            if label_embeddings_dict:
                                # Encode query once
                                query_embedding = semantic_model.encode(
                                    search_query, show_progress_bar=False
                                )
                                logger.info(
                                    f"Query embedding - type: {type(query_embedding).__name__}, shape: {query_embedding.shape}"
                                )

                                # Compute batch similarity with all cached label embeddings using numpy
                                import numpy as np

                                similarities = []
                                logger.info(
                                    f"Computing similarities for {len(label_embeddings_dict)} labels"
                                )

                                for (
                                    idx,
                                    label_embedding,
                                ) in label_embeddings_dict.items():
                                    try:
                                        # Compute cosine similarity using numpy (more reliable)
                                        # cos_sim = dot(a, b) / (norm(a) * norm(b))
                                        dot_product = np.dot(
                                            query_embedding, label_embedding
                                        )
                                        norm_query = np.linalg.norm(query_embedding)
                                        norm_label = np.linalg.norm(label_embedding)
                                        similarity = dot_product / (
                                            norm_query * norm_label
                                        )

                                        label = labels_service.get_label(idx)
                                        similarities.append((idx, label, similarity))
                                        logger.debug(
                                            f"  idx={idx}, label={label}, sim={similarity:.4f}"
                                        )
                                    except Exception as e:
                                        logger.error(
                                            f"Similarity computation failed for idx {idx}: {type(e).__name__}: {e}"
                                        )

                                # Sort by similarity and take top 10
                                similarities.sort(key=lambda x: x[2], reverse=True)
                                logger.info(
                                    f"Top 5 similarities: {[(idx, sim) for idx, _, sim in similarities[:5]]}"
                                )

                                matching_features = [
                                    (idx, label)
                                    for idx, label, sim in similarities[:semantic_top_k]
                                    if sim > semantic_threshold  # Threshold from config
                                ]
                                logger.info(
                                    f"Features matching threshold {semantic_threshold}: {len(matching_features)}"
                                )
                            else:
                                logger.warning("Label embeddings dict is None or empty")
                        else:
                            logger.warning("Semantic model is None")
                except Exception as e:
                    logger.error(f"Semantic search failed: {e}", exc_info=True)

            # Display all matching results in sidebar
            if matching_features:
                st.write(f"**Found {len(matching_features)} match(es)**")

                # Create a scrollable selection area in sidebar
                selected_idx = st.radio(
                    "Select a feature",
                    options=range(len(matching_features)),
                    format_func=lambda i: f"#{matching_features[i][0]}: {matching_features[i][1]}",
                    key="feature_selection_radio",
                    label_visibility="collapsed",
                )
                neuron_idx = matching_features[selected_idx][0]
                st.session_state.selected_feature_id = neuron_idx
            else:
                if search_query:
                    st.warning("No features found. Try a different search.")
        elif browse_mode != "Superfeature" and neuron_idx is None and not search_query:
            st.info("👉 Enter a feature name or number to get started")

    # ═══════════════════════════════════════════════════════════════════
    # MAIN AREA: Feature Details
    # ═══════════════════════════════════════════════════════════════════

    superfeature_context = None
    neuron_idx = None

    superfeature_id = st.session_state.get("selected_superfeature_id")
    if superfeature_id is not None:
        superfeature = superfeatures.get(str(superfeature_id), {})
        member_neurons = []
        for raw_n in superfeature.get("neurons", []):
            try:
                n = int(raw_n)
            except Exception:
                continue
            if 0 <= n <= max_neuron:
                member_neurons.append(n)

        if not member_neurons:
            st.warning("No valid member neurons available for this superfeature.")
            st.stop()

        sub_labels = superfeature.get("sub_labels", []) or []
        member_labels = {
            n: str(lbl) for n, lbl in zip(member_neurons, sub_labels) if lbl is not None
        }

        anchor = st.session_state.get("selected_superfeature_anchor_neuron")
        if anchor not in member_neurons:
            anchor = member_neurons[0]
            st.session_state.selected_superfeature_anchor_neuron = anchor

        neuron_idx = int(anchor)
        superfeature_context = {
            "id": str(superfeature_id),
            "label": superfeature.get("super_label", f"Superfeature {superfeature_id}"),
            "members": member_neurons,
            "member_labels": member_labels,
        }
    else:
        # Get the selected feature from sidebar (session_state handles the selection)
        neuron_idx = st.session_state.get("selected_feature_id")
        logger.info(f"🔍 Main area: retrieved selected_feature_id = {neuron_idx}")

        # Clear it so it doesn't interfere with next interaction
        if "selected_feature_id" in st.session_state:
            logger.info(f"   ✓ Clearing selected_feature_id from session_state")
            del st.session_state.selected_feature_id

    # Validate neuron index
    if neuron_idx is not None and not (0 <= neuron_idx <= max_neuron):
        logger.warning(
            f"   ✗ Invalid neuron_idx {neuron_idx} (max_neuron={max_neuron}), setting to None"
        )
        neuron_idx = None
    elif neuron_idx is not None:
        logger.info(f"   ✓ Valid neuron_idx: {neuron_idx} (max_neuron={max_neuron})")

    # If no feature selected, show placeholder
    if neuron_idx is None:
        st.info("👈 Use the left panel to search for and select a feature")
        st.stop()

    # Get label
    try:
        label = labels_service.get_label(neuron_idx)
    except Exception as e:
        label = f"Feature {neuron_idx}"
        logger.warning(f"Failed to get label: {e}")

    # Headline
    if superfeature_context is not None:
        st.markdown(f"## Superfeature: {superfeature_context['label']}")
        st.caption(
            f"{len(superfeature_context['members'])} member neurons from `{selected_method}` labels"
        )
        st.caption(
            "The sections below use statistics aggregated across the whole superfeature. "
        )
    else:
        st.markdown(f"## Feature #{neuron_idx}: {label}")
    st.divider()

    # Top Activating Categories Chart + Wordcloud (side by side)
    col_chart, col_wordcloud = st.columns([1.2, 1])

    with col_chart:
        st.subheader("🔥 Top Activating Categories")

        with st.expander("ℹ️ How is this calculated?", expanded=False):
            if superfeature_context is not None:
                st.write(
                    "**Σ** = sum of activation strengths across all member neurons for places with this category\n\n"
                    "**n** = how many member neurons' top lists included this category (frequency across members)\n\n"
                    "Categories are ranked by total contribution (Σ) aggregated across the superfeature (consistent with weighted-category labeling)."
                )
            else:
                st.write(
                    "**Σ** = sum of activation strengths for this neuron across all places with this category\n\n"
                    "**n** = number of recommended places with this category where the neuron activated\n\n"
                    "Categories are ranked by total contribution (Σ), consistent with weighted-category labeling."
                )

        if wordcloud_service:
            try:
                if superfeature_context is not None:
                    top_categories = _aggregate_superfeature_categories(
                        superfeature_context, wordcloud_service
                    )[:10]
                else:
                    top_categories = wordcloud_service.get_top_activating_categories(
                        neuron_idx, top_k=10
                    )

                if top_categories:
                    # Find max activation for normalization
                    max_total = (
                        max([c["total_activation"] for c in top_categories])
                        if top_categories
                        else 1.0
                    )

                    # Display categories with activation strength bars
                    for item in top_categories:
                        category = item["category"]
                        total_activation = item["total_activation"]
                        frequency = item["frequency"]

                        # Normalize for display
                        strength_pct = min(100, max(0, int(total_activation * 100)))
                        normalized_strength = (
                            total_activation / max_total if max_total > 0 else 0
                        )

                        # Multi-line display with activation strength and frequency
                        col_name, col_strength = st.columns([2, 1])

                        with col_name:
                            st.caption(f"**{category}**")

                        with col_strength:
                            st.caption(f"Σ={total_activation:.2f} (n={frequency})")

                        # Progress bar for visual strength
                        st.progress(
                            min(1.0, normalized_strength), text=f"{strength_pct}%"
                        )
                else:
                    st.info("No category data available for this feature")
            except Exception as e:
                st.warning(f"Could not load activation categories: {e}")
        else:
            st.info("Wordcloud service not available")

    with col_wordcloud:
        st.subheader("☁️ Wordcloud")

        with st.expander("ℹ️ How is this calculated?", expanded=False):
            if superfeature_context is not None:
                st.write(
                    "Word cloud shows the relative frequency and importance of categories based on aggregated total activation across all member neurons.\n\n"
                    "Category size reflects how much that category contributes to the superfeature's overall activation (combines frequency and strength across members)."
                )
            else:
                st.write(
                    "Word cloud shows the relative frequency and importance of categories based on their total activation.\n\n"
                    "Category size reflects how much that category contributes to the neuron's overall activation across all top-activating items."
                )

        if wordcloud_service:
            try:
                with st.spinner("📊 Generating wordcloud..."):
                    if superfeature_context is not None:
                        fig = wordcloud_service.generate_wordcloud_from_frequencies(
                            _aggregate_superfeature_wordcloud_frequencies(
                                superfeature_context, wordcloud_service
                            ),
                            title=f"Superfeature: {superfeature_context['label']}",
                            figsize=(6, 4),
                            width=600,
                            height=400,
                            colormap="tab20",
                        )
                    else:
                        fig = wordcloud_service.generate_wordcloud_fig(
                            neuron_idx,
                            figsize=(6, 4),
                            width=600,
                            height=400,
                            colormap="tab20",
                        )

                if fig is not None:
                    st.pyplot(fig, width="stretch")
                else:
                    st.info("No wordcloud data available")
            except Exception as e:
                st.warning(f"Wordcloud generation failed: {str(e)[:80]}")
                logger.error(f"Wordcloud generation error: {e}")
        else:
            st.info("Wordcloud service not available")

    st.divider()

    # Top activating businesses section
    if wordcloud_service:
        st.subheader("🏢 Top Activating Businesses")

        with st.expander("ℹ️ How is this calculated?", expanded=False):
            if superfeature_context is not None:
                st.write(
                    "Shows the top places aggregated across member neurons of the superfeature.\n\n"
                    "**Σ** = sum of activations for this business across all member neurons' top lists.\n"
                    "**n** = number of member neurons that listed this business among their top items."
                )
            else:
                st.write(
                    "Shows the top-10 individual places that most strongly activate this neuron.\n\n"
                    "**σ** = activation strength for this place (how much this neuron activated when recommending it)"
                )
        try:
            if superfeature_context is not None:
                top_items = _aggregate_superfeature_top_items(
                    superfeature_context, wordcloud_service, data_service
                )[:10]
            else:
                top_items = wordcloud_service.get_top_items(neuron_idx, top_k=10)

            logger.debug(f"🔍 DISPLAYING TOP ITEMS: {len(top_items)} items found")
            if top_items:
                logger.debug(f"   First item: {top_items[0]}")
                for i, item in enumerate(top_items, 1):
                    logger.debug(f"   Item #{i}: {item}")
                    # Resolve item to business name
                    if superfeature_context is not None:
                        business_name = item["business_name"]
                        activation = item["total_activation"]
                    else:
                        business_name, activation = resolve_item_to_business_name(
                            item, data_service, wordcloud_service
                        )
                    logger.debug(f"   Resolved to: {business_name}")

                    col_rank, col_name, col_stats = st.columns([0.5, 2, 1])
                    with col_rank:
                        st.caption(f"#{i}")
                    with col_name:
                        st.caption(f"**{business_name}**")
                    with col_stats:
                        if superfeature_context is not None:
                            st.caption(f"Σ={activation:.2f} (n={item['frequency']})")
                        else:
                            st.caption(f"σ={activation:.2f}")
            else:
                st.info("No business data available for this feature")
        except Exception as e:
            st.caption(f"Could not load top businesses: {e}")
            logger.error(f"Top businesses error: {e}", exc_info=True)

    st.divider()
    # Related section: co-activation for neuron mode, member neurons for superfeature mode
    if superfeature_context is not None:
        st.subheader("Member Neurons")
        members = superfeature_context["members"]
        member_labels = superfeature_context["member_labels"]
        sf_id = superfeature_context["id"]

        for member_id in members:
            member_label = member_labels.get(member_id)
            if not member_label:
                try:
                    member_label = labels_service.get_label(member_id)
                except Exception:
                    member_label = f"Feature {member_id}"

            if st.button(
                f"Feature {member_id}: {member_label}",
                key=f"superfeature_member_{sf_id}_{member_id}",
            ):
                st.session_state.selected_superfeature_anchor_neuron = int(member_id)
                st.rerun()
    else:
        coactivation_service = st.session_state.get("coactivation")

        if coactivation_service and neuron_idx is not None:
            st.subheader("Related Features")

            with st.expander("ℹ️ How is this calculated?", expanded=False):
                st.write(
                    "**Frequently Co-Activated** = features that activate together with this neuron (positive correlation)\n\n"
                    "**Rarely Co-Activated** = features that activate opposite to this neuron (negative correlation)\n\n"
                    "Computed from Pearson correlation across all neurons in the model."
                )

            # Log diagnostic info about data sources
            if (
                hasattr(coactivation_service, "coactivation_data")
                and coactivation_service.coactivation_data
            ):
                coact_neuron_ids = [
                    int(k) for k in coactivation_service.coactivation_data.keys()
                ]
                coact_max = max(coact_neuron_ids) if coact_neuron_ids else 0
                logger.debug(
                    f"Data source mismatch diagnostic:"
                    f"\n   Model max_neuron: {max_neuron}"
                    f"\n   Coactivation data max neuron_id: {coact_max}"
                    f"\n   Coactivation data neurons: {len(coactivation_service.coactivation_data)}"
                    f"\n   Current neuron_idx: {neuron_idx}"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Frequently Co-Activated With**")
                highly_coactivated = coactivation_service.get_highly_coactivated(
                    neuron_idx
                )
                if highly_coactivated:
                    for item in highly_coactivated:
                        render_clickable_feature(item["neuron_id"], item["label"])
                else:
                    st.caption("*No positive co-activation data found*")

            with col2:
                st.markdown("**Rarely Co-Activated With**")
                rarely_coactivated = coactivation_service.get_rarely_coactivated(
                    neuron_idx
                )
                if rarely_coactivated:
                    for item in rarely_coactivated:
                        render_clickable_feature(item["neuron_id"], item["label"])
                else:
                    st.caption("*No negative correlations found*")
