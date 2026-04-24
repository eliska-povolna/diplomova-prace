"""Neuron labeling and interpretation using multiple approaches."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

STOP_CATEGORIES = {"Restaurants", "Food"}

# Try to import secrets helper for Streamlit integration
try:
    from src.ui.services.secrets_helper import get_google_api_key

    HAS_SECRETS_HELPER = True
except ImportError:
    HAS_SECRETS_HELPER = False

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning(
        "google-generativeai not installed. Install with: pip install google-generativeai"
    )

from src.utils.gemini_models import select_gemini_model
from src.interpret.prompts import (
    NEURON_LABEL_SYSTEM_PROMPT,
    SUPERFEATURE_SYSTEM_PROMPT,
)

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers not installed. Install with: pip install sentence-transformers"
    )


# ==================== Base Neuron Labeler ====================


class NeuronLabeler(ABC):
    """Abstract base class for neuron labeling strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def label_neurons(
        self,
        neuron_profiles: dict,
        business_metadata: dict,
    ) -> dict[int, str]:
        """Label neurons based on their activation profiles.

        Parameters
        ----------
        neuron_profiles : dict
            {neuron_idx: {"max_activating": [...], "zero_activating": [...]}}
        business_metadata : dict
            {business_id: metadata with categories, tags, etc.}

        Returns
        -------
        dict[int, str]
            {neuron_idx: label}
        """
        pass


# ==================== Tag-Based Labeling ====================


class TagBasedLabeler(NeuronLabeler):
    """Label neurons from activation-weighted business categories."""

    def __init__(
        self,
        min_tag_frequency: int = 3,
        max_tags_per_neuron: int = 3,
        item_index_to_business_id: dict = None,
    ):
        super().__init__("weighted-category")
        self.min_tag_frequency = min_tag_frequency
        self.max_tags_per_neuron = max_tags_per_neuron
        self.item_index_to_business_id = item_index_to_business_id or {}

    def label_neurons(
        self,
        neuron_profiles: dict,
        business_metadata: dict,
    ) -> dict[int, str]:
        """Generate labels using tag frequency analysis.

        Parameters
        ----------
        neuron_profiles : dict
            {neuron_idx: {"max_activating": [(item_idx, activation), ...], ...}}
        business_metadata : dict
            Can be keyed by:
            - Integer item indices (0-2211) if remapped
            - business_id strings (if items are already resolved)
        """
        labels = {}

        for neuron_idx, profile in neuron_profiles.items():
            # Support multiple data formats:
            # Format 1: {"max_activating": {"indices": [...], "activations": [...]}}
            # Format 2: {"max_activating": {"items": [...]}}
            # Format 3: {"max_activating": [...]}
            max_activating = profile.get("max_activating", [])
            max_items = []

            if isinstance(max_activating, dict):
                # Try indices + activations first (standard format)
                if "indices" in max_activating and "activations" in max_activating:
                    indices = max_activating.get("indices", [])
                    activations = max_activating.get("activations", [])
                    max_items = list(zip(indices, activations))
                # Fall back to items format
                elif "items" in max_activating:
                    max_items = max_activating.get("items", [])
            elif isinstance(max_activating, list):
                max_items = max_activating

            if not max_items:
                labels[neuron_idx] = "Unknown"
                continue

            # Extract activation-weighted categories from max-activating items
            category_counts = defaultdict(int)

            for item_id, activation in max_items[:10]:  # Top 10
                # Resolve item_id to business metadata
                meta = None

                # Method 1: If we have an item_index → business_id mapping, use it
                if self.item_index_to_business_id and isinstance(item_id, int):
                    business_id = self.item_index_to_business_id.get(item_id)
                    if business_id:
                        meta = business_metadata.get(business_id, {})

                # Method 2: Try direct lookup (supports both int indices and string keys)
                if not meta:
                    meta = business_metadata.get(item_id, {})

                # Method 3: If no metadata found and item_id is an index, try position-based lookup
                if not meta and isinstance(item_id, int):
                    business_list = list(business_metadata.values())
                    if 0 <= item_id < len(business_list):
                        meta = business_list[item_id]

                # Count categories
                for cat in meta.get("categories", []):
                    category_counts[cat] += activation

            if not category_counts:
                labels[neuron_idx] = "Unknown"
                continue

            # Create label
            tag_names = _rank_categories(category_counts)
            label = " and ".join(tag_names)
            labels[neuron_idx] = label[:100]  # Limit length

        return labels


# ==================== LLM-Based Labeling ====================


class LLMBasedLabeler(NeuronLabeler):
    """Label neurons using Google's Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3,
        rate_limit_delay: float = 1.0,
        progress_log_every: int = 25,
    ):
        super().__init__("llm-based")

        if not HAS_GEMINI:
            raise ImportError("google-generativeai not installed")

        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limit_delay = rate_limit_delay
        self.progress_log_every = progress_log_every

        # Setup API
        if api_key is None:
            api_key = (
                get_google_api_key()
                if HAS_SECRETS_HELPER
                else os.getenv("GOOGLE_API_KEY")
            )

        if not api_key:
            raise ValueError(
                "Gemini API key not found. "
                "Set GOOGLE_API_KEY env var, pass api_key parameter, or configure .streamlit/secrets.toml"
            )

        genai.configure(api_key=api_key)
        self.model_name = select_gemini_model(genai, model_name)
        logger.info(f"Gemini API configured. Model: {self.model_name}")

    def _format_max_activating(
        self,
        max_items: list[tuple[str, float]],
        business_metadata: dict,
    ) -> str:
        """Format max-activating examples for the prompt."""
        lines = []
        detailed_items = max_items[:5]
        additional_items = max_items[5:15]

        for i, (bid, activation) in enumerate(detailed_items, 1):
            meta = business_metadata.get(bid, {})

            # If no metadata found and bid is an index, try looking it up by position
            if not meta and isinstance(bid, int):
                business_list = list(business_metadata.values())
                if 0 <= bid < len(business_list):
                    meta = business_list[bid]

            name = meta.get("name", "Unknown")
            categories = ", ".join(meta.get("categories", [])[:3])

            line = f"{i}. {name} (Activation: {activation:.4f})"
            if categories:
                line += f"\n   Categories: {categories}"
            lines.append(line)

        if additional_items:
            lines.append("\nAdditional high-activation places:")
            for i, (bid, activation) in enumerate(additional_items, 1):
                meta = business_metadata.get(bid, {})
                if not meta and isinstance(bid, int):
                    business_list = list(business_metadata.values())
                    if 0 <= bid < len(business_list):
                        meta = business_list[bid]

                name = meta.get("name", "Unknown")
                categories = ", ".join(meta.get("categories", [])[:3])
                line = f"{i}. {name} ({activation:.4f})"
                if categories:
                    line += f" - {categories}"
                lines.append(line)

        return "\n".join(lines)

    def _interpret_single(
        self,
        neuron_idx: int,
        max_examples_text: str,
        review_examples_text: str = "",
    ) -> Optional[str]:
        """Interpret a single neuron using Gemini."""

        user_message = f"""
Max-activating examples:
{max_examples_text}
"""
        if review_examples_text:
            user_message += f"""

Representative user reviews:
{review_examples_text}
"""

        for attempt in range(self.max_retries):
            try:
                model = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=NEURON_LABEL_SYSTEM_PROMPT,
                )

                response = model.generate_content(
                    user_message,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature,
                        max_output_tokens=100,
                    ),
                )

                if response.text:
                    text = response.text.strip()
                    if "LABEL:" in text:
                        label = text.split("LABEL:")[-1].strip()
                        return label
                    else:
                        return text[:100]

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt+1}/{self.max_retries} failed "
                    f"for neuron {neuron_idx}: {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        logger.error(f"Failed to label neuron {neuron_idx}")
        return None

    def label_neurons(
        self,
        neuron_profiles: dict,
        business_metadata: dict,
    ) -> dict[int, str]:
        """Label neurons using Gemini API."""
        labels = {}

        def _rank_categories(category_counts: dict) -> list[str]:
            ranked = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
            specific = [
                (cat, score) for cat, score in ranked if cat not in STOP_CATEGORIES
            ]
            chosen = specific if specific else ranked
            return [tag for tag, _count in chosen[: self.max_tags_per_neuron]]
        total = len(neuron_profiles)
        successes = 0

        logger.info(
            "Starting %s labeling for %s neurons with Gemini model %s",
            self.name,
            total,
            self.model_name,
        )

        for position, (neuron_idx, profile) in enumerate(
            sorted(neuron_profiles.items()), 1
        ):
            # Support multiple data formats:
            # Format 1: {"max_activating": {"indices": [...], "activations": [...]}}
            # Format 2: {"max_activating": {"items": [...]}}
            max_activating = profile.get("max_activating", {})
            max_items = []

            if isinstance(max_activating, dict):
                # Try indices + activations first
                if "indices" in max_activating and "activations" in max_activating:
                    indices = max_activating.get("indices", [])
                    activations = max_activating.get("activations", [])
                    max_items = list(zip(indices, activations))
                # Fall back to items format
                elif "items" in max_activating:
                    max_items = max_activating.get("items", [])

            if not max_items:
                labels[neuron_idx] = "Unknown"
                continue

            max_text = self._format_max_activating(max_items, business_metadata)
            label = self._interpret_single(neuron_idx, max_text)

            if label:
                labels[neuron_idx] = label
                successes += 1
            else:
                labels[neuron_idx] = "Unknown"

            if (
                position == 1
                or position == total
                or position % self.progress_log_every == 0
            ):
                logger.info(
                    "[%s] Progress %s/%s (%.1f%%) successes=%s failures=%s",
                    self.name,
                    position,
                    total,
                    100.0 * position / max(total, 1),
                    successes,
                    position - successes,
                )

            # Rate limiting
            time.sleep(self.rate_limit_delay)

        return labels


class ReviewBasedLLMLabeler(LLMBasedLabeler):
    """Label neurons using metadata plus top useful review snippets."""

    def __init__(
        self,
        review_lookup: Optional[dict[str, list[dict]]] = None,
        max_reviews_per_business: int = 2,
        max_review_chars: int = 240,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = "llm-review-based"
        self.review_lookup = review_lookup or {}
        self.max_reviews_per_business = max_reviews_per_business
        self.max_review_chars = max_review_chars

    def _format_reviews(self, max_items: list[tuple[str, float]], business_metadata: dict) -> str:
        lines = []
        for business_id, _activation in max_items[:5]:
            business_key = str(business_id)
            reviews = self.review_lookup.get(business_key, [])[: self.max_reviews_per_business]
            if not reviews:
                continue

            meta = business_metadata.get(business_key, {})
            business_name = meta.get("name", business_key)
            lines.append(f"{business_name}:")
            for review in reviews:
                text = str(review.get("text", "")).strip().replace("\n", " ")
                if not text:
                    continue
                useful = review.get("useful", 0)
                stars = review.get("stars")
                clipped = text[: self.max_review_chars].strip()
                suffix = "..." if len(text) > self.max_review_chars else ""
                if stars is not None:
                    lines.append(
                        f'- Useful {useful}, Stars {stars}: "{clipped}{suffix}"'
                    )
                else:
                    lines.append(f'- Useful {useful}: "{clipped}{suffix}"')
        return "\n".join(lines)

    def label_neurons(
        self,
        neuron_profiles: dict,
        business_metadata: dict,
    ) -> dict[int, str]:
        labels = {}
        total = len(neuron_profiles)
        successes = 0
        logger.info("Preparing review-enriched prompts for %s neurons", total)

        for position, (neuron_idx, profile) in enumerate(
            sorted(neuron_profiles.items()), 1
        ):
            max_items = profile.get("max_activating", {}).get("items", [])

            if not max_items:
                labels[neuron_idx] = "Unknown"
                continue

            max_text = self._format_max_activating(max_items, business_metadata)
            review_text = self._format_reviews(max_items, business_metadata)

            label = self._interpret_single(
                neuron_idx,
                max_text,
                review_examples_text=review_text,
            )
            labels[neuron_idx] = label or "Unknown"
            if label:
                successes += 1
            if (
                position == 1
                or position == total
                or position % self.progress_log_every == 0
            ):
                logger.info(
                    "[%s] Progress %s/%s (%.1f%%) successes=%s failures=%s",
                    self.name,
                    position,
                    total,
                    100.0 * position / max(total, 1),
                    successes,
                    position - successes,
                )
            time.sleep(self.rate_limit_delay)

        return labels


# ==================== Neuron Embeddings ====================


class NeuronEmbedder:
    """Generate embeddings for neuron labels using sentence transformers."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-distilroberta-v1",
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers not installed")

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name}")

    def embed_labels(self, labels: dict[int, str]) -> torch.Tensor:
        """Create embeddings for neuron labels.

        Parameters
        ----------
        labels : dict[int, str]
            {neuron_idx: label}

        Returns
        -------
        torch.Tensor
            Shape (num_neurons, embedding_dim)
        """
        neuron_indices = sorted(labels.keys())
        label_texts = [labels[idx] for idx in neuron_indices]

        embeddings = self.model.encode(
            label_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
        )

        return embeddings, neuron_indices

    def compute_similarity_matrix(self, embeddings: torch.Tensor) -> np.ndarray:
        """Compute cosine similarity matrix between embeddings.

        Parameters
        ----------
        embeddings : torch.Tensor
            Shape (num_neurons, embedding_dim)

        Returns
        -------
        np.ndarray
            Similarity matrix (num_neurons, num_neurons)
        """
        embeddings_np = embeddings.cpu().numpy()
        num_neurons = embeddings_np.shape[0]

        similarity_matrix = np.zeros((num_neurons, num_neurons))

        for i in range(num_neurons):
            for j in range(i, num_neurons):
                sim = 1 - cosine(embeddings_np[i], embeddings_np[j])
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        return similarity_matrix


# ==================== Superfeature Clustering ====================


class SuperfeatureGenerator:
    """Group similar neurons into feature families and generate super-labels."""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.similarity_threshold = similarity_threshold

        if not HAS_GEMINI:
            raise ImportError("google-generativeai not installed")

        if api_key is None:
            api_key = (
                get_google_api_key()
                if HAS_SECRETS_HELPER
                else os.getenv("GOOGLE_API_KEY")
            )

        if not api_key:
            raise ValueError("Gemini API key not found")

        genai.configure(api_key=api_key)
        self.model_name = select_gemini_model(genai, model_name)

    def cluster_neurons(
        self,
        similarity_matrix: np.ndarray,
        neuron_indices: list[int],
    ) -> dict[int, list[int]]:
        """Cluster similar neurons based on similarity matrix.

        Parameters
        ----------
        similarity_matrix : np.ndarray
            Similarity matrix from NeuronEmbedder
        neuron_indices : list[int]
            List of neuron indices (in order of similarity matrix)

        Returns
        -------
        dict[int, list[int]]
            {cluster_id: [neuron_idx, ...]}
        """
        clustered = set()
        clusters = {}
        cluster_id = 0

        for i, neuron_a in enumerate(neuron_indices):
            if i in clustered:
                continue

            # Start new cluster
            cluster = [neuron_a]
            clustered.add(i)

            # Find similar neurons
            for j, neuron_b in enumerate(neuron_indices):
                if j <= i or j in clustered:
                    continue

                if similarity_matrix[i, j] >= self.similarity_threshold:
                    cluster.append(neuron_b)
                    clustered.add(j)

            if len(cluster) >= 2:  # Only save clusters with 2+ neurons
                clusters[cluster_id] = sorted(cluster)
                cluster_id += 1

        return clusters

    def generate_superlabel(
        self,
        similar_labels: list[str],
    ) -> Optional[str]:
        """Generate a super-label for a cluster of similar neurons.

        Parameters
        ----------
        similar_labels : list[str]
            Labels of neurons in the cluster

        Returns
        -------
        str or None
            Super-label (1-5 words) or None if failed
        """
        if len(similar_labels) < 2:
            return None

        label_list = "- " + "\n- ".join(similar_labels)
        user_message = f"Labels to synthesize:\n{label_list}"

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SUPERFEATURE_SYSTEM_PROMPT,
            )

            response = model.generate_content(
                user_message,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=50,
                ),
            )

            if response.text:
                text = response.text.strip()
                if "SUPERLABEL:" in text:
                    superlabel = text.split("SUPERLABEL:")[-1].strip()
                    return superlabel
                else:
                    return text[:50]

        except Exception as e:
            logger.error(f"Failed to generate super-label: {e}")

        return None

    def create_superfeatures(
        self,
        clusters: dict[int, list[int]],
        labels: dict[int, str],
    ) -> dict[int, dict]:
        """Create superfeature definitions from clusters.

        Parameters
        ----------
        clusters : dict[int, list[int]]
            Neuron clusters from cluster_neurons()
        labels : dict[int, str]
            Neuron labels

        Returns
        -------
        dict[int, dict]
            {superfeature_id: {"neurons": [...], "label": str}}
        """
        superfeatures = {}
        total = len(clusters)
        logger.info("Starting Gemini superfeature naming for %s clusters", total)

        for position, (cluster_id, neuron_list) in enumerate(clusters.items(), 1):
            similar_labels = [labels[nid] for nid in neuron_list]
            superlabel = self.generate_superlabel(similar_labels)

            if superlabel:
                superfeatures[cluster_id] = {
                    "neurons": neuron_list,
                    "sub_labels": similar_labels,
                    "super_label": superlabel,
                }
                logger.info(
                    f"Superfeature {cluster_id}: {superlabel} "
                    f"({len(neuron_list)} neurons)"
                )
            if position == 1 or position == total or position % 10 == 0:
                logger.info(
                    "[superfeatures] Progress %s/%s (%.1f%%)",
                    position,
                    total,
                    100.0 * position / max(total, 1),
                )

        return superfeatures
