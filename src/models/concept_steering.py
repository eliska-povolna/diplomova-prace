"""Concept Steering: Find neurons semantically similar to user text queries.

This module implements semantic neuron discovery using SentenceTransformer embeddings.
Users can type a query like "Italian restaurants" and find neurons with similar labels,
enabling guided exploration of the SAE feature space.

Reference: IMPLEMENTATION_PLAN.md Task 8.4
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "sentence-transformers not installed. Install with: pip install sentence-transformers"
    )


class ConceptSteering:
    """Find neurons semantically similar to user text queries.

    Algorithm:
    1. Encode user query to 384-dimensional embedding
    2. Compute cosine similarity with all neuron label embeddings
    3. Return top-k neurons sorted by similarity (highest first)
    4. Apply steering formula: z' = α·z + γ·m_C
       where α is model steering strength (user-selected), m_C is concept vector

    Parameters
    ----------
    neuron_labels : dict[int, str]
        Mapping from neuron index to label (e.g., {5: "Italian Fine Dining", 42: "Casual Cafes"})
    model_name : str
        SentenceTransformer model (default: "all-MiniLM-L6-v2" - 384-dim, fast)
    precomputed_embeddings_path : str, optional
        Path to precomputed embeddings cache (JSON file with {str(neuron_idx): [...]} format)
    """

    def __init__(
        self,
        neuron_labels: dict[int, str],
        model_name: str = "all-MiniLM-L6-v2",
        precomputed_embeddings_path: Optional[str] = None,
    ):
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError(
                "sentence-transformers library required for Concept Steering. "
                "Install with: pip install sentence-transformers"
            )

        self.neuron_labels = neuron_labels
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

        # Embedding dimension (all-MiniLM-L6-v2 is 384-dim)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Concept Steering: Using {model_name} ({self.embedding_dim}D)")

        # Precomputed embeddings cache
        self.neuron_embeddings = {}
        self.precomputed_path = precomputed_embeddings_path

        # Load precomputed embeddings if available
        if precomputed_embeddings_path and Path(precomputed_embeddings_path).exists():
            self._load_precomputed_embeddings()
        else:
            # Compute embeddings on first use
            logger.info("No precomputed embeddings found. Will compute on first query.")

    def _load_precomputed_embeddings(self):
        """Load neuron label embeddings from cache file."""
        try:
            with open(self.precomputed_path, "r") as f:
                data = json.load(f)
                # Convert list back to numpy array
                self.neuron_embeddings = {
                    int(k): np.array(v, dtype=np.float32) for k, v in data.items()
                }
                logger.info(
                    f"✅ Loaded {len(self.neuron_embeddings)} precomputed embeddings"
                )
        except Exception as e:
            logger.warning(f"Failed to load precomputed embeddings: {e}")
            self.neuron_embeddings = {}

    def _compute_embeddings(self):
        """Compute embeddings for all neuron labels."""
        if self.neuron_embeddings:
            return  # Already computed or loaded

        logger.info(f"Computing embeddings for {len(self.neuron_labels)} neurons...")

        labels_list = []
        neuron_indices = []

        for neuron_idx, label in sorted(self.neuron_labels.items()):
            labels_list.append(label)
            neuron_indices.append(neuron_idx)

        # Encode all labels at once (efficient batching)
        embeddings = self.model.encode(
            labels_list, convert_to_numpy=True, show_progress_bar=False
        )

        # Store as dict
        for idx, neuron_idx in enumerate(neuron_indices):
            self.neuron_embeddings[neuron_idx] = embeddings[idx].astype(np.float32)

        logger.info(f"✅ Computed {len(self.neuron_embeddings)} embeddings")

    def save_embeddings(self, output_path: str):
        """Save neuron embeddings to cache file (JSON format).

        Parameters
        ----------
        output_path : str
            Path to save embeddings cache
        """
        # Ensure embeddings are computed
        self._compute_embeddings()

        # Convert to JSON-serializable format
        data = {str(k): v.tolist() for k, v in self.neuron_embeddings.items()}

        with open(output_path, "w") as f:
            json.dump(data, f)

        logger.info(f"✅ Saved embeddings to {output_path}")

    def find_related_neurons(
        self, query_text: str, top_k: int = 10
    ) -> list[tuple[int, str, float]]:
        """Find neurons with labels semantically similar to query text.

        Parameters
        ----------
        query_text : str
            User query (e.g., "Italian restaurants", "quiet coffee shops")
        top_k : int
            Number of top neurons to return (default: 10)

        Returns
        -------
        list[tuple[int, str, float]]
            List of (neuron_idx, label, similarity_score) tuples, sorted by similarity (highest first)
            - neuron_idx: Index of the neuron (int)
            - label: Pre-computed neuron label (str)
            - similarity_score: Cosine similarity between query and neuron label (0-1 range)

        Example
        -------
        >>> concept = ConceptSteering(neuron_labels)
        >>> results = concept.find_related_neurons("Italian fine dining", top_k=10)
        >>> for neuron_idx, label, similarity in results:
        ...     print(f"Neuron {neuron_idx}: {label} (similarity: {similarity:.3f})")
        """
        # Ensure embeddings are computed
        self._compute_embeddings()

        # Encode query
        query_embedding = self.model.encode(query_text, convert_to_numpy=True).astype(
            np.float32
        )

        # Compute cosine similarities
        similarities = []
        for neuron_idx, neuron_embedding in self.neuron_embeddings.items():
            # Cosine similarity = (A · B) / (||A|| ||B||)
            similarity = np.dot(query_embedding, neuron_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(neuron_embedding)
            )

            label = self.neuron_labels.get(neuron_idx, "Unknown")
            similarities.append((neuron_idx, label, float(similarity)))

        # Sort by similarity (highest first) and return top-k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    def compute_concept_vector(
        self, query_text: str, strength: float = 1.0
    ) -> np.ndarray:
        """Compute steering vector for concept-based steering.

        Applies the steering formula: z' = α·z + γ·m_C
        where m_C is the average embedding of top-k neurons similar to the query,
        and γ is the steering strength (0-1, where 1 = full steering).

        Parameters
        ----------
        query_text : str
            Concept query (e.g., "upscale dining")
        strength : float
            Steering strength (0-1, default: 1.0)
            - 0 = no steering (return zero vector)
            - 1 = full steering (return concept vector)

        Returns
        -------
        np.ndarray
            Concept vector (same dimension as model embeddings, typically 384D)

        Example
        -------
        >>> concept = ConceptSteering(neuron_labels)
        >>> concept_vec = concept.compute_concept_vector("Italian restaurants", strength=0.5)
        >>> # To apply to model: z_new = z_original + concept_vec
        """
        # Ensure embeddings are computed
        self._compute_embeddings()

        # Find top-5 related neurons (use their embeddings to form concept)
        related = self.find_related_neurons(query_text, top_k=5)

        if not related:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Average embeddings of top-k related neurons
        related_embeddings = np.array(
            [self.neuron_embeddings[neuron_idx] for neuron_idx, _, _ in related]
        )
        concept_vector = np.mean(related_embeddings, axis=0)

        # Apply strength scaling
        concept_vector = strength * concept_vector

        return concept_vector.astype(np.float32)
