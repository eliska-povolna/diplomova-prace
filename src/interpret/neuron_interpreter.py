"""LLM-based neuron interpretation for SAE sparse features.

This module provides automated labeling of sparse autoencoder neurons
using a two-phase LLM approach:

Phase 1: Initial labeling via max/zero-activating examples
Phase 2: Hierarchical organization into super-features (feature families)

Supports two LLM APIs:
- GitHub Models (GPT-4o) - Recommended for quality and availability
- Google Gemini API - Free tier alternative with rate limiting
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import time
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try to import both APIs
try:
    from openai import OpenAI

    HAS_GITHUB_MODELS = True
except ImportError:
    HAS_GITHUB_MODELS = False
    logger.debug("openai not installed. GitHub Models support unavailable.")

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.debug("google-generativeai not installed. Gemini support unavailable.")


class NeuronInterpreter:
    """Interprets SAE neurons using LLM with max/zero-activating examples.

    Supports both GitHub Models (GPT-4o) and Google Gemini APIs.

    Parameters
    ----------
    provider : str, optional
        LLM provider: "github_models" or "gemini" (default: auto-detect from available APIs)
    api_key : str, optional
        API key for the provider. If not provided, reads from environment.
        - GitHub Models: GITHUB_TOKEN env variable
        - Gemini: GOOGLE_API_KEY env variable
    model_name : str, optional
        Model to use (default: depends on provider)
        - GitHub Models: "gpt-4o" (via Azure inference)
        - Gemini: "gemini-2.0-flash"
    """

    def __init__(
        self,
        provider: Optional[Literal["github_models", "gemini"]] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        # Auto-detect provider if not specified
        if provider is None:
            if HAS_GITHUB_MODELS and os.environ.get("GITHUB_TOKEN"):
                provider = "github_models"
            elif HAS_GEMINI:
                provider = "gemini"
            else:
                raise ValueError(
                    "No LLM provider available. Install one of:\n"
                    "  - openai (for GitHub Models): pip install openai\n"
                    "  - google-generativeai (for Gemini): pip install google-generativeai"
                )

        self.provider = provider
        self._client = None

        # Set up API key and model name based on provider
        if provider == "github_models":
            if not HAS_GITHUB_MODELS:
                raise ImportError(
                    "openai library required for GitHub Models. "
                    "Install with: pip install openai"
                )
            self.api_key = api_key or os.environ.get("GITHUB_TOKEN")
            if not self.api_key:
                raise ValueError(
                    "GITHUB_TOKEN not provided and not in environment variables. "
                    "Set it: export GITHUB_TOKEN=<your_github_pat>"
                )
            self.model_name = model_name or "gpt-4o"

        elif provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError(
                    "google-generativeai library required for Gemini. "
                    "Install with: pip install google-generativeai"
                )
            self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "GOOGLE_API_KEY not provided and not in environment variables. "
                    "Get a free key from: https://aistudio.google.com/app/apikey"
                )
            self.model_name = model_name or "gemini-2.0-flash"
        else:
            raise ValueError(
                f"Unknown provider: {provider}. Use 'github_models' or 'gemini'"
            )

    @property
    def client(self):
        """Lazy-load API client to avoid import errors if not installed."""
        if self._client is None:
            if self.provider == "github_models":
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://models.inference.ai.azure.com",
                )
            elif self.provider == "gemini":
                genai.configure(api_key=self.api_key)
                self._client = genai
        return self._client

    def _format_examples(self, items: list[dict], max_items: int = 5) -> str:
        """Format item metadata for LLM input.

        Parameters
        ----------
        items : list[dict]
            List of items with metadata (id, name, category, tags, etc)
        max_items : int
            Maximum items to include

        Returns
        -------
        str
            Formatted text for prompt
        """
        if not items:
            return "No examples available."

        lines = []
        for i, item in enumerate(items[:max_items], 1):
            # Build item description
            parts = []
            if "name" in item:
                parts.append(f"Name: {item['name']}")
            if "category" in item:
                parts.append(f"Category: {item['category']}")
            if "tags" in item and item["tags"]:
                tags = item["tags"][:3]  # Top 3 tags
                parts.append(f"Tags: {', '.join(tags)}")
            if "review_keywords" in item and item["review_keywords"]:
                keywords = item["review_keywords"][:3]
                parts.append(f"Common review keywords: {', '.join(keywords)}")
            if "avg_rating" in item:
                parts.append(f"Avg rating: {item['avg_rating']:.1f}")

            text = " | ".join(parts) if parts else f"Item {item.get('id', 'unknown')}"
            lines.append(f"{i}. {text}")

        return "\n".join(lines)

    def prepare_neuron_prompt(
        self,
        neuron_idx: int,
        max_activating: list[dict],
        zero_activating: list[dict],
    ) -> str:
        """Prepare the LLM prompt for a neuron without calling the API.

        This allows the notebook to capture and display prompts for evaluation
        without duplicating the prompt construction logic.

        Parameters
        ----------
        neuron_idx : int
            Index of the neuron
        max_activating : list[dict]
            Items that maximally activate this neuron
        zero_activating : list[dict]
            Items that don't activate this neuron

        Returns
        -------
        str
            The full LLM prompt text
        """
        formatted_max = self._format_examples(max_activating, max_items=5)
        formatted_zero = self._format_examples(zero_activating, max_items=5)

        prompt = f"""You are a meticulous recommender systems researcher conducting an important investigation into a certain neuron in a recommendation model trained on Point-of-Interest (POI) datasets. Your task is to figure out what sort of behaviour this neuron is responsible for – namely, on what general concepts, features, themes, categories or contexts does this neuron fire?

Here's how you'll complete the task:

INPUT DESCRIPTION: You will be given two inputs: 1) Max Activating Examples and 2) Zero Activating Examples.
1. You will be given several examples of POIs (with their categories, tags, or typical review keywords) that activate the neuron. This means there is some feature, theme, or concept in these POIs that 'excites' this neuron.
2. You will also be given several examples of POIs that don't activate the neuron. This means the feature or concept is not present in these POIs.

OUTPUT DESCRIPTION: Given the inputs provided, complete the following tasks.
1. Based on the MAX ACTIVATING EXAMPLES provided, write down potential topics, concepts, categories, or contextual themes that they share in common. Give higher weight to concepts more present/prominent in examples with higher activations.
2. Based on the ZERO ACTIVATING EXAMPLES, rule out any of the topics/concepts/features listed above that are in the zero-activating examples. Systematically go through your list above.
3. Based on the above two steps, perform a thorough analysis of which feature, concept or topic, at what level of granularity, is likely to activate this neuron. Use Occam's razor, as long as it fits the provided evidence. Be highly rational and analytical here.
4. Based on step 3, summarise this concept in 1-8 words, in the form FINAL: <explanation>. Do NOT return anything after these 1-8 words.

Max-activating examples for neuron {neuron_idx}:
{formatted_max}

Zero-activating examples for neuron {neuron_idx}:
{formatted_zero}
"""
        return prompt

    def label_neuron(
        self,
        neuron_idx: int,
        max_activating: list[dict],
        zero_activating: list[dict],
        max_attempts: int = 3,
        rate_limit_delay: float = 0.5,
    ) -> Optional[str]:
        """Label a single neuron using LLM.

        Phase 1: Interpreter Prompt with max/zero-activating examples.

        Parameters
        ----------
        neuron_idx : int
            Index of the neuron
        max_activating : list[dict]
            Items that maximally activate this neuron
        zero_activating : list[dict]
            Items that don't activate this neuron
        max_attempts : int
            Number of retries if parsing fails
        rate_limit_delay : float
            Delay between attempts in seconds

        Returns
        -------
        str or None
            The assigned label (1-8 words), or None if labeling failed
        """

        # Use the reusable prompt generation method
        prompt = self.prepare_neuron_prompt(neuron_idx, max_activating, zero_activating)

        for attempt in range(max_attempts):
            try:
                response_text = None

                if self.provider == "github_models":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=400,  # Increased from 100 to allow full analysis + FINAL label
                    )
                    response_text = response.choices[0].message.content.strip()

                elif self.provider == "gemini":
                    model = self.client.GenerativeModel(self.model_name)
                    response = model.generate_content(prompt)
                    response_text = response.text.strip()

                # Parse response for "FINAL: <label>" format
                # Try exact match first (response includes FINAL: line)
                match = re.search(
                    r"FINAL:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE
                )
                if match:
                    label = match.group(1).strip()
                    # Clean up any markdown formatting
                    label = re.sub(
                        r"\*\*|\*|__", "", label
                    )  # Remove bold/italic markers
                    label = label.split("\n")[0]  # Take only first line if multiline
                    if label and len(label.split()) <= 8:
                        logger.info(f"  Neuron {neuron_idx}: '{label}'")
                        return label

                # Fallback: if response was truncated and doesn't have FINAL:,
                # try to extract a meaningful concept from available text
                # Look for lines that seem like conclusions (after numbered steps)
                lines = response_text.split("\n")
                for i, line in enumerate(lines):
                    # Skip step headers
                    if re.match(r"^\d+\.|^#{1,3}\s", line):
                        continue
                    # Get non-empty, substantial lines
                    if line.strip() and len(line.split()) >= 2 and len(line) > 20:
                        # Likely a conclusion or finding
                        candidate = re.sub(r"\*\*|\*|__|-|•", "", line).strip()
                        if len(candidate.split()) <= 8:
                            logger.info(
                                f"  Neuron {neuron_idx}: '{candidate}' (extracted from analysis)"
                            )
                            return candidate

                logger.warning(
                    f"Neuron {neuron_idx}, attempt {attempt+1}: "
                    f"Could not parse response: {response_text[:100] if response_text else 'No response'}"
                )

            except Exception as e:
                logger.warning(
                    f"Neuron {neuron_idx}, attempt {attempt+1}: {self.provider} error: {e}"
                )

            # Rate limiting between attempts
            if attempt < max_attempts - 1:
                time.sleep(rate_limit_delay)

        return None

    def label_neurons_batch(
        self,
        activations: torch.Tensor,
        items_metadata: list[dict],
        item2index: dict,
        top_k: int = 5,
        bottom_k: int = 5,
        output_file: Optional[str] = None,
    ) -> dict[int, str]:
        """Label all neurons in batch.

        Parameters
        ----------
        activations : torch.Tensor
            SAE sparse activations of shape (n_items, n_neurons)
        items_metadata : list[dict]
            Metadata for each item (index corresponds to item2index)
        item2index : dict
            Mapping from business_id to index
        top_k : int
            Number of top-activating items to use
        bottom_k : int
            Number of zero-activating items to use
        output_file : str, optional
            File to save neuron labels

        Returns
        -------
        dict[int, str]
            Mapping from neuron_idx to label
        """

        logger.info(f"Labeling {activations.shape[1]} neurons...")

        labels = {}

        # Convert to numpy if torch tensor
        if isinstance(activations, torch.Tensor):
            activations = activations.detach().cpu().numpy()

        n_neurons = activations.shape[1]

        for neuron_idx in range(n_neurons):
            neuron_acts = activations[:, neuron_idx]

            # Get max and zero activating items
            top_indices = np.argsort(-neuron_acts)[:top_k]
            bottom_indices = np.argsort(neuron_acts)[:bottom_k]

            max_activating = [
                items_metadata[i] for i in top_indices if i < len(items_metadata)
            ]
            zero_activating = [
                items_metadata[i] for i in bottom_indices if i < len(items_metadata)
            ]

            label = self.label_neuron(neuron_idx, max_activating, zero_activating)

            if label:
                labels[neuron_idx] = label

        logger.info(f"Successfully labeled {len(labels)}/{n_neurons} neurons")

        # Save if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(labels, f, indent=2)
            logger.info(f"Saved labels to {output_path}")

        return labels


class SuperfeatureGenerator:
    """Generate hierarchical super-features from neuron labels.

    Supports both GitHub Models and Gemini APIs (same as NeuronInterpreter).

    Parameters
    ----------
    provider : str, optional
        LLM provider: "github_models" or "gemini"
    api_key : str, optional
        API key for the provider
    model_name : str, optional
        Model to use
    """

    def __init__(
        self,
        provider: Optional[Literal["github_models", "gemini"]] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.interpreter = NeuronInterpreter(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
        )

    def cluster_labels_by_similarity(
        self,
        labels: dict[int, str],
        similarity_threshold: float = 0.7,
    ) -> dict[str, list[tuple[int, str]]]:
        """Cluster neuron labels by semantic similarity.

        This uses a simple approach: labels that share keywords are clustered together.
        In a production system, you'd use sentence embeddings.

        Parameters
        ----------
        labels : dict[int, str]
            Mapping from neuron_idx to label
        similarity_threshold : float
            Similarity threshold for clustering (0-1)

        Returns
        -------
        dict[str, list]
            Mapping from cluster_name to list of (neuron_idx, label) tuples
        """

        # Simple keyword-based clustering
        clusters = {}
        used = set()

        for neuron_idx, label in labels.items():
            if neuron_idx in used:
                continue

            # Start new cluster with this label
            cluster_name = f"feature_family_{neuron_idx}"
            cluster_labels = [(neuron_idx, label)]

            # Find similar labels (match keywords)
            label_words = set(label.lower().split())

            for other_idx, other_label in labels.items():
                if other_idx in used or other_idx == neuron_idx:
                    continue

                other_words = set(other_label.lower().split())
                intersection = label_words & other_words
                union = label_words | other_words

                similarity = len(intersection) / len(union) if union else 0

                if similarity >= similarity_threshold:
                    cluster_labels.append((other_idx, other_label))
                    used.add(other_idx)

            clusters[cluster_name] = cluster_labels
            used.add(neuron_idx)

        logger.info(f"Created {len(clusters)} clusters from {len(labels)} labels")
        return clusters

    def generate_superlabels(
        self,
        clusters: dict[str, list[tuple[int, str]]],
        output_file: Optional[str] = None,
    ) -> dict[str, str]:
        """Generate super-labels for clusters.

        Phase 2: Superfeature Prompt with multiple related labels.

        Parameters
        ----------
        clusters : dict[str, list]
            Mapping from cluster_name to list of (neuron_idx, label) tuples
        output_file : str, optional
            File to save superlabels

        Returns
        -------
        dict[str, str]
            Mapping from cluster_name to superlabel
        """

        superlabels = {}

        logger.info(f"Generating superlabels for {len(clusters)} clusters...")

        for cluster_name, items in clusters.items():
            labels_list = [label for _, label in items]

            if len(labels_list) == 1:
                # Single label, use as superlabel
                superlabels[cluster_name] = labels_list[0]
                continue

            prompt = f"""You are a recommender systems expert. Here is a group of closely related semantic labels that were assigned to neurons with a highly similar activation pattern in a POI recommendation system. They form a "feature family".

List of labels to synthesize:
{chr(10).join(f'- {label}' for label in labels_list)}

YOUR TASK:
Find a more abstract, overarching "super-label" that represents the common parent concept for all these child labels. The result should describe an overarching user preference or type of location behavior (e.g., if the labels are "Running", "Gym", and "Yoga", the overarching concept is "Sports and physical activity").

OUTPUT:
Return ONLY the overarching super-label of length 1 to 5 words in the following format:
SUPERLABEL: <super_label>
"""

            try:
                response_text = None

                if self.interpreter.provider == "github_models":
                    response = self.interpreter.client.chat.completions.create(
                        model=self.interpreter.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=50,
                    )
                    response_text = response.choices[0].message.content.strip()

                elif self.interpreter.provider == "gemini":
                    model = self.interpreter.client.GenerativeModel(
                        self.interpreter.model_name
                    )
                    response = model.generate_content(prompt)
                    response_text = response.text.strip()

                match = re.search(r"SUPERLABEL:\s*(.+?)(?:\n|$)", response_text)
                if match:
                    superlabel = match.group(1).strip()
                    if superlabel and len(superlabel.split()) <= 5:
                        superlabels[cluster_name] = superlabel
                        logger.info(
                            f"  {cluster_name}: '{superlabel}' (from {len(labels_list)} labels)"
                        )
                        continue

                logger.warning(
                    f"Failed to parse superlabel for {cluster_name}, using first label"
                )
                superlabels[cluster_name] = labels_list[0]

            except Exception as e:
                logger.warning(
                    f"Error generating superlabel for {cluster_name} ({self.interpreter.provider}): {e}"
                )
                superlabels[cluster_name] = labels_list[0]

        # Save if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(superlabels, f, indent=2)
            logger.info(f"Saved superlabels to {output_path}")

        return superlabels

    def save_interpretation_hierarchy(
        self,
        neuron_labels: dict[int, str],
        superlabels: dict[str, str],
        clusters: dict[str, list],
        output_file: str,
    ) -> None:
        """Save complete interpretation hierarchy to file.

        Parameters
        ----------
        neuron_labels : dict[int, str]
            Neuron idx to label mapping
        superlabels : dict[str, str]
            Cluster name to superlabel mapping
        clusters : dict[str, list]
            Cluster name to neuron items mapping
        output_file : str
            Output file path
        """

        hierarchy = {
            "neuron_labels": neuron_labels,
            "superlabels": superlabels,
            "clusters": {
                cluster_name: [
                    {"neuron_idx": idx, "label": label} for idx, label in items
                ]
                for cluster_name, items in clusters.items()
            },
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(hierarchy, f, indent=2)

        logger.info(f"Saved interpretation hierarchy to {output_path}")
