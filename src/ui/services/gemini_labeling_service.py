#!/usr/bin/env python3
"""
Gemini-powered neuron labeling service with verification workflow.

This service uses Google's Gemini API to generate intelligent labels for neurons
based on their activation patterns and top activating items. It includes a
verification workflow to prevent expensive API calls on untrusted data.

Supports multiple credential sources:
1. Direct Gemini API key (GEMINI_API_KEY env var) - simplest
2. Vertex AI credentials (GOOGLE_CLOUD_PROJECT) - if already set up
3. Google Application Default Credentials - from `gcloud auth`

Usage:
    from src.ui.services.gemini_labeling_service import GeminiLabelingService
    
    # Auto-detects credentials:
    labeler = GeminiLabelingService()
    
    # Or explicitly provide API key:
    labeler = GeminiLabelingService(api_key="your-key")
    
    labels = labeler.label_neurons_with_verification(neurons_data)
"""

import json
import logging
from typing import Dict, List, Optional, Tuple

try:
    import google.generativeai as genai

    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import vertexai

    HAS_VERTEX_AI = True
except ImportError:
    HAS_VERTEX_AI = False

from .secrets_helper import (
    get_gemini_api_key,
    get_gcp_project,
    get_gcp_region,
)

logger = logging.getLogger(__name__)


class GeminiLabelingService:
    """Service for AI-powered neuron labeling with verification."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        """
        Initialize Gemini labeling service.

        Supports multiple credential sources (in priority order):
        1. Explicit api_key parameter
        2. GEMINI_API_KEY environment variable
        3. Vertex AI credentials (from google_api_key or Application Default)
        4. Google Application Default Credentials

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model to use (default: gemini-1.5-flash for cost efficiency)
        """
        if not HAS_GEMINI:
            raise ImportError("google-generativeai not installed")

        self.api_key = api_key or get_gemini_api_key()
        self.use_vertex_ai = False

        # Try Gemini API first (if key provided)
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model)
                self.model_name = model
                logger.info(f"✅ Initialized Gemini with direct API key using {model}")
                return
            except Exception as e:
                logger.warning(f"Failed to init Gemini with API key: {e}")
                self.api_key = None

        # Try Vertex AI (uses existing Google Cloud credentials)
        if HAS_VERTEX_AI:
            try:
                # Initialize Vertex AI (uses ADC or google_api_key)
                project_id = get_gcp_project()
                region = get_gcp_region()

                if project_id:
                    vertexai.init(project=project_id, location=region)

                    # Import and use Vertex AI Generative models
                    from vertexai.generative_models import GenerativeModel

                    self.model = GenerativeModel(model)
                    self.model_name = model
                    self.use_vertex_ai = True
                    msg = f"✅ Initialized Gemini via Vertex AI ({project_id}) using {model}"
                    logger.info(msg)
                    return
            except Exception as e:
                logger.debug(f"Failed to init Vertex AI: {e}")

        # Fallback: Try with Application Default Credentials and genai
        try:
            genai.configure(api_key=None)  # Uses Application Default Credentials
            self.model = genai.GenerativeModel(model)
            self.model_name = model
            msg = f"✅ Initialized Gemini with Application Default Credentials using {model}"
            logger.info(msg)
            return
        except Exception as e:
            logger.debug(f"Failed with Application Default Credentials: {e}")

        raise ValueError(
            "Could not initialize Gemini. Please provide one of:\n"
            "  1. GEMINI_API_KEY environment variable\n"
            "  2. GOOGLE_CLOUD_PROJECT for Vertex AI\n"
            "  3. Application Default Credentials (gcloud auth application-default login)"
        )

    def generate_label(
        self,
        neuron_id: int,
        activation_pattern: Dict,
        top_items: List[Dict],
        max_retries: int = 3,
    ) -> Tuple[str, str, float]:
        """
        Generate a label for a single neuron using Gemini.

        Args:
            neuron_id: Neuron identifier
            activation_pattern: Dict with activation statistics
            top_items: List of top-10 activating businesses/items
            max_retries: Number of retries on API failure

        Returns:
            (label, description, confidence_score)
        """

        # Build context about the neuron
        context = self._build_neuron_context(neuron_id, top_items, activation_pattern)

        prompt = f"""Analyze this neuron's activation pattern and generate a label.

NEURON ID: {neuron_id}

CONTEXT:
{context}

Based on the top activating items/businesses above, provide:
1. A SHORT LABEL (2-4 words, e.g., "Italian Restaurants", "Coffee Shops")
2. A DESCRIPTION (1-2 sentences explaining what this neuron represents)
3. Your CONFIDENCE (0-100 scale)

Format your response as JSON:
{{
    "label": "...",
    "description": "...",
    "confidence": 85
}}"""

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)

                # Extract JSON from response (works with both Gemini and Vertex AI)
                text = response.text if hasattr(response, "text") else str(response)
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = text[start:end]
                    result = json.loads(json_str)

                    return (
                        result.get("label", "Unknown").strip(),
                        result.get("description", "No description").strip(),
                        float(result.get("confidence", 50)) / 100.0,  # Normalize to 0-1
                    )
                else:
                    msg = f"Could not parse JSON response for neuron {neuron_id}"
                    logger.warning(msg)
                    return ("Unknown", "Could not generate label", 0.0)

            except Exception as e:
                logger.debug(f"Gemini API attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise

        return ("Unknown", "API error", 0.0)

    def label_neurons_with_verification(
        self,
        neurons_data: Dict,
        confidence_threshold: float = 0.7,
        max_labels: int = 10,
    ) -> Dict[str, Dict]:
        """
        Label neurons with verification workflow.

        This is the main entry point that:
        1. Scores neurons by quality metrics
        2. Asks for user verification before labeling top neurons
        3. Labels only verified neurons to save API costs

        Args:
            neurons_data: Dict with neuron info (should have 'neurons' key)
            confidence_threshold: Only ask for verification if confidence >= this
            max_labels: Maximum number of neurons to label per session

        Returns:
            Dict mapping neuron_id to {label, description, confidence, verified}
        """

        msg = f"🧠 Preparing {len(neurons_data)} neurons for verification...\n"
        logger.info(msg)

        # Score and rank neurons by quality
        scored_neurons = self._score_neurons(neurons_data)

        # Select top neurons for labeling
        candidates = [
            n for n in scored_neurons if n["quality_score"] >= confidence_threshold
        ][:max_labels]

        msg = f"📊 Found {len(candidates)} candidate neurons for labeling\n"

        if not candidates:
            logger.warning("⚠️  No neurons meet confidence threshold")
            return {}

        # Show verification UI
        labels = self._show_verification_ui(candidates)

        return labels

    def _score_neurons(self, neurons_data: Dict) -> List[Dict]:
        """
        Score neurons by quality (how informative their activations are).

        Quality factors:
        - Sparsity: Neurons that activate on specific items score higher
        - Coherence: Top items being similar (by name/category) score higher
        - Distinctiveness: Rare activation patterns score higher
        """

        scored = []
        for neuron_id, neuron_info in neurons_data.items():
            score = 0.0

            # Check if has top items
            top_items = neuron_info.get("top_items", [])
            if not top_items:
                continue

            # Sparsity bonus
            score += min(
                len([i for i in top_items if i.get("activation", 0) > 0.1]), 10
            )

            # Item coherence (rough check: common words in names)
            names = [str(i.get("name", "")).lower() for i in top_items]
            common_words = self._find_common_words(names)
            score += len(common_words) * 0.5

            # High activation bonus
            max_activation = max((i.get("activation", 0) for i in top_items), default=0)
            score += max_activation * 20

            scored.append(
                {
                    "neuron_id": neuron_id,
                    "quality_score": score / 30.0,  # Normalize to 0-1
                    "top_items": top_items,
                    "activation_data": neuron_info,
                }
            )

        # Sort by score
        scored.sort(key=lambda x: x["quality_score"], reverse=True)
        return scored

    def _find_common_words(self, names: List[str], min_length: int = 3) -> List[str]:
        """Find common words appearing in multiple names."""
        if not names:
            return []

        # Simple word frequency
        word_freq = {}
        for name in names:
            words = set(word for word in name.split() if len(word) >= min_length)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        return [w for w, count in word_freq.items() if count > 1]

    def _build_neuron_context(
        self,
        neuron_id: int,
        top_items: List[Dict],
        activation_pattern: Dict,
    ) -> str:
        """Build context string for prompt."""

        lines = ["Top 5 Activating Items:"]
        for i, item in enumerate(top_items[:5], 1):
            name = item.get("name", "Unknown")
            activation = item.get("activation", 0)
            lines.append(f"  {i}. {name} (activation: {activation:.2f})")

        # Add category info if available
        categories = set()
        for item in top_items[:5]:
            # Try to extract category from name or metadata
            cat = item.get("category")
            if cat:
                categories.add(cat)

        if categories:
            lines.append(f"\nCommon Categories: {', '.join(categories)}")

        return "\n".join(lines)

    def _show_verification_ui(self, candidates: List[Dict]) -> Dict[str, Dict]:
        """
        Show verification UI for human approval before labeling.

        This prevents unnecessary API calls on low-confidence neurons.
        """

        logger.info("👤 Human Verification Required\n")
        logger.info("=" * 80)
        logger.info(f"Ready to request {len(candidates)} neuron labels from Gemini.")
        logger.info(f"Estimated API cost: {len(candidates)} API calls\n")

        logger.info("Preview of candidates:\n")

        labels = {}
        for i, neuron in enumerate(candidates[:5], 1):  # Show first 5
            top_items_preview = neuron["top_items"][:3]
            logger.info(
                f"{i}. Neuron {neuron['neuron_id']} (Quality: {neuron['quality_score']:.1%})"
            )
            for item in top_items_preview:
                logger.info(f"   - {item.get('name', 'Unknown')}")
            logger.info("")

        if len(candidates) > 5:
            logger.info(f"... and {len(candidates) - 5} more candidates\n")

        logger.info("=" * 80)
        logger.info("\n⚠️  VERIFICATION REQUIRED:")
        logger.info(
            "  Option 1: Approve all candidates → run: labeler.verify_and_label_all()"
        )
        logger.info(
            "  Option 2: Select specific neurons → run: labeler.label_specific([0, 3, 5])"
        )
        logger.info("  Option 3: Cancel → return without labeling")
        logger.info("\nNote: Gemini API will be called for each labeled neuron.")
        logger.info("=" * 80)

        return labels

    def verify_and_label_all(self, candidates: List[Dict]) -> Dict[str, Dict]:
        """
        Label all candidate neurons (requires explicit approval).

        This is a safe way to ensure the user understands API costs.
        """

        logger.info(f"\n✅ Proceeding to label {len(candidates)} neurons...\n")

        labels = {}
        for i, neuron in enumerate(candidates, 1):
            try:
                logger.info(
                    f"[{i}/{len(candidates)}] Labeling neuron {neuron['neuron_id']}..."
                )

                label, desc, confidence = self.generate_label(
                    neuron["neuron_id"],
                    neuron["activation_data"],
                    neuron["top_items"],
                )

                labels[str(neuron["neuron_id"])] = {
                    "label": label,
                    "description": desc,
                    "confidence": confidence,
                    "verified": True,
                }

                logger.info(f"  → {label} (confidence: {confidence:.1%})")

            except Exception as e:
                logger.error(f"Failed to label neuron {neuron['neuron_id']}: {e}")
                labels[str(neuron["neuron_id"])] = {
                    "label": "Error",
                    "description": f"Failed to generate label: {str(e)}",
                    "confidence": 0.0,
                    "verified": False,
                }

        logger.info(f"\n✅ Labeled {len(labels)} neurons")
        return labels

    def label_specific(self, candidate_indices: List[int]) -> Dict[str, Dict]:
        """
        Label only specific candidates from the verification UI.

        Args:
            candidate_indices: Indices of candidates to label from verification list

        Returns:
            Dict of labels for selected neurons
        """

        logger.info(
            f"✅ Proceeding to label {len(candidate_indices)} selected neurons...\n"
        )
        return {}  # Implement selection logic


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    try:
        service = GeminiLabelingService()
        logger.info("Gemini service initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
