"""Inference service for steering and recommendation generation."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn.functional as F
import numpy as np

from src.models.collaborative_filtering import ELSA
from src.models.sparse_autoencoder import TopKSAE

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Core recommendation engine with steering support.

    Responsibilities:
    - Load and cache models
    - Encode users from interaction history
    - Apply steering via SAE decoder basis vectors
    - Generate top-k recommendations with attribution
    """

    def __init__(
        self,
        elsa_checkpoint_path: Path,
        sae_checkpoint_path: Path,
        config: Optional[Dict] = None,
    ):
        """
        Initialize inference service and load models.

        Args:
            elsa_checkpoint_path: Path to elsa_best.pt
            sae_checkpoint_path: Path to sae_best.pt
            config: Configuration dict with keys:
                - latent_dim: ELSA latent dimension (optional, will be read from checkpoint)
                - k: SAE sparsity level (default 32)
                - width_ratio: SAE hidden dim ratio (default 4)
                - steering_alpha: interpolation strength (default 0.3)
                - device: 'cpu' or 'cuda' (default 'cpu')

            Note: n_items is read from the ELSA checkpoint metadata, NOT from config.
                  The checkpoint metadata is the definitive source of truth.
        """
        self.config = config or {}
        self.device = self.config.get("device", "cpu")
        self.alpha = self.config.get("steering_alpha", 0.3)

        # Model hyperparameters
        # n_items is NOT read from config - it will be loaded from checkpoint metadata
        self.n_items = None  # Will be set after loading ELSA
        self.latent_dim = self.config.get("latent_dim", 512)
        self.k = self.config.get("k", 32)
        self.width_ratio = self.config.get("width_ratio", 4)

        logger.info(f"Loading models on device: {self.device}")
        logger.info(f"Model config: latent_dim={self.latent_dim}, k={self.k}")

        self.elsa = self._load_elsa(elsa_checkpoint_path)
        # Now n_items is set from checkpoint
        logger.info(
            f"Model dimensions from checkpoint: n_items={self.n_items}, latent_dim={self.latent_dim}"
        )

        self.sae = self._load_sae(sae_checkpoint_path)

        # Per-session caches (cleared on user selection)
        self.user_latents = {}  # {user_id: z_u tensor}
        self.user_sliders = {}  # {user_id: {neuron_idx: value}}

        # Item score cache for current user
        self._item_cache = None  # Tuple of (scores, user_id)

        # Latency tracking for performance monitoring
        self.latency_ms = []  # List of inference times in milliseconds

        logger.info("✅ Inference service ready")

    def _load_elsa(self, ckpt_path: Path) -> ELSA:
        """Load ELSA model from checkpoint, using metadata from the checkpoint.

        The checkpoint metadata (n_items, latent_dim) is the definitive source of truth
        for model instantiation. This ensures compatibility regardless of how the data
        is filtered or structured on the inference machine.
        """
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"ELSA checkpoint not found: {ckpt_path}")

        logger.info(f"Loading ELSA from {ckpt_path}")

        # Load checkpoint (PyTorch 2.6+ requires special handling for numpy types)
        # These are our own trusted checkpoints, so weights_only=False is safe
        try:
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        # Extract metadata from checkpoint (definitive source of truth)
        metadata = checkpoint.get("metadata", {})
        n_items = metadata.get("n_items")
        latent_dim = metadata.get("latent_dim")

        if n_items is None or latent_dim is None:
            # Fallback: try to infer from state_dict shape (for old checkpoints)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                if "A" in state_dict:
                    n_items, latent_dim = state_dict["A"].shape
                    logger.warning(
                        f"No metadata in checkpoint! Inferred n_items={n_items}, "
                        f"latent_dim={latent_dim} from state_dict. "
                        f"This is a fallback for old checkpoints."
                    )

        if n_items is None or latent_dim is None:
            raise RuntimeError(
                f"Cannot determine model dimensions. Did not find metadata in checkpoint "
                f"and could not infer from state_dict. n_items={n_items}, latent_dim={latent_dim}"
            )

        logger.info(f"Checkpoint metadata: n_items={n_items}, latent_dim={latent_dim}")

        # Store n_items in self for use in other methods
        self.n_items = n_items
        self.latent_dim = latent_dim

        # Instantiate model with correct dimensions from checkpoint
        model = ELSA(n_items, latent_dim=latent_dim)
        model = model.to(self.device)
        model.eval()

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback if checkpoint is just state dict
            model.load_state_dict(checkpoint)

        logger.info(f"✅ ELSA loaded: {ckpt_path.name} (n_items={n_items})")
        return model

    def _load_sae(self, ckpt_path: Path) -> TopKSAE:
        """Load TopK SAE model from checkpoint, using metadata from the checkpoint when available.

        Falls back to config values if metadata is not available (for backward compatibility).
        """
        ckpt_path = Path(ckpt_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

        logger.info(f"Loading SAE from {ckpt_path}")

        # Load checkpoint (PyTorch 2.6+ requires special handling for numpy types)
        # These are our own trusted checkpoints, so weights_only=False is safe
        try:
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=False
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

        # Extract metadata from checkpoint if available
        metadata = checkpoint.get("metadata", {})
        k = metadata.get("k", self.k)
        width_ratio = metadata.get("width_ratio", self.width_ratio)
        latent_dim = metadata.get("latent_dim", self.latent_dim)

        if metadata:
            logger.info(
                f"Checkpoint metadata: k={k}, width_ratio={width_ratio}, latent_dim={latent_dim}"
            )
        else:
            logger.info(
                f"Using config values: k={k}, width_ratio={width_ratio}, latent_dim={latent_dim}"
            )

        # Calculate hidden_dim from width_ratio
        sae_hidden_dim = width_ratio * latent_dim

        # Instantiate model
        model = TopKSAE(
            input_dim=latent_dim,
            hidden_dim=sae_hidden_dim,
            k=k,
            l1_coef=self.config.get("l1_coef", 3e-4),
        )
        model = model.to(self.device)
        model.eval()

        # Load state dict
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Fallback if checkpoint is just state dict
            model.load_state_dict(checkpoint)

        logger.info(f"✅ SAE loaded: {ckpt_path.name}")
        return model

    def encode_user(self, user_id: str, user_interactions_csr) -> torch.Tensor:
        """
        Encode a user from interactions into latent space using ELSA.

        Args:
            user_id: Yelp user ID
            user_interactions_csr: CSR matrix of user interactions
                (typically: row vector of 0/1 indicating liked POIs)

        Returns:
            Latent embedding vector (cached for steering)
        """
        import time

        start_time = time.time()

        # Validate input with detailed logging
        logger.debug(
            f"encode_user called with user_id={user_id}, csr_type={type(user_interactions_csr)}, csr_is_none={user_interactions_csr is None}"
        )

        if user_interactions_csr is None:
            error_msg = f"user_interactions_csr cannot be None for user {user_id}. This usually means POI indices retrieval or CSR matrix creation failed."
            logger.error(f"CRITICAL: {error_msg}")
            logger.error(
                f"Debug info: Type={type(user_interactions_csr)}, Value={user_interactions_csr}"
            )
            raise ValueError(error_msg)

        logger.debug(
            f"Encoding user {user_id}. user_interactions type: {type(user_interactions_csr)}"
        )

        if not hasattr(user_interactions_csr, "toarray"):
            logger.warning(
                f"user_interactions_csr does not have toarray method. Type: {type(user_interactions_csr)}"
            )

        with torch.no_grad():
            # Convert CSR to dense tensor if needed
            if hasattr(user_interactions_csr, "toarray"):
                # It's a sparse matrix - convert to dense
                try:
                    dense_array = user_interactions_csr.toarray()
                    logger.debug(
                        f"Converted CSR to dense array: shape={dense_array.shape}, dtype={dense_array.dtype}"
                    )
                except Exception as e:
                    logger.error(f"Failed to convert CSR matrix to dense: {e}")
                    logger.error(
                        f"CSR matrix debug: shape={user_interactions_csr.shape}, nnz={user_interactions_csr.nnz}, dtype={user_interactions_csr.dtype}"
                    )
                    raise

                x_dense = torch.tensor(
                    dense_array,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                # Already dense
                x_dense = torch.tensor(
                    user_interactions_csr, dtype=torch.float32, device=self.device
                )

            logger.debug(
                f"Created tensor: shape={x_dense.shape}, dtype={x_dense.dtype}"
            )

            # Ensure 2D (batch_size, n_items)
            if x_dense.dim() == 1:
                x_dense = x_dense.unsqueeze(0)

            # Forward through ELSA encoder
            user_z = self.elsa.encode(x_dense)  # Shape: (batch, latent_dim)

            # Squeeze if single user
            if user_z.shape[0] == 1:
                user_z = user_z.squeeze(0)

        self.user_latents[user_id] = user_z.detach().cpu()
        self.user_sliders[user_id] = {}

        elapsed_ms = (time.time() - start_time) * 1000
        self.latency_ms.append(elapsed_ms)
        logger.debug(
            f"Encoded user {user_id}: z_shape={user_z.shape}, latency={elapsed_ms:.2f}ms"
        )
        return user_z.detach().cpu()

    def get_top_activations(self, user_latent: torch.Tensor, k: int = 10) -> List[Dict]:
        """
        Get top-k active features for a user's latent vector.

        Features are ranked by absolute activation strength.

        Returns:
            List of dicts with keys:
                - neuron_idx: int
                - activation: float (absolute value)
                - label: str (requires LabelingService)
        """
        # Pass through SAE encoder to get feature activations
        with torch.no_grad():
            h = self.sae.encode(user_latent.unsqueeze(0)).squeeze()

        # Get top-k by absolute activation
        topk_vals, topk_idx = torch.topk(h.abs(), k=min(k, h.shape[0]))

        result = []
        for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
            result.append(
                {
                    "neuron_idx": idx,
                    "activation": float(val),
                    "label": f"Feature {idx}",  # TODO: Get from LabelingService
                }
            )

        return result

    def steer_and_recommend(
        self,
        user_id: str,
        steering_overrides: Optional[Dict[int, float]] = None,
        top_k: int = 20,
    ) -> Dict:
        """
        Apply steering and generate recommendations.

        **Algorithm**:

        1. Retrieve cached user latent: z_u = user_latents[user_id]
        2. Encode to sparse features: h_u = SAE.encode(z_u)
        3. Apply steering overrides to h_u (set feature activations)
        4. Decode: z_steered = SAE.decode(h_steered)
        5. Interpolate: z_final = (1-α)·z_u + α·z_steered
        6. Score items: scores = z_final @ ELSA.A_norm.T
        7. Return top-k with neuron attribution

        Args:
            user_id: Yelp user ID (must be encoded first)
            steering_overrides: Dict mapping neuron_idx -> slider value ∈ [-1, 2]
            top_k: Number of recommendations to return

        Returns:
            Dict with keys:
                - recommendations: List of dicts (poi_idx, score, neurons)
                - steering_applied: The steering_overrides used
                - alpha: Interpolation strength applied
                - latency_ms: Inference time in milliseconds
        """
        import time

        start_time = time.time()
        steering_overrides = steering_overrides or {}

        # Validate user is encoded
        if user_id not in self.user_latents:
            raise ValueError(
                f"User {user_id} not encoded yet. Call encode_user() first."
            )

        user_z = self.user_latents[user_id].to(self.device)
        self.user_sliders[user_id] = steering_overrides

        with torch.no_grad():
            # === Step 1-2: Get sparse features for user ===
            h_user = self.sae.encode(user_z.unsqueeze(0))  # (1, hidden_dim)
            h_steered = h_user.clone()

            # === Step 3: Apply steering overrides ===
            for neuron_idx, slider_value in steering_overrides.items():
                if 0 <= neuron_idx < h_steered.shape[1]:
                    # Clamp slider to valid range
                    h_steered[0, neuron_idx] = torch.clamp(
                        torch.tensor(slider_value, device=self.device),
                        min=-1.0,
                        max=2.0,
                    )

            # === Step 4: Decode ===
            z_steered = self.sae.decode(h_steered).squeeze(0)  # (latent_dim,)

            # === Step 5: Interpolate ===
            z_final = (1.0 - self.alpha) * user_z + self.alpha * z_steered

            # === Step 6: Score items using ELSA ===
            # Score via ELSA's normalized item factors
            scores = z_final @ self.elsa._A_norm.T  # (n_items,)

            # === Step 7: Get top-k ===
            top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[0]))

        result = {
            "recommendations": [
                {
                    "poi_idx": int(idx.item()),
                    "score": float(score.item()),
                    "contributing_neurons": self._get_attribution(
                        h_steered.squeeze(0), idx.item()
                    ),
                }
                for score, idx in zip(top_scores, top_indices)
            ],
            "steering_applied": steering_overrides,
            "alpha": self.alpha,
            "latency_ms": (time.time() - start_time) * 1000,
        }

        # Track latency
        self.latency_ms.append(result["latency_ms"])

        # Cache for subsequent queries
        self._item_cache = (scores, user_id)

        return result

    def get_latency_stats(self) -> Dict[str, float]:
        """
        Get inference latency statistics in milliseconds.

        Returns:
            Dict with keys: mean, p50, p95, p99, max, min, count
        """
        if not self.latency_ms:
            return {
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "max": 0.0,
                "min": 0.0,
                "count": 0,
            }

        latencies = sorted(self.latency_ms)
        n = len(latencies)

        return {
            "mean": np.mean(latencies),
            "p50": latencies[int(n * 0.5)],
            "p95": latencies[int(n * 0.95)],
            "p99": latencies[int(n * 0.99)] if n >= 100 else latencies[-1],
            "max": max(latencies),
            "min": min(latencies),
            "count": len(latencies),
        }

    def _score_items(self, latent_batch: torch.Tensor) -> torch.Tensor:
        """
        Score all items given latent representation(s).

        Args:
            latent_batch: Tensor of shape (batch_size, latent_dim)

        Returns:
            Scores of shape (batch_size, n_items)
        """
        with torch.no_grad():
            scores = latent_batch @ self.elsa._A_norm.T  # (batch, n_items)

        return scores

    def _get_attribution(
        self, h_sparse: torch.Tensor, item_idx: int, top_k: int = 3
    ) -> List[Dict]:
        """
        Identify which sparse features (neurons) contributed to this recommendation.

        Args:
            h_sparse: Sparse feature code (or latent vector to encode)
            item_idx: Index of recommended item
            top_k: Number of top neurons to return

        Returns:
            List of dicts with neuron_idx and contribution score
        """
        # If h_sparse is latent (not sparse), encode it first
        if h_sparse.shape[0] != self.sae.encoder[0].out_features:
            # It's a latent vector, encode it
            with torch.no_grad():
                h_sparse = self.sae.encode(h_sparse.unsqueeze(0)).squeeze(0)

        # Get top-k neurons by absolute activation
        topk_vals, topk_idx = torch.topk(
            h_sparse.abs(), k=min(top_k, h_sparse.shape[0])
        )

        return [
            {
                "idx": int(idx.item()),
                "label": f"Feature {idx.item()}",
                "contribution": float(val.item()),
            }
            for idx, val in zip(topk_idx, topk_vals)
        ]

    def get_user_history(self, user_id: str) -> List[Dict]:
        """
        Get user's past interactions for reference display.

        Args:
            user_id: Yelp user ID

        Returns:
            List of dicts with past interactions (requires DataService)
        """
        # TODO: Query from DataService
        return []

    def clear_cache(self):
        """Clear all cached data (for memory efficiency or user reset)."""
        self.user_latents.clear()
        self.user_sliders.clear()
        self._item_cache = None
        logger.debug("Cache cleared")
