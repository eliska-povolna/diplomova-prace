"""Inference service for steering and recommendation generation."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

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
        labels: Optional[object] = None,
        data_service: Optional[object] = None,
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
            labels: LabelingService instance for neuron labels (optional)
            data_service: DataService instance for accessing user interaction data (optional)

            Note: n_items is read from the ELSA checkpoint metadata, NOT from config.
                  The checkpoint metadata is the definitive source of truth.
        """
        self.config = config or {}
        self.device = self.config.get("device", "cpu")
        self.alpha = self.config.get("steering_alpha", 0.3)
        self.labels = labels
        self.data_service = data_service

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

        # Baseline recommendations for position delta calculation
        self.baseline_recommendations = {}  # {user_id: {item_id: baseline_rank}}

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
            # Get label from LabelingService (or fallback to Feature N)
            label = f"Feature {idx}"
            if hasattr(self, "labels") and self.labels:
                label = self.labels.get_label(idx)

            result.append(
                {
                    "neuron_idx": idx,
                    "activation": float(val),
                    "label": label,
                }
            )

        return result

    def get_steered_neuron_activation(self, user_id: str, neuron_idx: int, slider_value: float) -> float:
        """
        Compute the actual activation value for a single neuron after steering is applied.
        
        Shows what the real activation will be after the steering interpolation.
        
        Args:
            user_id: User ID (must be encoded)
            neuron_idx: Index of neuron being steered
            slider_value: The slider value (-1 to +2)
        
        Returns:
            Float: The computed activation after steering + alpha interpolation
        """
        if user_id not in self.user_latents:
            raise ValueError(f"User {user_id} not encoded yet")
        
        user_z = self.user_latents[user_id].to(self.device)
        
        with torch.no_grad():
            # Step 1: Get baseline sparse features
            h_user = self.sae.encode(user_z.unsqueeze(0))  # (1, hidden_dim)
            h_steered = h_user.clone()
            
            # Step 2: Apply steering to this neuron
            if 0 <= neuron_idx < h_steered.shape[1]:
                h_steered[0, neuron_idx] = torch.clamp(
                    torch.tensor(slider_value, device=self.device),
                    min=-1.0,
                    max=2.0,
                )
            
            # Step 3: Decode steered features
            z_steered = self.sae.decode(h_steered).squeeze(0)
            
            # Step 4: Interpolate with original
            z_final = (1.0 - self.alpha) * user_z + self.alpha * z_steered
            
            # Step 5: Re-encode to get final activation
            h_final = self.sae.encode(z_final.unsqueeze(0)).squeeze()
            
            # Get the activation for this specific neuron
            if 0 <= neuron_idx < h_final.shape[0]:
                return float(h_final[neuron_idx].abs().item())
        
        return 0.0
    
    def get_steered_activations(self, user_id: str, steering_overrides: Dict[int, float], k: int = 10) -> List[Dict]:
        """
        Compute top-k activations AFTER applying steering.
        
        This shows what the actual feature activations will be after the user's steering is applied.
        
        Args:
            user_id: User ID (must be encoded)
            steering_overrides: Dict mapping neuron_idx -> slider value
            k: Number of top features to return
        
        Returns:
            List of dicts with keys:
                - neuron_idx: int
                - activation: float (after steering + alpha interpolation)
                - label: str
        """
        if user_id not in self.user_latents:
            raise ValueError(f"User {user_id} not encoded yet")
        
        steering_overrides = steering_overrides or {}
        if not steering_overrides:
            # No steering, just return baseline
            return self.get_top_activations(self.user_latents[user_id], k=k)
        
        user_z = self.user_latents[user_id].to(self.device)
        
        with torch.no_grad():
            # Step 1: Get baseline sparse features
            h_user = self.sae.encode(user_z.unsqueeze(0))  # (1, hidden_dim)
            h_steered = h_user.clone()
            
            # Step 2: Apply steering overrides
            for neuron_idx, slider_value in steering_overrides.items():
                if 0 <= neuron_idx < h_steered.shape[1]:
                    h_steered[0, neuron_idx] = torch.clamp(
                        torch.tensor(slider_value, device=self.device),
                        min=-1.0,
                        max=2.0,
                    )
            
            # Step 3: Decode steered features back to latent
            z_steered = self.sae.decode(h_steered).squeeze(0)
            
            # Step 4: Interpolate with original latent using alpha
            z_final = (1.0 - self.alpha) * user_z + self.alpha * z_steered
            
            # Step 5: Re-encode final latent to get NEW activations
            h_final = self.sae.encode(z_final.unsqueeze(0)).squeeze()
        
        # Get top-k by absolute activation of final features
        topk_vals, topk_idx = torch.topk(h_final.abs(), k=min(k, h_final.shape[0]))
        
        result = []
        for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
            label = f"Feature {idx}"
            if hasattr(self, "labels") and self.labels:
                label = self.labels.get_label(idx)
            
            result.append({
                "neuron_idx": idx,
                "activation": float(val),
                "label": label,
            })
        
        return result

    def get_user_steering(self, user_id: str) -> Dict[int, float]:
        """
        Get current SAE activation values for all neurons for a user.

        Returns a dict mapping neuron_idx to current activation strength.
        These values can be used as slider defaults in the steering UI.

        Args:
            user_id: User ID (must be encoded first via encode_user)

        Returns:
            Dict[int, float]: {neuron_idx: activation_value, ...}
                Neuron indices from 0 to sae_hidden_dim-1.
                Activation values are the raw sparse activations from SAE encoder.

        Raises:
            ValueError: If user hasn't been encoded yet
        """
        if user_id not in self.user_latents:
            raise ValueError(
                f"User {user_id} not encoded yet. Call encode_user() first."
            )

        user_z = self.user_latents[user_id]

        # Get all SAE activations
        with torch.no_grad():
            h = self.sae.encode(user_z.unsqueeze(0)).squeeze()

        # Return as dict: {neuron_idx: activation_value}
        # Use absolute values to represent magnitude (as per UI convention)
        steering_dict = {
            i: float(h[i].abs().item()) for i in range(h.shape[0])
        }

        return steering_dict

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
            List of dicts with neuron_idx, label, and activation value
        """
        # If h_sparse is latent (not sparse), encode it first
        if h_sparse.shape[0] != self.sae.hidden_dim:
            # It's a latent vector, encode it
            with torch.no_grad():
                h_sparse = self.sae.encode(h_sparse.unsqueeze(0)).squeeze(0)

        # Get top-k neurons by absolute activation
        topk_vals, topk_idx = torch.topk(
            h_sparse.abs(), k=min(top_k, h_sparse.shape[0])
        )

        result = []
        for idx, val in zip(topk_idx, topk_vals):
            neuron_idx = int(idx.item())
            activation_val = float(val.item())
            
            # Get label from LabelingService if available
            if self.labels:
                label = self.labels.get_label(neuron_idx)
            else:
                label = f"Feature {neuron_idx}"
            
            result.append({
                "idx": neuron_idx,
                "label": label,
                "activation": activation_val,
            })
        
        return result

    def get_user_history(self, user_id: str) -> List[int]:
        """
        Get user's past interactions for reference display.

        Args:
            user_id: Yelp user ID

        Returns:
            List of POI indices for past interactions (min_stars=1.0 filter applied to include all ratings)
        """
        if not self.data_service:
            logger.warning("DataService not available for user history")
            return []
        
        try:
            # Ensure item2index is loaded by triggering a dummy POI lookup if needed
            if not self.data_service.item2index:
                logger.warning("item2index not yet loaded, attempting to initialize...")
                # Try to load at least one POI to build the mapping
                self.data_service.get_poi_details(0)
            
            # Get POI indices the user has interacted with (all ratings >= 1 star)
            # Using min_stars=1.0 to get all interactions, not just highly-rated ones
            history = self.data_service.get_user_interactions(user_id, min_stars=1.0)
            logger.debug(f"Retrieved {len(history)} past interactions for user {user_id}")
            return history
        except Exception as e:
            logger.error(f"Failed to get user history for {user_id}: {e}")
            return []

    def get_baseline_recommendations(
        self, user_id: str, top_k: int = 20
    ) -> Dict[int, int]:
        """
        Get baseline recommendations WITHOUT steering (cached for position delta calculation).

        Returns:
            {item_id: baseline_rank} where rank is 0-based position (0 = top)
        """
        if user_id not in self.user_latents:
            raise ValueError(
                f"User {user_id} not encoded yet. Call encode_user() first."
            )

        # Return cached baseline if exists
        if user_id in self.baseline_recommendations:
            return self.baseline_recommendations[user_id]

        logger.debug(f"Computing baseline recommendations for {user_id}")

        user_z = self.user_latents[user_id].to(self.device)

        with torch.no_grad():
            # Score items WITHOUT any steering
            scores = user_z @ self.elsa._A_norm.T  # (n_items,)
            top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[0]))

        # Create rank mapping: {item_id: rank}
        baseline = {
            int(item_id.item()): rank for rank, item_id in enumerate(top_indices)
        }

        self.baseline_recommendations[user_id] = baseline
        logger.debug(f"Baseline computed: {len(baseline)} items")

        return baseline

    def get_recommendations_with_delta(
        self,
        user_id: str,
        steering_config: Optional[Dict] = None,
        top_k: int = 20,
    ) -> List[Dict]:
        """
        Get recommendations with position deltas after steering.

        Returns list of dicts with fields:
            - item_id: int
            - rank_after: int (current rank after steering)
            - rank_before: int (baseline rank before steering)
            - position_delta: int (rank_after - rank_before)
                             (negative = moved up = improved)
            - arrow: str ('↑' green, '↓' red, '→' no change)
            - arrow_value: int (abs(position_delta))
            - arrow_color: str ('green', 'red', 'gray')
            - show_delta: bool (True if steering applied)
            - score: float

        Args:
            user_id: Yelp user ID
            steering_config: Optional steering dict with keys:
                - type: 'neuron' or 'concept'
                - neuron_values: {neuron_idx: strength} for neuron steering
                - concept_vector: tensor for concept steering
                - alpha: interpolation strength (default 0.3)
            top_k: Number of recommendations to return

        Returns:
            List of recommendation dicts with position deltas
        """
        if user_id not in self.user_latents:
            raise ValueError(
                f"User {user_id} not encoded yet. Call encode_user() first."
            )

        # Get or create baseline
        if user_id not in self.baseline_recommendations:
            self.get_baseline_recommendations(user_id, top_k)

        baseline = self.baseline_recommendations[user_id]

        # Apply steering and get scores
        user_z = self.user_latents[user_id].to(self.device)

        if steering_config:
            z_final = self._apply_steering(user_z, steering_config)
            show_delta = True
        else:
            z_final = user_z
            show_delta = False

        with torch.no_grad():
            scores = z_final @ self.elsa._A_norm.T
            top_scores, top_indices = torch.topk(scores, k=min(top_k, scores.shape[0]))

            # Get SAE activations for attribution
            h_final = self.sae.encode(z_final.unsqueeze(0)).squeeze(0)

        # Build recommendations with deltas
        recommendations = []

        for rank_after, (score, item_id) in enumerate(zip(top_scores, top_indices)):
            item_id = int(item_id.item())
            score = float(score.item())

            # Get baseline rank (if not in baseline, assign penalty)
            rank_before = baseline.get(item_id, top_k + 10)
            position_delta = rank_after - rank_before

            # Determine arrow and color
            if position_delta < 0:
                arrow = "↑"
                arrow_color = "green"
            elif position_delta > 0:
                arrow = "↓"
                arrow_color = "red"
            else:
                arrow = "→"
                arrow_color = "gray"

            # Get contributing neurons/features for this item
            contributing_neurons = self._get_attribution(h_final, item_id)

            recommendations.append(
                {
                    "item_id": item_id,
                    "rank_after": rank_after,
                    "rank_before": rank_before,
                    "position_delta": position_delta,
                    "arrow": arrow,
                    "arrow_value": abs(position_delta),
                    "arrow_color": arrow_color,
                    "show_delta": show_delta,
                    "score": score,
                    "contributing_neurons": contributing_neurons,
                }
            )

        logger.debug(
            f"Generated {len(recommendations)} recommendations with deltas for {user_id}"
        )
        return recommendations

    def _apply_steering(
        self, user_z: torch.Tensor, steering_config: Dict
    ) -> torch.Tensor:
        """
        Apply steering transformation to user latent using SAE features.

        This follows the original steer_and_recommend logic:
        1. Encode user_z to SAE features: h_u = sae.encode(user_z)
        2. Apply steering to features: h_steered[neuron_idx] = value
        3. Decode back to ELSA space: z_steered = sae.decode(h_steered)
        4. Interpolate: z_final = (1-α)·user_z + α·z_steered

        Args:
            user_z: User latent vector (latent_dim,) - already on device
            steering_config: Dict with keys:
                - type: 'neuron' (concept steering not yet supported)
                - neuron_values: {neuron_idx: strength} where:
                    - neuron_idx: SAE feature index (0 to sae_hidden_dim-1)
                    - strength: slider value in [-1, 2]
                - alpha: interpolation strength (default 0.3)

        Returns:
            Steered latent vector z_final (on device)
        """
        steering_type = steering_config.get("type", "neuron")
        alpha = steering_config.get("alpha", self.alpha)

        with torch.no_grad():
            if steering_type == "neuron":
                # Get steering overrides for SAE features
                neuron_values = steering_config.get("neuron_values", {})

                # Step 1: Encode user latent to SAE features
                h_user = self.sae.encode(user_z.unsqueeze(0))  # (1, sae_hidden_dim)
                h_steered = h_user.clone()

                # Step 2: Apply steering overrides to SAE features
                for neuron_idx, slider_value in neuron_values.items():
                    if 0 <= neuron_idx < h_steered.shape[1]:
                        # Clamp slider to valid range
                        h_steered[0, neuron_idx] = torch.clamp(
                            torch.tensor(slider_value, device=self.device),
                            min=-1.0,
                            max=2.0,
                        )
                    else:
                        logger.warning(
                            f"Neuron index {neuron_idx} out of bounds (SAE hidden dim: {h_steered.shape[1]})"
                        )

                # Step 3: Decode steered features back to ELSA latent space
                z_steered = self.sae.decode(h_steered).squeeze(0)  # (latent_dim,)

                # Step 4: Interpolate between original and steered
                z_final = (1.0 - alpha) * user_z + alpha * z_steered

            else:
                logger.warning(f"Unknown steering type: {steering_type}, using baseline")
                z_final = user_z

        return z_final

    def clear_cache(self):
        """Clear all cached data (for memory efficiency or user reset)."""
        self.user_latents.clear()
        self.user_sliders.clear()
        self._item_cache = None
        self.baseline_recommendations.clear()
        logger.debug("Cache cleared")
