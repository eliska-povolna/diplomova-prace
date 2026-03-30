"""Inference service for steering and recommendation generation."""

from pathlib import Path
from typing import Dict, List, Optional
import logging

import torch
import torch.nn.functional as F
import numpy as np

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
        config: Optional[Dict] = None
    ):
        """
        Initialize inference service and load models.
        
        Args:
            elsa_checkpoint_path: Path to elsa_best.pt
            sae_checkpoint_path: Path to sae_best.pt
            config: Configuration dict with keys:
                - steering_alpha: interpolation strength (default 0.3)
                - device: 'cpu' or 'cuda' (default 'cpu')
        """
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.alpha = self.config.get('steering_alpha', 0.3)
        
        logger.info(f"Loading models on device: {self.device}")
        self.elsa = self._load_checkpoint(elsa_checkpoint_path)
        self.sae = self._load_checkpoint(sae_checkpoint_path)
        
        # Per-session caches (cleared on user selection)
        self.user_latents = {}      # {user_id: z_u tensor}
        self.user_sliders = {}      # {user_id: {neuron_idx: value}}
        
        # Item score cache for current user
        self._item_cache = None     # Tuple of (scores, user_id)
        
        logger.info("✅ Inference service ready")
    
    def _load_checkpoint(self, ckpt_path: Path):
        """Load a PyTorch model checkpoint."""
        ckpt_path = Path(ckpt_path)
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # TODO: Instantiate model class and load state
        # For now, return checkpoint (will be updated with actual model class)
        return checkpoint
    
    def encode_user(
        self,
        user_id: str,
        user_interactions_csr
    ) -> torch.Tensor:
        """
        Encode a user from interactions into latent space.
        
        Args:
            user_id: Yelp user ID
            user_interactions_csr: CSR matrix of user interactions
                (typically: row vector of 0/1 indicating liked POIs)
        
        Returns:
            Latent embedding vector (cached for steering)
        """
        # TODO: Forward pass through ELSA encoder
        # user_z = self.elsa.encode(user_interactions_csr)
        
        # For now, placeholder
        user_z = torch.randn(128, device=self.device)
        
        self.user_latents[user_id] = user_z
        self.user_sliders[user_id] = {}
        
        logger.debug(f"Encoded user {user_id}: z_shape={user_z.shape}")
        return user_z
    
    def get_top_activations(
        self,
        user_latent: torch.Tensor,
        k: int = 10
    ) -> List[Dict]:
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
        topk_vals, topk_idx = torch.topk(
            h.abs(),
            k=min(k, h.shape[0])
        )
        
        result = []
        for idx, val in zip(topk_idx.tolist(), topk_vals.tolist()):
            result.append({
                'neuron_idx': idx,
                'activation': float(val),
                'label': f"Feature {idx}",  # TODO: Get from LabelingService
            })
        
        return result
    
    def steer_and_recommend(
        self,
        user_id: str,
        steering_overrides: Optional[Dict[int, float]] = None,
        top_k: int = 20
    ) -> Dict:
        """
        Apply steering and generate recommendations.
        
        **Algorithm** (from EasyStudy SAESteering):
        
        1. Retrieve cached user latent: z_u = user_latents[user_id]
        2. Extract SAE decoder basis vectors (normalized)
        3. Build steering vector: Σ(weight_i * basis_i)
        4. Normalize steering vector
        5. Interpolate: z_steered = (1 - α)·z_u + α·v_steer
        6. Score all items: scores = SAE·decode(z_steered)
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
        """
        steering_overrides = steering_overrides or {}
        
        # Validate user is encoded
        if user_id not in self.user_latents:
            raise ValueError(
                f"User {user_id} not encoded yet. Call encode_user() first."
            )
        
        user_z = self.user_latents[user_id]
        self.user_sliders[user_id] = steering_overrides
        
        # === Build steering vector ===
        # Get SAE decoder basis vectors (normalized columns)
        decoder_weight = self.sae.decoder.weight  # shape: (latent_dim, item_dim)
        basis_vectors = F.normalize(decoder_weight, dim=0).T  # (item_dim, latent_dim)
        
        # Accumulate steering from slider overrides
        steering_vector = torch.zeros_like(user_z, dtype=torch.float32)
        for neuron_idx, slider_value in steering_overrides.items():
            if 0 <= neuron_idx < basis_vectors.shape[0]:
                # Clamp slider to valid range
                clamped_value = torch.clamp(
                    torch.tensor(slider_value, device=self.device),
                    min=-1.0,
                    max=2.0
                )
                steering_vector = steering_vector + (
                    clamped_value * basis_vectors[neuron_idx]
                )
        
        # Normalize steering vector
        if steering_vector.norm() > 0:
            steering_vector = F.normalize(steering_vector, dim=-1)
        
        # === Interpolate ===
        z_steered = (1.0 - self.alpha) * user_z + self.alpha * steering_vector
        
        # === Score items ===
        with torch.no_grad():
            scores = self._score_items(z_steered.unsqueeze(0)).squeeze()
        
        # === Get top-k ===
        top_indices = torch.argsort(-scores, descending=True)[:top_k]
        
        result = {
            'recommendations': [
                {
                    'poi_idx': int(idx.item()),
                    'score': float(scores[idx].item()),
                    'contributing_neurons': self._get_attribution(
                        z_steered, idx
                    ),
                }
                for idx in top_indices
            ],
            'steering_applied': steering_overrides,
            'alpha': self.alpha,
        }
        
        # Cache for subsequent queries
        self._item_cache = (scores, user_id)
        
        return result
    
    def _score_items(self, latent_batch: torch.Tensor) -> torch.Tensor:
        """
        Score all items given latent representation(s).
        
        Args:
            latent_batch: Tensor of shape (batch_size, latent_dim)
        
        Returns:
            Scores of shape (batch_size, n_items)
        """
        with torch.no_grad():
            h = self.sae.encode(latent_batch)
            scores = self.sae.decode(h)
        
        return scores
    
    def _get_attribution(
        self,
        z_steered: torch.Tensor,
        item_idx: torch.Tensor,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Identify which neurons most contributed to this recommendation.
        
        Args:
            z_steered: Steered latent vector
            item_idx: Index of recommended item
            top_k: Number of top neurons to return
        
        Returns:
            List of dicts with neuron_idx and contribution score
        """
        # Simplified attribution: rank neurons by activation magnitude
        with torch.no_grad():
            h = self.sae.encode(z_steered.unsqueeze(0)).squeeze()
        
        # Get top-k neurons
        topk_vals, topk_idx = torch.topk(
            h.abs(),
            k=min(top_k, h.shape[0])
        )
        
        return [
            {
                'idx': int(idx.item()),
                'label': f"Feature {idx.item()}",
                'contribution': float(val.item()),
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
