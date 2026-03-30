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
        config: Optional[Dict] = None
    ):
        """
        Initialize inference service and load models.
        
        Args:
            elsa_checkpoint_path: Path to elsa_best.pt
            sae_checkpoint_path: Path to sae_best.pt
            config: Configuration dict with keys:
                - n_items: Number of items (required for ELSA)
                - latent_dim: ELSA latent dimension (default 512)
                - k: SAE sparsity level (default 32)
                - width_ratio: SAE hidden dim ratio (default 4)
                - steering_alpha: interpolation strength (default 0.3)
                - device: 'cpu' or 'cuda' (default 'cpu')
        """
        self.config = config or {}
        self.device = self.config.get('device', 'cpu')
        self.alpha = self.config.get('steering_alpha', 0.3)
        
        # Model hyperparameters
        self.n_items = self.config.get('n_items')
        self.latent_dim = self.config.get('latent_dim', 512)
        self.k = self.config.get('k', 32)
        self.width_ratio = self.config.get('width_ratio', 4)
        
        if not self.n_items:
            raise ValueError("config must include 'n_items' for model initialization")
        
        logger.info(f"Loading models on device: {self.device}")
        logger.info(f"Model config: n_items={self.n_items}, latent_dim={self.latent_dim}, k={self.k}")
        
        self.elsa = self._load_elsa(elsa_checkpoint_path)
        self.sae = self._load_sae(sae_checkpoint_path)
        
        # Per-session caches (cleared on user selection)
        self.user_latents = {}      # {user_id: z_u tensor}
        self.user_sliders = {}      # {user_id: {neuron_idx: value}}
        
        # Item score cache for current user
        self._item_cache = None     # Tuple of (scores, user_id)
        
        logger.info("✅ Inference service ready")
    
    def _load_elsa(self, ckpt_path: Path) -> ELSA:
        """Load ELSA model from checkpoint."""
        ckpt_path = Path(ckpt_path)
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"ELSA checkpoint not found: {ckpt_path}")
        
        logger.info(f"Loading ELSA from {ckpt_path}")
        
        # Instantiate model
        model = ELSA(self.n_items, latent_dim=self.latent_dim)
        model = model.to(self.device)
        model.eval()
        
        # Load state dict
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback if checkpoint is just state dict
            model.load_state_dict(checkpoint)
        
        logger.info(f"✅ ELSA loaded: {ckpt_path.name}")
        return model
    
    def _load_sae(self, ckpt_path: Path) -> TopKSAE:
        """Load TopK SAE model from checkpoint."""
        ckpt_path = Path(ckpt_path)
        
        if not ckpt_path.exists():
            raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")
        
        logger.info(f"Loading SAE from {ckpt_path}")
        
        # Calculate hidden_dim from width_ratio
        sae_hidden_dim = self.width_ratio * self.latent_dim
        
        # Instantiate model
        model = TopKSAE(
            input_dim=self.latent_dim,
            hidden_dim=sae_hidden_dim,
            k=self.k,
            l1_coef=self.config.get('l1_coef', 3e-4)
        )
        model = model.to(self.device)
        model.eval()
        
        # Load state dict
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback if checkpoint is just state dict
            model.load_state_dict(checkpoint)
        
        logger.info(f"✅ SAE loaded: {ckpt_path.name}")
        return model
    
    def encode_user(
        self,
        user_id: str,
        user_interactions_csr
    ) -> torch.Tensor:
        """
        Encode a user from interactions into latent space using ELSA.
        
        Args:
            user_id: Yelp user ID
            user_interactions_csr: CSR matrix of user interactions
                (typically: row vector of 0/1 indicating liked POIs)
        
        Returns:
            Latent embedding vector (cached for steering)
        """
        with torch.no_grad():
            # Convert CSR to dense tensor if needed
            if hasattr(user_interactions_csr, 'toarray'):
                # It's a sparse matrix - convert to dense
                x_dense = torch.tensor(
                    user_interactions_csr.toarray(),
                    dtype=torch.float32,
                    device=self.device
                )
            else:
                # Already dense
                x_dense = torch.tensor(
                    user_interactions_csr,
                    dtype=torch.float32,
                    device=self.device
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
        
        logger.debug(f"Encoded user {user_id}: z_shape={user_z.shape}")
        return user_z.detach().cpu()
    
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
        """
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
                        max=2.0
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
            'recommendations': [
                {
                    'poi_idx': int(idx.item()),
                    'score': float(score.item()),
                    'contributing_neurons': self._get_attribution(
                        h_steered.squeeze(0), idx.item()
                    ),
                }
                for score, idx in zip(top_scores, top_indices)
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
            scores = latent_batch @ self.elsa._A_norm.T  # (batch, n_items)
        
        return scores
    
    def _get_attribution(
        self,
        h_sparse: torch.Tensor,
        item_idx: int,
        top_k: int = 3
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
            h_sparse.abs(),
            k=min(top_k, h_sparse.shape[0])
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
