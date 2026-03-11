"""Combined SAE-CF model for interpretable, controllable POI recommendation.

The model works in two stages:

1. **Train** a collaborative filtering model (``MatrixFactorization``) to
   obtain dense user embeddings.
2. **Post-hoc SAE** training: fit a ``SparseAutoencoder`` on the frozen user
   embeddings so that each user is described by a sparse, interpretable code.

At inference time:
    - Lookup the sparse code for user ``u``.
    - Optionally, allow the user to modify individual feature activations.
    - Reconstruct a (modified) dense embedding from the SAE decoder.
    - Score items using the CF scoring function with the reconstructed embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .collaborative_filtering import MatrixFactorization
from .sparse_autoencoder import SparseAutoencoder


class SAECFModel(nn.Module):
    """Sparse Autoencoder enhanced Collaborative Filtering model.

    Parameters
    ----------
    n_users:
        Number of users.
    n_items:
        Number of items (POIs).
    embedding_dim:
        CF embedding dimensionality (= SAE input dimension).
    sae_hidden_dim:
        Dimensionality of the sparse code (``hidden_dim >> embedding_dim``).
    sparsity_lambda:
        L1 coefficient for the SAE loss.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        sae_hidden_dim: int = 512,
        sparsity_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        self.cf = MatrixFactorization(n_users, n_items, embedding_dim)
        self.sae = SparseAutoencoder(embedding_dim, sae_hidden_dim, sparsity_lambda)

    # ── CF training interface ─────────────────────────────────────────────

    def cf_score(
        self, user_ids: torch.Tensor, item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Return raw CF interaction scores."""
        return self.cf(user_ids, item_ids)

    # ── SAE interface ─────────────────────────────────────────────────────

    def encode_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Return sparse feature codes for a batch of users.

        Parameters
        ----------
        user_ids:
            Long tensor of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Sparse codes of shape ``(batch, sae_hidden_dim)``.
        """
        embeddings = self.cf.user_embedding(user_ids)
        z, _ = self.sae(embeddings)
        return z

    def reconstruct_user(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Return SAE-reconstructed user embeddings."""
        z = self.encode_user(user_ids)
        return self.sae.decode(z)

    # ── Controlled recommendation ─────────────────────────────────────────

    def recommend(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        *,
        feature_overrides: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """Score items for users, optionally with feature-level overrides.

        Parameters
        ----------
        user_ids:
            Long tensor of shape ``(batch,)``.
        item_ids:
            Long tensor of shape ``(batch, n_candidates)`` or ``(n_candidates,)``.
        feature_overrides:
            Optional mapping ``{feature_index: new_activation_value}``.
            If provided, the specified SAE features are overridden before
            reconstructing the user embedding.  This is the main
            *controllability* mechanism.

        Returns
        -------
        torch.Tensor
            Predicted scores of the same leading shape as ``item_ids``.
        """
        z = self.encode_user(user_ids)

        if feature_overrides:
            z = z.clone()
            for feat_idx, value in feature_overrides.items():
                z[:, feat_idx] = value

        user_emb = self.sae.decode(z)
        item_emb = self.cf.item_embedding(item_ids)

        # Dot-product scoring
        if item_ids.dim() == 1:
            return (user_emb * item_emb).sum(dim=-1)
        # item_ids: (batch, n_candidates) → scores: (batch, n_candidates)
        return torch.einsum("bd,bcd->bc", user_emb, item_emb)

    # ── Loss helpers ──────────────────────────────────────────────────────

    def sae_loss(self) -> torch.Tensor:
        """SAE reconstruction + sparsity loss on **all** user embeddings."""
        all_embeddings = self.cf.get_user_embeddings()
        return self.sae.loss(all_embeddings)
