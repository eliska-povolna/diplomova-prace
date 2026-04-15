"""Combined ELSA + TopK SAE model for interpretable POI recommendation.

Pipeline
--------
1. **ELSA** is trained on the user–item interaction matrix (CSR) to produce
   item factors ``A`` and per-user latent vectors ``z = normalize(x @ A)``.
2. **TopK SAE** is trained on the frozen ELSA latent vectors to decompose
   ``z`` into a sparse, interpretable feature code ``h_sparse``.
3. At inference time:
   - Encode user ``u`` → ``z_u = ELSA.encode(x_u)``
   - Decompose  → ``h_u = SAE.encode(z_u)``
   - Optionally override specific feature activations (the "knobs").
   - Reconstruct → ``ẑ_u = SAE.decode(h_u')``
   - Score items → ``scores = ẑ_u @ normalize(A).T``

This matches the architecture described in
``src/yelp_initial_exploration/train_elsa.py`` and
``src/yelp_initial_exploration/train_sae.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .collaborative_filtering import ELSA
from .sparse_autoencoder import TopKSAE


class ELSASAEModel(nn.Module):
    """ELSA + TopK SAE model for interpretable, controllable POI recommendation.

    Parameters
    ----------
    n_items:
        Number of items (POIs) in the catalogue.
    latent_dim:
        ELSA latent dimensionality.
    sae_hidden_dim:
        SAE dictionary size (``hidden_dim = width_ratio * latent_dim``).
    k:
        TopK sparsity level for the SAE.
    l1_coef:
        L1 regularisation coefficient for the SAE loss.
    """

    def __init__(
        self,
        n_items: int,
        latent_dim: int = 512,
        sae_hidden_dim: int = 2048,
        k: int = 32,
        l1_coef: float = 3e-4,
    ) -> None:
        super().__init__()
        self.elsa = ELSA(n_items, latent_dim)
        self.sae = TopKSAE(
            input_dim=latent_dim,
            hidden_dim=sae_hidden_dim,
            k=k,
            l1_coef=l1_coef,
        )

    # ── ELSA interface ────────────────────────────────────────────────────

    def elsa_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised ELSA latent vectors.

        Parameters
        ----------
        x:
            Dense interaction vectors of shape ``(batch, n_items)``.

        Returns
        -------
        torch.Tensor
            Latent vectors of shape ``(batch, latent_dim)``.
        """
        return self.elsa.encode(x)

    def elsa_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return ELSA item scores (full reconstruction).

        Parameters
        ----------
        x:
            Dense interaction vectors of shape ``(batch, n_items)``.

        Returns
        -------
        torch.Tensor
            Predicted scores of shape ``(batch, n_items)``.
        """
        return self.elsa(x)

    # ── SAE interface ─────────────────────────────────────────────────────

    def sae_encode(self, z: torch.Tensor) -> torch.Tensor:
        """Return sparse feature codes for latent vectors.

        Parameters
        ----------
        z:
            ELSA latent vectors of shape ``(batch, latent_dim)``.

        Returns
        -------
        torch.Tensor
            Sparse codes of shape ``(batch, sae_hidden_dim)`` with at most
            ``k`` non-zero entries per row.
        """
        return self.sae.encode(z)

    # ── Controlled recommendation ─────────────────────────────────────────

    def recommend(
        self,
        x: torch.Tensor,
        *,
        feature_overrides: dict[int, float] | None = None,
    ) -> torch.Tensor:
        """Score all items for a batch of users, with optional feature steering.

        Parameters
        ----------
        x:
            Dense interaction vectors of shape ``(batch, n_items)``.
        feature_overrides:
            Optional ``{feature_index: new_activation_value}`` mapping.
            Override specific SAE feature activations before decoding to
            steer recommendations (the "knobs" interface).

        Returns
        -------
        torch.Tensor
            Item scores of shape ``(batch, n_items)``.
        """
        z = self.elsa.encode(x)
        h = self.sae.encode(z)

        if feature_overrides:
            h = h.clone()
            sae_hidden_dim = h.size(1)
            for feat_idx, value in feature_overrides.items():
                if not isinstance(feat_idx, int):
                    raise ValueError(
                        f"feature_overrides keys must be integers, got {type(feat_idx).__name__!r}"
                    )
                if feat_idx < 0 or feat_idx >= sae_hidden_dim:
                    raise ValueError(
                        f"feature_overrides index {feat_idx} is out of bounds for SAE hidden "
                        f"dimension {sae_hidden_dim}; valid indices are in [0, {sae_hidden_dim})."
                    )
                h[:, feat_idx] = value

        z_steered = self.sae.decode(h)

        # Score items: steered_z @ normalize(A)ᵀ
        return z_steered @ self.elsa._A_norm.T

    # ── Loss helpers ──────────────────────────────────────────────────────

    def sae_loss(self, z: torch.Tensor) -> torch.Tensor:
        """SAE cosine reconstruction + L1 loss for a batch of latent vectors."""
        return self.sae.loss(z)
