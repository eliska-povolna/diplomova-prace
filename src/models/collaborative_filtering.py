"""ELSA — Scalable Linear Shallow Autoencoder for collaborative filtering.

ELSA learns a single item-factor matrix ``A ∈ R^{n_items × latent_dim}``.
The forward pass:

    z = X @ normalize(A)           # user latent vectors
    X̂ = z @ normalize(A)ᵀ         # reconstruction

Training uses a Normalised MSE loss between L2-normalised input and output.
This matches the reference implementation by Spišák et al. (RecSys 2024).

Reference
---------
    Spišák M., Bartyzal R., Hoskovec A., Peška L. (2024).
    "On Interpretability of Linear Autoencoders."
    Proceedings of RecSys '24, ACM.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default latent dimension used throughout the codebase
LATENT_DIM: int = 512


class ELSA(nn.Module):
    """Scalable Linear Shallow Autoencoder.

    Parameters
    ----------
    n_items:
        Number of items in the catalogue.
    latent_dim:
        Dimensionality of the latent space.
    """

    def __init__(self, n_items: int, latent_dim: int = LATENT_DIM) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.A = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(n_items, latent_dim))
        )

    @property
    def _A_norm(self) -> torch.Tensor:
        """Return column-normalised item factor matrix (cached per forward call)."""
        return F.normalize(self.A, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input interaction vectors.

        Parameters
        ----------
        x:
            Dense user–item interaction vectors of shape ``(batch, n_items)``.

        Returns
        -------
        torch.Tensor
            Reconstructed vectors of shape ``(batch, n_items)``.
        """
        A = self._A_norm
        z = x @ A
        return z @ A.T

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce L2-normalised latent vectors.

        Parameters
        ----------
        x:
            Dense interaction vectors of shape ``(batch, n_items)`` or
            a sparse CSR matrix chunk.

        Returns
        -------
        torch.Tensor
            L2-normalised latent vectors of shape ``(batch, latent_dim)``.
        """
        z = x @ self._A_norm
        return F.normalize(z, dim=-1)

    def encode_csr_chunked(
        self, X_csr, *, chunk_size: int = 4096, device: str = "cpu"
    ) -> torch.Tensor:
        """Encode a large sparse CSR matrix in memory-efficient chunks.

        Parameters
        ----------
        X_csr:
            scipy sparse CSR matrix of shape ``(n_users, n_items)``.
        chunk_size:
            Number of rows processed at a time.
        device:
            Torch device string.

        Returns
        -------
        torch.Tensor
            L2-normalised latent matrix of shape ``(n_users, latent_dim)``
            on CPU.
        """
        A = self._A_norm.detach()
        parts: list[torch.Tensor] = []
        n = X_csr.shape[0]
        for start in range(0, n, chunk_size):
            chunk = torch.tensor(
                X_csr[start : start + chunk_size].toarray(),
                dtype=torch.float32,
                device=device,
            )
            z = chunk @ A
            z = F.normalize(z, dim=-1)
            parts.append(z.cpu())
            del chunk, z
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        return torch.cat(parts, dim=0)


class NMSELoss(nn.Module):
    """Normalised MSE loss: MSE between L2-normalised input and output."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(
            F.normalize(pred, dim=-1),
            F.normalize(target, dim=-1),
        )


# ── Evaluation helpers ────────────────────────────────────────────────────

def recall_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """Recall@k for a single user."""
    hits = y_true[y_pred_argsorted[:k]].sum()
    total = y_true.sum()
    return float(hits / total) if total > 0 else float("nan")


def ndcg_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """NDCG@k for a single user."""
    hits = y_true[y_pred_argsorted[:k]]
    if hits.sum() == 0:
        return 0.0
    gains = hits / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()
    ideal = np.sort(y_true)[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, k + 2))).sum()
    return float(dcg / idcg) if idcg > 0 else 0.0

