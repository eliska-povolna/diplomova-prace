"""TopK Sparse Autoencoder for decomposing ELSA latent vectors.

Architecture
------------
    encoder:  Linear (no bias)    dense latent → pre-activations
    TopK mask: keep only the k largest-magnitude activations (signed)
    decoder:  Linear (with bias)  sparse code → reconstruction

Loss
----
    L = MSE_reconstruction(x̂, x) + λ · mean(|h_sparse|)

where ``x̂`` is the reconstructed latent vector and ``h_sparse`` has at most ``k`` non-zero entries per sample.

CRITICAL: Uses MSE loss (not cosine similarity) to ensure both direction AND magnitude
are preserved during reconstruction. This aligns the latent-space training objective
with the item-space evaluation metric (recommendation ranking quality).

Reference
---------
    Spišák M., Bartyzal R., Hoskovec A., Peška L. (2024).
    "On Interpretability of Linear Autoencoders."
    Proceedings of RecSys '24, ACM.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_mask(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return a binary mask that is 1 for the ``k`` largest-magnitude entries.

    Parameters
    ----------
    x:
        Tensor of shape ``(batch, hidden_dim)``.
    k:
        Number of entries to keep per row.

    Returns
    -------
    torch.Tensor
        Binary mask of shape ``(batch, hidden_dim)``.
    """
    # Validate k to avoid runtime errors from torch.topk, which requires
    # 1 <= k <= x.shape[1].
    if not isinstance(k, int):
        raise TypeError(f"topk_mask: k must be an int, got {type(k).__name__}.")
    if k <= 0:
        raise ValueError(f"topk_mask: k must be positive, got {k}.")
    hidden_dim = x.shape[1]
    if k > hidden_dim:
        raise ValueError(
            f"topk_mask: k ({k}) cannot be greater than hidden dimension ({hidden_dim})."
        )

    _, idx = torch.topk(x.abs(), k, dim=1)
    return torch.zeros_like(x).scatter(1, idx, 1.0)


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Parameters
    ----------
    input_dim:
        Dimensionality of the dense ELSA latent vector.
    hidden_dim:
        Dimensionality of the (over-complete) dictionary.
        Typically ``hidden_dim = width_ratio * input_dim``.
    k:
        Number of active features per sample (sparsity level).
    l1_coef:
        L1 regularisation coefficient applied to sparse activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 32,
        l1_coef: float = 3e-4,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.l1_coef = l1_coef

        # No bias on encoder (matches reference implementation)
        self.enc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dec = nn.Linear(hidden_dim, input_dim, bias=True)

        nn.init.xavier_uniform_(self.enc.weight)
        nn.init.xavier_uniform_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode and decode with TopK sparsification.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        recon:
            Reconstructed tensor of shape ``(batch, input_dim)``.
        h_sparse:
            Sparse code (at most ``k`` non-zeros per row) of shape
            ``(batch, hidden_dim)``.
        h_pre:
            Pre-activation (dense) encoder output, shape
            ``(batch, hidden_dim)``.
        """
        h_pre = self.enc(x)
        mask = topk_mask(h_pre, self.k)
        h_sparse = h_pre * mask
        recon = self.dec(h_sparse)
        return recon, h_sparse, h_pre

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the sparse feature code for input ``x``."""
        h_pre = self.enc(x)
        mask = topk_mask(h_pre, self.k)
        return h_pre * mask

    def decode(self, h_sparse: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from a sparse feature code."""
        return self.dec(h_sparse)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute combined MSE reconstruction + L1 sparsity loss.

        CRITICAL FIX: Changed from cosine similarity to MSE in original space.
        Cosine loss in latent space doesn't reflect reconstruction quality in item space.

        Parameters
        ----------
        x:
            Input batch of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        recon, h_sparse, _ = self(x)
        # Use MSE which preserves both direction AND magnitude
        rec_loss = F.mse_loss(recon, x)
        l1_loss = self.l1_coef * h_sparse.abs().mean()
        return rec_loss + l1_loss

    @property
    def sparsity_k(self) -> int:
        """Return the configured TopK sparsity level."""
        return self.k
