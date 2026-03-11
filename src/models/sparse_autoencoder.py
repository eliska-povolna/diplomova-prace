"""Sparse Autoencoder (SAE) for decomposing dense latent vectors.

Architecture
------------
    encoder:  Linear → ReLU          (dense  →  sparse code)
    decoder:  Linear (no activation) (sparse code  →  reconstruction)

Loss
----
    L = ||x - x̂||² + λ · ||z||₁

where ``z`` is the sparse code and ``λ`` controls sparsity.

Reference
---------
    Bricken et al. (2023), "Towards Monosemanticity: Decomposing Language
    Models with Dictionary Learning", Anthropic Transformer Circuits Thread.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder with L1 regularisation on activations.

    Parameters
    ----------
    input_dim:
        Dimensionality of the dense input representation.
    hidden_dim:
        Dimensionality of the (over-complete) sparse code.
        Typically ``hidden_dim >> input_dim``.
    sparsity_lambda:
        L1 regularisation coefficient (λ).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_lambda: float = 1e-3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_lambda = sparsity_lambda

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        # Tie decoder columns to unit norm (dictionary normalisation)
        nn.init.xavier_uniform_(self.decoder.weight)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Produce sparse codes from input vectors.

        Parameters
        ----------
        x:
            Input tensor of shape ``(..., input_dim)``.

        Returns
        -------
        torch.Tensor
            Sparse code of shape ``(..., hidden_dim)`` with ReLU applied.
        """
        return F.relu(self.encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct input from sparse codes.

        Parameters
        ----------
        z:
            Sparse code tensor of shape ``(..., hidden_dim)``.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of shape ``(..., input_dim)``.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode and decode in one pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        z:
            Sparse codes of shape ``(batch, hidden_dim)``.
        x_hat:
            Reconstruction of shape ``(batch, input_dim)``.
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return z, x_hat

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute combined reconstruction + sparsity loss.

        Parameters
        ----------
        x:
            Input batch of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        z, x_hat = self(x)
        reconstruction_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_lambda * z.abs().mean()
        return reconstruction_loss + sparsity_loss

    def normalise_decoder(self) -> None:
        """Normalise decoder column vectors to unit norm.

        Should be called after each optimiser step to keep the
        dictionary atoms well-conditioned (prevents degenerate solutions).
        """
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1.0)
            self.decoder.weight.div_(norms)

    @property
    def sparsity(self) -> float:
        """Return the configured sparsity regularisation coefficient."""
        return self.sparsity_lambda
