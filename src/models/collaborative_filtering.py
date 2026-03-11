"""Baseline collaborative filtering / matrix factorisation model.

Implements:
    - MatrixFactorization  – classic MF with BPR loss
    - NeuralCF             – neural collaborative filtering (He et al., 2017)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """Bilinear matrix factorisation for implicit feedback.

    Parameters
    ----------
    n_users:
        Number of users.
    n_items:
        Number of items (POIs).
    embedding_dim:
        Dimensionality of user and item embeddings.
    """

    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Return predicted interaction scores.

        Parameters
        ----------
        user_ids:
            Long tensor of shape ``(batch,)``.
        item_ids:
            Long tensor of shape ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Predicted scores of shape ``(batch,)``.
        """
        u = self.user_embedding(user_ids)
        i = self.item_embedding(item_ids)
        return (u * i).sum(dim=-1)

    def get_user_embeddings(self) -> torch.Tensor:
        """Return all user embedding vectors."""
        return self.user_embedding.weight.detach()

    def get_item_embeddings(self) -> torch.Tensor:
        """Return all item embedding vectors."""
        return self.item_embedding.weight.detach()


class NeuralCF(nn.Module):
    """Neural collaborative filtering (He et al., 2017).

    Combines a generalised matrix factorisation (GMF) path with a
    multi-layer perceptron (MLP) path and fuses them before prediction.

    Parameters
    ----------
    n_users:
        Number of users.
    n_items:
        Number of items (POIs).
    embedding_dim:
        Embedding dimensionality for **each** of the two paths.
    mlp_layers:
        Sizes of hidden MLP layers (excluding the input layer).
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 32,
        mlp_layers: tuple[int, ...] = (64, 32, 16),
    ) -> None:
        super().__init__()

        # GMF path
        self.gmf_user = nn.Embedding(n_users, embedding_dim)
        self.gmf_item = nn.Embedding(n_items, embedding_dim)

        # MLP path
        self.mlp_user = nn.Embedding(n_users, embedding_dim)
        self.mlp_item = nn.Embedding(n_items, embedding_dim)

        mlp_input_dim = 2 * embedding_dim
        layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.output = nn.Linear(embedding_dim + in_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        gmf_out = self.gmf_user(user_ids) * self.gmf_item(item_ids)
        mlp_in = torch.cat([self.mlp_user(user_ids), self.mlp_item(item_ids)], dim=-1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.output(fused).squeeze(-1)
