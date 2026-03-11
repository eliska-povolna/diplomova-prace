"""Tests for src.models.collaborative_filtering."""

import pytest
import torch

from src.models.collaborative_filtering import MatrixFactorization, NeuralCF


class TestMatrixFactorization:
    @pytest.fixture
    def mf(self) -> MatrixFactorization:
        return MatrixFactorization(n_users=10, n_items=20, embedding_dim=16)

    def test_forward_shape(self, mf: MatrixFactorization) -> None:
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([0, 5, 10])
        scores = mf(users, items)
        assert scores.shape == (3,)

    def test_user_embeddings_shape(self, mf: MatrixFactorization) -> None:
        emb = mf.get_user_embeddings()
        assert emb.shape == (10, 16)

    def test_item_embeddings_shape(self, mf: MatrixFactorization) -> None:
        emb = mf.get_item_embeddings()
        assert emb.shape == (20, 16)


class TestNeuralCF:
    @pytest.fixture
    def ncf(self) -> NeuralCF:
        return NeuralCF(n_users=10, n_items=20, embedding_dim=16, mlp_layers=(32, 16))

    def test_forward_shape(self, ncf: NeuralCF) -> None:
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([0, 5, 10])
        scores = ncf(users, items)
        assert scores.shape == (3,)

    def test_output_is_scalar_per_pair(self, ncf: NeuralCF) -> None:
        users = torch.tensor([0])
        items = torch.tensor([0])
        scores = ncf(users, items)
        assert scores.shape == (1,)
