"""Tests for src.models.sparse_autoencoder (TopK SAE)."""

import pytest
import torch

from src.models.sparse_autoencoder import TopKSAE, topk_mask


class TestTopKMask:
    def test_shape(self) -> None:
        x = torch.randn(4, 32)
        mask = topk_mask(x, k=5)
        assert mask.shape == (4, 32)

    def test_exactly_k_ones_per_row(self) -> None:
        x = torch.randn(4, 32)
        mask = topk_mask(x, k=5)
        assert (mask.sum(dim=1) == 5).all()

    def test_binary_values(self) -> None:
        x = torch.randn(4, 32)
        mask = topk_mask(x, k=5)
        assert set(mask.unique().tolist()).issubset({0.0, 1.0})


@pytest.fixture
def sae() -> TopKSAE:
    return TopKSAE(input_dim=16, hidden_dim=64, k=4, l1_coef=3e-4)


class TestTopKSAE:
    def test_forward_shapes(self, sae: TopKSAE) -> None:
        x = torch.randn(8, 16)
        recon, h_sparse, h_pre = sae(x)
        assert recon.shape == (8, 16)
        assert h_sparse.shape == (8, 64)
        assert h_pre.shape == (8, 64)

    def test_sparse_code_has_exactly_k_nonzeros(self, sae: TopKSAE) -> None:
        # With random inputs of dim > k, the TopK mask should select exactly k positions
        torch.manual_seed(0)
        x = torch.randn(8, 16)
        _, _, h_pre = sae(x)
        mask = topk_mask(h_pre, sae.k)
        ones_per_row = mask.sum(dim=1)
        assert (ones_per_row == sae.k).all()

    def test_sparse_code_has_at_most_k_nonzeros(self, sae: TopKSAE) -> None:
        x = torch.randn(8, 16)
        _, h_sparse, _ = sae(x)
        nonzeros_per_row = (h_sparse != 0).sum(dim=1)
        assert (nonzeros_per_row <= sae.k).all()

    def test_encode_shape(self, sae: TopKSAE) -> None:
        x = torch.randn(8, 16)
        h = sae.encode(x)
        assert h.shape == (8, 64)

    def test_decode_shape(self, sae: TopKSAE) -> None:
        h = torch.zeros(8, 64)
        recon = sae.decode(h)
        assert recon.shape == (8, 16)

    def test_loss_is_scalar(self, sae: TopKSAE) -> None:
        x = torch.randn(8, 16)
        loss = sae.loss(x)
        assert loss.shape == ()

    def test_loss_is_nonnegative(self, sae: TopKSAE) -> None:
        x = torch.randn(8, 16)
        assert sae.loss(x).item() >= 0

    def test_sparsity_k_property(self, sae: TopKSAE) -> None:
        assert sae.sparsity_k == 4
