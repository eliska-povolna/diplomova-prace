"""Tests for src.models.sparse_autoencoder."""

import pytest
import torch

from src.models.sparse_autoencoder import SparseAutoencoder


@pytest.fixture
def sae() -> SparseAutoencoder:
    return SparseAutoencoder(input_dim=16, hidden_dim=64, sparsity_lambda=1e-3)


class TestSparseAutoencoder:
    def test_encode_shape(self, sae: SparseAutoencoder) -> None:
        x = torch.randn(8, 16)
        z = sae.encode(x)
        assert z.shape == (8, 64)

    def test_encode_nonnegative(self, sae: SparseAutoencoder) -> None:
        x = torch.randn(8, 16)
        z = sae.encode(x)
        assert (z >= 0).all(), "ReLU encoding must be non-negative"

    def test_decode_shape(self, sae: SparseAutoencoder) -> None:
        z = torch.randn(8, 64).clamp(min=0)
        x_hat = sae.decode(z)
        assert x_hat.shape == (8, 16)

    def test_forward_shapes(self, sae: SparseAutoencoder) -> None:
        x = torch.randn(8, 16)
        z, x_hat = sae(x)
        assert z.shape == (8, 64)
        assert x_hat.shape == (8, 16)

    def test_loss_is_scalar(self, sae: SparseAutoencoder) -> None:
        x = torch.randn(8, 16)
        loss = sae.loss(x)
        assert loss.shape == ()

    def test_loss_is_positive(self, sae: SparseAutoencoder) -> None:
        x = torch.randn(8, 16)
        loss = sae.loss(x)
        assert loss.item() > 0

    def test_normalise_decoder(self, sae: SparseAutoencoder) -> None:
        sae.normalise_decoder()
        norms = sae.decoder.weight.norm(dim=0)
        # All norms should be <= 1.0 after normalisation
        assert (norms <= 1.0 + 1e-6).all()

    def test_sparsity_property(self, sae: SparseAutoencoder) -> None:
        assert sae.sparsity == pytest.approx(1e-3)
