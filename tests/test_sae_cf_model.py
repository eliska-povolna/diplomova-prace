"""Tests for src.models.sae_cf_model (ELSASAEModel)."""

import pytest
import torch

from src.models.sae_cf_model import ELSASAEModel


@pytest.fixture
def model() -> ELSASAEModel:
    return ELSASAEModel(
        n_items=20,
        latent_dim=16,
        sae_hidden_dim=64,
        k=4,
        l1_coef=3e-4,
    )


class TestELSASAEModel:
    def test_elsa_score_shape(self, model: ELSASAEModel) -> None:
        x = torch.randn(8, 20)
        scores = model.elsa_score(x)
        assert scores.shape == (8, 20)

    def test_elsa_encode_shape(self, model: ELSASAEModel) -> None:
        x = torch.randn(8, 20)
        z = model.elsa_encode(x)
        assert z.shape == (8, 16)

    def test_sae_encode_shape(self, model: ELSASAEModel) -> None:
        z = torch.randn(8, 16)
        h = model.sae_encode(z)
        assert h.shape == (8, 64)

    def test_sae_encode_at_most_k_nonzeros(self, model: ELSASAEModel) -> None:
        z = torch.randn(8, 16)
        h = model.sae_encode(z)
        assert (h != 0).sum(dim=1).max().item() <= model.sae.k

    def test_recommend_shape(self, model: ELSASAEModel) -> None:
        x = torch.randn(8, 20)
        scores = model.recommend(x)
        assert scores.shape == (8, 20)

    def test_recommend_with_feature_override(self, model: ELSASAEModel) -> None:
        x = torch.randn(4, 20)
        baseline = model.recommend(x)
        steered = model.recommend(x, feature_overrides={0: 100.0})
        assert steered.shape == (4, 20)
        # Overriding a feature should change the scores
        assert not torch.allclose(baseline, steered)

    def test_sae_loss_is_scalar(self, model: ELSASAEModel) -> None:
        z = torch.randn(8, 16)
        loss = model.sae_loss(z)
        assert loss.shape == ()

    def test_sae_loss_nonnegative(self, model: ELSASAEModel) -> None:
        z = torch.randn(8, 16)
        assert model.sae_loss(z).item() >= 0

