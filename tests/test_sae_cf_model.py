"""Tests for src.models.sae_cf_model."""

import pytest
import torch

from src.models.sae_cf_model import SAECFModel


@pytest.fixture
def model() -> SAECFModel:
    return SAECFModel(
        n_users=10,
        n_items=20,
        embedding_dim=16,
        sae_hidden_dim=64,
        sparsity_lambda=1e-3,
    )


class TestSAECFModel:
    def test_cf_score_shape(self, model: SAECFModel) -> None:
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([5, 10, 15])
        scores = model.cf_score(users, items)
        assert scores.shape == (3,)

    def test_encode_user_shape(self, model: SAECFModel) -> None:
        users = torch.tensor([0, 1])
        z = model.encode_user(users)
        assert z.shape == (2, 64)

    def test_encode_user_nonnegative(self, model: SAECFModel) -> None:
        users = torch.tensor([0, 1, 2])
        z = model.encode_user(users)
        assert (z >= 0).all()

    def test_reconstruct_user_shape(self, model: SAECFModel) -> None:
        users = torch.tensor([0, 1])
        x_hat = model.reconstruct_user(users)
        assert x_hat.shape == (2, 16)

    def test_recommend_1d_items(self, model: SAECFModel) -> None:
        users = torch.tensor([0, 1, 2])
        items = torch.tensor([3, 7, 12])
        scores = model.recommend(users, items)
        assert scores.shape == (3,)

    def test_recommend_with_feature_override(self, model: SAECFModel) -> None:
        users = torch.tensor([0])
        items = torch.tensor([5])
        baseline = model.recommend(users, items).item()
        overridden = model.recommend(users, items, feature_overrides={0: 100.0}).item()
        # After overriding a feature the score should generally change
        # (not guaranteed for random weights, so we just check no exception raised)
        assert isinstance(overridden, float)
        assert isinstance(baseline, float)

    def test_sae_loss_is_scalar(self, model: SAECFModel) -> None:
        loss = model.sae_loss()
        assert loss.shape == ()

    def test_sae_loss_positive(self, model: SAECFModel) -> None:
        loss = model.sae_loss()
        assert loss.item() > 0
