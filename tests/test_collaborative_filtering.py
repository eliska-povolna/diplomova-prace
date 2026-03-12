"""Tests for src.models.collaborative_filtering (ELSA)."""

import torch
import pytest

from src.models.collaborative_filtering import ELSA, NMSELoss, recall_at_k, ndcg_at_k
import numpy as np


@pytest.fixture
def elsa() -> ELSA:
    return ELSA(n_items=20, latent_dim=16)


class TestELSA:
    def test_forward_shape(self, elsa: ELSA) -> None:
        x = torch.randn(8, 20)
        recon = elsa(x)
        assert recon.shape == (8, 20)

    def test_encode_shape(self, elsa: ELSA) -> None:
        x = torch.randn(8, 20)
        z = elsa.encode(x)
        assert z.shape == (8, 16)

    def test_encode_normalised(self, elsa: ELSA) -> None:
        x = torch.randn(8, 20)
        z = elsa.encode(x)
        norms = z.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_item_factor_shape(self, elsa: ELSA) -> None:
        assert elsa.A.shape == (20, 16)


class TestNMSELoss:
    def test_zero_for_identical(self) -> None:
        x = torch.randn(8, 20)
        loss = NMSELoss()(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_positive_for_different(self) -> None:
        pred = torch.randn(8, 20)
        target = torch.randn(8, 20)
        assert NMSELoss()(pred, target).item() >= 0.0


class TestEvalMetrics:
    def test_recall_perfect(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([1, 3, 0, 2, 4])
        assert recall_at_k(y_true, y_pred, k=2) == pytest.approx(1.0)

    def test_recall_zero(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 2, 4, 0, 2])
        assert recall_at_k(y_true, y_pred, k=1) == pytest.approx(0.0)

    def test_ndcg_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 0, 0])
        y_pred = np.array([2, 0, 1, 3, 4])
        assert ndcg_at_k(y_true, y_pred, k=1) == pytest.approx(1.0)

