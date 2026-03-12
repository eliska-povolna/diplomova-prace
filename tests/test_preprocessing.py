"""Tests for src.data.preprocessing (CSR builder + ID maps)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import (
    build_id_map,
    build_csr,
    user_train_val_split,
    DatasetMaps,
)


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """Small implicit-feedback DataFrame (12 unique user–item pairs)."""
    data = {
        "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4", "u5", "u5", "u5", "u5"],
        "business_id": ["b1", "b2", "b1", "b3", "b2", "b4", "b3", "b5", "b1", "b2", "b3", "b4"],
    }
    return pd.DataFrame(data)


class TestBuildIdMap:
    def test_returns_series(self, sample_interactions: pd.DataFrame) -> None:
        m = build_id_map(sample_interactions["user_id"])
        assert hasattr(m, "index")

    def test_consecutive_integers(self, sample_interactions: pd.DataFrame) -> None:
        m = build_id_map(sample_interactions["user_id"])
        assert set(m.values) == set(range(len(m)))

    def test_unique_ids(self, sample_interactions: pd.DataFrame) -> None:
        m = build_id_map(sample_interactions["user_id"])
        assert m.index.nunique() == len(m)


class TestBuildCSR:
    def test_returns_dataset_maps(self, sample_interactions: pd.DataFrame) -> None:
        result = build_csr(sample_interactions)
        assert isinstance(result, DatasetMaps)

    def test_csr_shape(self, sample_interactions: pd.DataFrame) -> None:
        result = build_csr(sample_interactions)
        n_users = len(set(sample_interactions["user_id"]))
        n_items = len(set(sample_interactions["business_id"]))
        assert result.csr.shape == (n_users, n_items)

    def test_csr_is_binary(self, sample_interactions: pd.DataFrame) -> None:
        result = build_csr(sample_interactions)
        assert set(result.csr.data.tolist()).issubset({0.0, 1.0})

    def test_maps_cover_all_ids(self, sample_interactions: pd.DataFrame) -> None:
        result = build_csr(sample_interactions)
        assert set(result.user_map.keys()) == set(sample_interactions["user_id"].unique())
        assert set(result.item_map.keys()) == set(sample_interactions["business_id"].unique())

    def test_deduplicates_interactions(self) -> None:
        df = pd.DataFrame({
            "user_id": ["u1", "u1"],
            "business_id": ["b1", "b1"],  # duplicate
        })
        result = build_csr(df)
        assert result.csr.nnz == 1

    def test_nnz_matches_unique_pairs(self, sample_interactions: pd.DataFrame) -> None:
        n_unique = sample_interactions.drop_duplicates().shape[0]
        result = build_csr(sample_interactions)
        assert result.csr.nnz == n_unique


class TestUserTrainValSplit:
    def test_split_sizes(self, sample_interactions: pd.DataFrame) -> None:
        maps = build_csr(sample_interactions)
        train_idx, val_idx = user_train_val_split(maps.csr, val_ratio=0.2, seed=0)
        n_users = maps.csr.shape[0]
        assert len(train_idx) + len(val_idx) == n_users

    def test_no_overlap(self, sample_interactions: pd.DataFrame) -> None:
        maps = build_csr(sample_interactions)
        train_idx, val_idx = user_train_val_split(maps.csr, val_ratio=0.2, seed=0)
        assert len(set(train_idx) & set(val_idx)) == 0
