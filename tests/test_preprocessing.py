"""Tests for src.data.preprocessing utilities (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    IDMapper,
    add_integer_indices,
    build_id_mappers,
    filter_interactions,
    sample_negatives,
)


@pytest.fixture
def sample_reviews() -> pd.DataFrame:
    """6 users × 6 businesses with one missing interaction per (user, item) pair.

    User u_i does NOT interact with b_i (diagonal missing), giving each user
    5 interactions (≥ min_user_interactions=5) and each business 5 interactions
    (≥ min_item_interactions=5), while leaving exactly one valid negative item
    per user for ``sample_negatives``.
    """
    users = [f"u{i}" for i in range(1, 7)]
    businesses = [f"b{j}" for j in range(1, 7)]
    records = [
        {"user_id": u, "business_id": b, "stars": 5}
        for k, u in enumerate(users)
        for j, b in enumerate(businesses)
        if j != k  # skip diagonal: u_k doesn't rate b_k
    ]
    return pd.DataFrame(records)


class TestIDMapper:
    def test_from_series(self) -> None:
        series = pd.Series(["c", "a", "b", "a"])
        mapper = IDMapper.from_series(series)
        assert len(mapper) == 3
        assert set(mapper.id_to_idx.keys()) == {"a", "b", "c"}
        # idx_to_id is the inverse
        for uid, idx in mapper.id_to_idx.items():
            assert mapper.idx_to_id[idx] == uid


class TestFilterInteractions:
    def test_returns_positive_interactions(self, sample_reviews: pd.DataFrame) -> None:
        result = filter_interactions(sample_reviews, implicit_threshold=4)
        assert (result["interaction"] == 1).all()

    def test_drops_low_rated(self, sample_reviews: pd.DataFrame) -> None:
        # Add a 1-star review for a brand-new business — it should be filtered
        # out both by the star threshold AND by k-core (only 1 interaction).
        extra = pd.DataFrame({"user_id": ["u1"], "business_id": ["b_new"], "stars": [1]})
        reviews = pd.concat([sample_reviews, extra], ignore_index=True)
        result = filter_interactions(reviews, implicit_threshold=4)
        assert "b_new" not in result["business_id"].values

    def test_kcore_filter(self) -> None:
        # Create a user with only 1 interaction — should be filtered out
        reviews = pd.DataFrame(
            {
                "user_id": ["u_rare", "u1", "u1", "u1", "u1", "u1"],
                "business_id": ["b1", "b1", "b2", "b3", "b4", "b5"],
                "stars": [5, 5, 5, 5, 5, 5],
            }
        )
        result = filter_interactions(reviews, min_user_interactions=5)
        assert "u_rare" not in result["user_id"].values


class TestBuildIDMappers:
    def test_mapper_sizes(self, sample_reviews: pd.DataFrame) -> None:
        interactions = filter_interactions(sample_reviews)
        user_mapper, item_mapper = build_id_mappers(interactions)
        assert len(user_mapper) == interactions["user_id"].nunique()
        assert len(item_mapper) == interactions["business_id"].nunique()


class TestAddIntegerIndices:
    def test_adds_columns(self, sample_reviews: pd.DataFrame) -> None:
        interactions = filter_interactions(sample_reviews)
        user_mapper, item_mapper = build_id_mappers(interactions)
        result = add_integer_indices(interactions, user_mapper, item_mapper)
        assert "user_idx" in result.columns
        assert "item_idx" in result.columns

    def test_indices_in_range(self, sample_reviews: pd.DataFrame) -> None:
        interactions = filter_interactions(sample_reviews)
        user_mapper, item_mapper = build_id_mappers(interactions)
        result = add_integer_indices(interactions, user_mapper, item_mapper)
        assert result["user_idx"].between(0, len(user_mapper) - 1).all()
        assert result["item_idx"].between(0, len(item_mapper) - 1).all()


class TestSampleNegatives:
    def test_output_columns(self, sample_reviews: pd.DataFrame) -> None:
        interactions = filter_interactions(sample_reviews)
        user_mapper, item_mapper = build_id_mappers(interactions)
        interactions = add_integer_indices(interactions, user_mapper, item_mapper)
        negs = sample_negatives(interactions, n_items=len(item_mapper), rng=np.random.default_rng(42))
        assert set(negs.columns) >= {"user_idx", "pos_item_idx", "neg_item_idx"}

    def test_neg_not_same_as_pos(self, sample_reviews: pd.DataFrame) -> None:
        interactions = filter_interactions(sample_reviews)
        user_mapper, item_mapper = build_id_mappers(interactions)
        interactions = add_integer_indices(interactions, user_mapper, item_mapper)
        negs = sample_negatives(interactions, n_items=len(item_mapper), rng=np.random.default_rng(42))
        # Negative item must differ from positive item
        assert (negs["neg_item_idx"] != negs["pos_item_idx"]).all()
