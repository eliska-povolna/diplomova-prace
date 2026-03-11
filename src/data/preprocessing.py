"""Feature engineering, ID mapping, and train/validation/test splitting.

This module assumes that raw DataFrames have already been loaded via
:mod:`src.data.yelp_loader`.  It produces the integer-indexed interaction
tensors that the model training code expects.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── ID mapping ────────────────────────────────────────────────────────────

class IDMapper(NamedTuple):
    """Bidirectional mapping between raw string IDs and integer indices."""

    id_to_idx: dict[str, int]
    idx_to_id: dict[int, str]

    @classmethod
    def from_series(cls, series: pd.Series) -> "IDMapper":
        unique_ids = sorted(series.unique())
        id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
        idx_to_id = {i: uid for uid, i in id_to_idx.items()}
        return cls(id_to_idx=id_to_idx, idx_to_id=idx_to_id)

    def __len__(self) -> int:
        return len(self.id_to_idx)


def build_id_mappers(
    reviews: pd.DataFrame,
) -> tuple[IDMapper, IDMapper]:
    """Build user and item ID mappers from a review DataFrame.

    Parameters
    ----------
    reviews:
        DataFrame with at least ``user_id`` and ``business_id`` columns.

    Returns
    -------
    user_mapper, item_mapper:
        :class:`IDMapper` instances for users and businesses respectively.
    """
    user_mapper = IDMapper.from_series(reviews["user_id"])
    item_mapper = IDMapper.from_series(reviews["business_id"])
    logger.info(
        "Built ID mappers: %d users, %d items", len(user_mapper), len(item_mapper)
    )
    return user_mapper, item_mapper


# ── Interaction filtering & binarisation ─────────────────────────────────

def filter_interactions(
    reviews: pd.DataFrame,
    *,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    implicit_threshold: int = 4,
) -> pd.DataFrame:
    """Filter reviews to create a clean implicit-feedback dataset.

    Parameters
    ----------
    reviews:
        Raw review DataFrame with ``user_id``, ``business_id``, ``stars``.
    min_user_interactions:
        Drop users with fewer than this many positive interactions.
    min_item_interactions:
        Drop items with fewer than this many positive interactions.
    implicit_threshold:
        Minimum star rating to treat as a positive interaction.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with ``user_id``, ``business_id``, ``interaction``
        (always 1 after binarisation).
    """
    df = reviews[reviews["stars"] >= implicit_threshold][
        ["user_id", "business_id"]
    ].drop_duplicates()

    # Iterative k-core filtering
    prev_len = -1
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)]
        item_counts = df["business_id"].value_counts()
        df = df[df["business_id"].isin(item_counts[item_counts >= min_item_interactions].index)]

    df = df.copy()
    df["interaction"] = 1
    logger.info(
        "After filtering: %d interactions (%d users, %d items)",
        len(df),
        df["user_id"].nunique(),
        df["business_id"].nunique(),
    )
    return df.reset_index(drop=True)


# ── Train / validation / test split ──────────────────────────────────────

class InteractionSplit(NamedTuple):
    """Container for train/val/test interaction DataFrames."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def temporal_split(
    interactions: pd.DataFrame,
    reviews_with_dates: pd.DataFrame,
    *,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> InteractionSplit:
    """Split interactions temporally (latest interactions → val/test).

    Each user's interactions are sorted by date; the most recent fraction
    goes to test, the second most recent to validation, and the rest to train.

    Parameters
    ----------
    interactions:
        Filtered interaction DataFrame (``user_id``, ``business_id``).
    reviews_with_dates:
        Original review DataFrame including a ``date`` column.
    val_ratio:
        Fraction of each user's interactions reserved for validation.
    test_ratio:
        Fraction of each user's interactions reserved for testing.

    Returns
    -------
    InteractionSplit
    """
    # Attach dates
    date_lookup = reviews_with_dates.set_index(["user_id", "business_id"])["date"]
    df = interactions.copy()
    df["date"] = df.set_index(["user_id", "business_id"]).index.map(date_lookup)
    df = df.sort_values("date")

    train_rows, val_rows, test_rows = [], [], []

    for _, group in df.groupby("user_id", sort=False):
        n = len(group)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))
        n_train = n - n_val - n_test
        if n_train <= 0:
            train_rows.append(group)
            continue
        train_rows.append(group.iloc[:n_train])
        val_rows.append(group.iloc[n_train : n_train + n_val])
        test_rows.append(group.iloc[n_train + n_val :])

    return InteractionSplit(
        train=pd.concat(train_rows).reset_index(drop=True),
        val=pd.concat(val_rows).reset_index(drop=True),
        test=pd.concat(test_rows).reset_index(drop=True),
    )


# ── Negative sampling ─────────────────────────────────────────────────────

def sample_negatives(
    interactions: pd.DataFrame,
    n_items: int,
    n_neg_per_pos: int = 1,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Uniform negative sampling for BPR training.

    For each positive ``(user, item)`` pair, sample ``n_neg_per_pos``
    items that the user has **not** interacted with.

    Parameters
    ----------
    interactions:
        Positive interactions with integer ``user_idx`` and ``item_idx``.
    n_items:
        Total number of items in the catalogue.
    n_neg_per_pos:
        Number of negative samples per positive.
    rng:
        Optional NumPy random generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_idx``, ``pos_item_idx``, ``neg_item_idx``.
    """
    if rng is None:
        rng = np.random.default_rng()

    user_positives: dict[int, set[int]] = (
        interactions.groupby("user_idx")["item_idx"]
        .apply(set)
        .to_dict()
    )

    records: list[dict[str, int]] = []
    for _, row in interactions.iterrows():
        u = int(row["user_idx"])
        pos = int(row["item_idx"])
        pos_set = user_positives.get(u, set())
        for _ in range(n_neg_per_pos):
            neg = rng.integers(n_items)
            while neg in pos_set:
                neg = rng.integers(n_items)
            records.append({"user_idx": u, "pos_item_idx": pos, "neg_item_idx": int(neg)})

    return pd.DataFrame(records)


# ── Utility: add integer indices to interaction DataFrame ─────────────────

def add_integer_indices(
    interactions: pd.DataFrame,
    user_mapper: IDMapper,
    item_mapper: IDMapper,
) -> pd.DataFrame:
    """Add ``user_idx`` and ``item_idx`` columns using pre-built mappers."""
    df = interactions.copy()
    df["user_idx"] = df["user_id"].map(user_mapper.id_to_idx)
    df["item_idx"] = df["business_id"].map(item_mapper.id_to_idx)
    return df
