"""Build CSR interaction matrix and ID maps from filtered review DataFrames.

This module complements ``src/data/yelp_loader.py``.  It mirrors the
logic in ``src/yelp_initial_exploration/yelp_build_csr.py`` but exposes
it as importable Python functions rather than a CLI script.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, save_npz

logger = logging.getLogger(__name__)


# ── ID mapping ────────────────────────────────────────────────────────────

def build_id_map(series: pd.Series) -> pd.Series:
    """Build a ``{raw_id → integer_index}`` mapping from a Series.

    Parameters
    ----------
    series:
        Series of raw string IDs (may contain duplicates).

    Returns
    -------
    pd.Series
        Index = unique raw IDs, values = consecutive integers starting at 0.
    """
    unique = series.drop_duplicates().reset_index(drop=True)
    return pd.Series(index=unique.values, data=np.arange(len(unique)), name="idx")


class DatasetMaps(NamedTuple):
    """Container for user/item ID maps and CSR matrix."""

    user_map: dict[str, int]
    item_map: dict[str, int]
    csr: csr_matrix


# ── CSR builder ───────────────────────────────────────────────────────────

def build_csr(
    interactions: pd.DataFrame,
    *,
    user_col: str = "user_id",
    item_col: str = "business_id",
) -> DatasetMaps:
    """Build a sparse CSR user–item interaction matrix.

    Parameters
    ----------
    interactions:
        DataFrame with at least ``user_col`` and ``item_col`` columns.
        Each row is treated as a positive (implicit = 1) interaction.
        Duplicate ``(user, item)`` pairs are automatically deduplicated.
    user_col:
        Name of the user-ID column.
    item_col:
        Name of the item-ID column.

    Returns
    -------
    DatasetMaps
        Named tuple with ``user_map``, ``item_map``, and ``csr``.
    """
    df = interactions[[user_col, item_col]].drop_duplicates()

    if df.empty:
        logger.info("No interactions after deduplication; returning empty CSR.")
        empty_mat = csr_matrix((0, 0), dtype=np.float32)
        return DatasetMaps(user_map={}, item_map={}, csr=empty_mat)

    uid_map = build_id_map(df[user_col])
    iid_map = build_id_map(df[item_col])

    u = df[user_col].map(uid_map).astype(int).values
    i = df[item_col].map(iid_map).astype(int).values
    vals = np.ones(len(df), dtype=np.float32)

    n_users = int(u.max()) + 1
    n_items = int(i.max()) + 1
    mat = coo_matrix((vals, (u, i)), shape=(n_users, n_items)).tocsr()

    logger.info(
        "Built CSR: %d users × %d items, %d interactions (density=%.4f%%)",
        n_users, n_items, mat.nnz, 100.0 * mat.nnz / (n_users * n_items),
    )
    return DatasetMaps(
        user_map=uid_map.to_dict(),
        item_map=iid_map.to_dict(),
        csr=mat,
    )


# ── Persistence ───────────────────────────────────────────────────────────

def save_dataset(maps: DatasetMaps, out_dir: str | Path) -> None:
    """Persist CSR matrix and ID maps to ``out_dir``.

    Outputs
    -------
    ``processed_train.npz`` — scipy sparse CSR in NPZ format
    ``user2index.pkl``       — ``{user_id: int}`` mapping
    ``item2index.pkl``       — ``{business_id: int}`` mapping
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_npz(out_dir / "processed_train.npz", maps.csr)
    with (out_dir / "user2index.pkl").open("wb") as f:
        pickle.dump(maps.user_map, f)
    with (out_dir / "item2index.pkl").open("wb") as f:
        pickle.dump(maps.item_map, f)
    logger.info("Saved dataset artefacts to %s", out_dir)


def load_dataset(out_dir: str | Path) -> DatasetMaps:
    """Load previously saved CSR matrix and ID maps.

    Parameters
    ----------
    out_dir:
        Directory containing ``processed_train.npz``, ``user2index.pkl``,
        ``item2index.pkl``.

    Returns
    -------
    DatasetMaps
    """
    from scipy.sparse import load_npz  # noqa: PLC0415

    out_dir = Path(out_dir)
    csr = load_npz(out_dir / "processed_train.npz")
    with (out_dir / "user2index.pkl").open("rb") as f:
        user_map: dict[str, int] = pickle.load(f)
    with (out_dir / "item2index.pkl").open("rb") as f:
        item_map: dict[str, int] = pickle.load(f)
    return DatasetMaps(user_map=user_map, item_map=item_map, csr=csr)


# ── Train / validation split ──────────────────────────────────────────────

def user_train_val_split(
    csr: csr_matrix,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split user indices into train and validation sets.

    Parameters
    ----------
    csr:
        Full user–item CSR matrix.
    val_ratio:
        Fraction of users to use for validation.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    train_indices, val_indices:
        Arrays of integer user indices.
    """
    from sklearn.model_selection import train_test_split  # noqa: PLC0415

    all_idx = np.arange(csr.shape[0])
    train_idx, val_idx = train_test_split(
        all_idx, test_size=val_ratio, random_state=seed
    )
    return train_idx, val_idx

