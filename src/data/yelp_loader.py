"""Load and parse raw Yelp Open Dataset JSON files.

The Yelp dataset ships as several large JSON-lines files.  This module
provides helpers to load them into Pandas DataFrames and do initial
category-level filtering to keep only Points of Interest (POI) relevant to
leisure activities.

Expected raw-data layout (place files under ``data/raw/``)::

    data/raw/
        yelp_academic_dataset_business.json
        yelp_academic_dataset_review.json
        yelp_academic_dataset_user.json
        yelp_academic_dataset_checkin.json   (optional)
        yelp_academic_dataset_tip.json       (optional)

Download from: https://www.yelp.com/dataset
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

# Yelp categories considered as "Points of Interest" for this thesis
POI_CATEGORIES: frozenset[str] = frozenset(
    {
        "Restaurants",
        "Bars",
        "Nightlife",
        "Food",
        "Coffee & Tea",
        "Arts & Entertainment",
        "Active Life",
        "Hotels & Travel",
        "Shopping",
        "Beauty & Spas",
        "Local Services",
    }
)


# ── Low-level I/O ─────────────────────────────────────────────────────────

def _iter_json_lines(path: Path) -> Iterator[dict]:
    """Yield parsed JSON objects from a JSON-lines file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    """Load a Yelp JSON-lines file into a DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.json`` file.
    max_rows:
        If set, stop after reading this many rows (useful for quick tests).

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Yelp file not found: {path}\n"
            "Download the dataset from https://www.yelp.com/dataset "
            "and place it under data/raw/"
        )
    rows = []
    for i, record in enumerate(_iter_json_lines(path)):
        rows.append(record)
        if max_rows is not None and i + 1 >= max_rows:
            break
    logger.info("Loaded %d rows from %s", len(rows), path)
    return pd.DataFrame(rows)


# ── Business (POI) loading ────────────────────────────────────────────────

def load_businesses(
    raw_dir: str | Path = "data/raw",
    *,
    poi_categories: frozenset[str] | None = None,
    min_review_count: int = 5,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load and filter business records.

    Parameters
    ----------
    raw_dir:
        Directory containing ``yelp_academic_dataset_business.json``.
    poi_categories:
        Set of top-level Yelp categories to keep.
        Defaults to :data:`POI_CATEGORIES`.
    min_review_count:
        Drop businesses with fewer reviews than this threshold.
    max_rows:
        Cap on number of rows read (for quick iteration).

    Returns
    -------
    pd.DataFrame
        Filtered business records with at least the columns:
        ``business_id``, ``name``, ``city``, ``state``, ``stars``,
        ``review_count``, ``categories``, ``latitude``, ``longitude``.
    """
    if poi_categories is None:
        poi_categories = POI_CATEGORIES

    raw_dir = Path(raw_dir)
    df = load_jsonl(raw_dir / "yelp_academic_dataset_business.json", max_rows=max_rows)

    # Filter to open businesses with enough reviews
    df = df[df["is_open"] == 1].copy()
    df = df[df["review_count"] >= min_review_count].copy()

    # Filter by category (Yelp stores categories as a comma-separated string)
    def _has_poi_category(categories: str | None) -> bool:
        if not categories:
            return False
        return bool(poi_categories & {c.strip() for c in categories.split(",")})

    mask = df["categories"].apply(_has_poi_category)
    df = df[mask].reset_index(drop=True)
    logger.info("Kept %d POI businesses after filtering", len(df))
    return df


# ── Review loading ────────────────────────────────────────────────────────

def load_reviews(
    raw_dir: str | Path = "data/raw",
    *,
    business_ids: set[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load review records, optionally restricted to specific businesses.

    Parameters
    ----------
    raw_dir:
        Directory containing ``yelp_academic_dataset_review.json``.
    business_ids:
        If provided, keep only reviews for these business IDs.
    max_rows:
        Cap on number of rows read.

    Returns
    -------
    pd.DataFrame
        Columns: ``review_id``, ``user_id``, ``business_id``, ``stars``,
        ``useful``, ``funny``, ``cool``, ``date``.
    """
    raw_dir = Path(raw_dir)
    df = load_jsonl(raw_dir / "yelp_academic_dataset_review.json", max_rows=max_rows)

    keep_cols = ["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "date"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    if business_ids is not None:
        df = df[df["business_id"].isin(business_ids)].reset_index(drop=True)

    logger.info("Loaded %d reviews", len(df))
    return df


# ── User loading ──────────────────────────────────────────────────────────

def load_users(
    raw_dir: str | Path = "data/raw",
    *,
    user_ids: set[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load user records, optionally restricted to a subset.

    Parameters
    ----------
    raw_dir:
        Directory containing ``yelp_academic_dataset_user.json``.
    user_ids:
        If provided, keep only these users.
    max_rows:
        Cap on number of rows read.

    Returns
    -------
    pd.DataFrame
        Columns include ``user_id``, ``review_count``, ``average_stars``,
        ``friends``, etc.
    """
    raw_dir = Path(raw_dir)
    df = load_jsonl(raw_dir / "yelp_academic_dataset_user.json", max_rows=max_rows)

    if user_ids is not None:
        df = df[df["user_id"].isin(user_ids)].reset_index(drop=True)

    logger.info("Loaded %d users", len(df))
    return df
