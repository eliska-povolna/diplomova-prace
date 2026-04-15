"""Load and query Yelp Parquet data via DuckDB.

This module provides helpers to query the Yelp dataset stored as Parquet
files (partitioned by state for businesses, by year for reviews).  It
wraps the DuckDB-based approach used in
``src/yelp_initial_exploration/yelp_build_csr.py``.

Expected Parquet layout (produced by converting Yelp JSON → Parquet)::

    yelp_parquet/
        business/state=XX/part-*.parquet
        review/year=YYYY/part-*.parquet
        user.parquet                          (optional)

See ``data/README.md`` for download and conversion instructions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def connect(db_path: str | Path = "yelp.duckdb"):
    """Open (or create) a DuckDB connection.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file.

    Returns
    -------
    duckdb.DuckDBPyConnection
    """
    try:
        import duckdb  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "duckdb is required. Install it with: pip install duckdb"
        ) from exc
    return duckdb.connect(str(db_path))


def load_businesses(
    parquet_dir: str | Path,
    *,
    db_path: str | Path = "yelp.duckdb",
    state_filter: str | None = None,
    min_review_count: int = 5,
) -> pd.DataFrame:
    """Load business records from Parquet via DuckDB.

    Parameters
    ----------
    parquet_dir:
        Root directory containing ``business/state=*/`` Parquet files.
    db_path:
        DuckDB database file path.
    state_filter:
        If given, keep only businesses from this US state code (e.g. ``"PA"``).
    min_review_count:
        Drop businesses with fewer reviews (uses the ``review_count`` column).

    Returns
    -------
    pd.DataFrame
    """
    parquet_dir = Path(parquet_dir)
    # Use POSIX-style paths for DuckDB and escape single quotes for SQL safety
    glob = (
        (parquet_dir / "business" / "state=*" / "*.parquet")
        .as_posix()
        .replace("'", "''")
    )

    con = connect(db_path)
    try:
        # Build the base query with a positional parameter for min_review_count
        where_clauses = ["review_count >= $1"]
        params: list = [int(min_review_count)]

        if state_filter is not None:
            # Use a second positional parameter to avoid any string injection
            where_clauses.append("state = $2")
            params.append(str(state_filter))

        where_sql = " AND ".join(where_clauses)
        df = con.execute(
            f"SELECT * FROM read_parquet('{glob}') WHERE {where_sql}",
            params,
        ).fetchdf()
        logger.info("Loaded %d businesses from %s", len(df), parquet_dir)
        return df
    finally:
        con.close()


def load_reviews(
    parquet_dir: str | Path,
    *,
    db_path: str | Path = "yelp.duckdb",
    business_ids: set[str] | None = None,
    pos_threshold: float = 4.0,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    """Load positive-feedback review records from Parquet via DuckDB.

    Parameters
    ----------
    parquet_dir:
        Root directory containing ``review/year=*/`` Parquet files.
    db_path:
        DuckDB database file path.
    business_ids:
        If provided, restrict to reviews for these businesses.
    pos_threshold:
        Minimum star rating to treat as a positive interaction.
    year_min / year_max:
        Optional year-range filter on the review date.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id``, ``business_id``, ``ts`` (epoch ms), ``implicit``.
    """
    parquet_dir = Path(parquet_dir)
    review_glob = str(parquet_dir / "review" / "year=*" / "*.parquet")

    con = connect(db_path)

    # Build WHERE clause using positional parameters for all user-supplied values
    where = [
        "user_id IS NOT NULL",
        "business_id IS NOT NULL",
        "TRY_CAST(stars AS DOUBLE) >= $1",
    ]
    params: list = [float(pos_threshold)]
    param_idx = 2

    if year_min is not None:
        where.append(f"CAST(strftime(date, '%Y') AS INTEGER) >= ${param_idx}")
        params.append(int(year_min))
        param_idx += 1
    if year_max is not None:
        where.append(f"CAST(strftime(date, '%Y') AS INTEGER) <= ${param_idx}")
        params.append(int(year_max))
        param_idx += 1

    where_sql = " AND ".join(where)

    # Optionally restrict to a specific set of business_ids inside DuckDB
    if business_ids is not None:
        business_filter_df = pd.DataFrame({"business_id": list(business_ids)})
        con.register("business_filter", business_filter_df)
        join_clause = "JOIN business_filter USING (business_id)"
    else:
        join_clause = ""

    query = f"""
        SELECT user_id,
               business_id,
               epoch_ms(CAST(date AS TIMESTAMP)) AS ts,
               1 AS implicit
        FROM read_parquet('{review_glob}')
        {join_clause}
        WHERE {where_sql}
    """

    try:
        df = con.execute(query, params).fetchdf()
    finally:
        con.close()

    logger.info("Loaded %d reviews from %s", len(df), parquet_dir)
    return df
