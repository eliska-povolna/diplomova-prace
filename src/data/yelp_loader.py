"""Load and query Yelp data via DuckDB or CloudSQL.

This module provides helpers to query the Yelp dataset from:
  1. Database tables (yelp_business, yelp_review) - primary method

Expected database schema::

    yelp_business (business_id, name, state, review_count, stars, ...)
    yelp_review (review_id, user_id, business_id, stars, date, text, ...)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _list_tables(con) -> set[str]:
    """Return available table names in current DuckDB catalog."""
    rows = con.execute("SHOW TABLES").fetchall()
    return {str(row[0]) for row in rows}


def _resolve_table_name(con, candidates: list[str], db_path: str | Path) -> str:
    """Resolve first matching table name from candidates.

    Raises
    ------
    RuntimeError
        If none of the candidate names exist.
    """
    existing = _list_tables(con)
    for table_name in candidates:
        if table_name in existing:
            return table_name

    raise RuntimeError(
        "Required Yelp tables are missing in DuckDB. "
        f"Checked {candidates} in {Path(db_path).resolve()}. "
        f"Found tables: {sorted(existing)[:20]}. "
        "Run: python -m src.setup_database --json-dir <path-to-yelp-json>"
    )


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
    *,
    db_path: str | Path = "yelp.duckdb",
    state_filter: str | None = None,
    min_review_count: int = 5,
) -> pd.DataFrame:
    """Load business records from DuckDB table.

    Parameters
    ----------
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
    con = connect(db_path)
    try:
        business_table = _resolve_table_name(
            con,
            ["yelp_business", "businesses", "business"],
            db_path,
        )

        # Build WHERE clause using positional parameters
        where_clauses = ["review_count >= $1"]
        params: list = [int(min_review_count)]

        if state_filter is not None:
            where_clauses.append("state = $2")
            params.append(str(state_filter))

        where_sql = " AND ".join(where_clauses)

        # Read from database table
        query = f"SELECT * FROM {business_table} WHERE {where_sql}"
        df = con.execute(query, params).fetchdf()
        logger.info("Loaded %d businesses from database", len(df))

        return df
    finally:
        con.close()


def load_reviews(
    *,
    db_path: str | Path = "yelp.duckdb",
    business_ids: set[str] | None = None,
    pos_threshold: float = 4.0,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    """Load positive-feedback review records from DuckDB table.

    Parameters
    ----------
    db_path:
        DuckDB database file path.
    business_ids:
        If provided, restrict to reviews for these businesses.
    pos_threshold:
        Minimum star rating to treat as a positive interaction (default 4.0).
    year_min / year_max:
        Optional year-range filter on the review date.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id``, ``business_id``, ``ts`` (epoch ms), ``implicit``.
    """
    con = connect(db_path)

    try:
        review_table = _resolve_table_name(
            con,
            ["yelp_review", "reviews", "review"],
            db_path,
        )

        # Build WHERE clause using positional parameters
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

        # Optionally restrict to a specific set of business_ids
        if business_ids is not None:
            business_filter_df = pd.DataFrame({"business_id": list(business_ids)})
            con.register("business_filter", business_filter_df)
            join_clause = "JOIN business_filter USING (business_id)"
        else:
            join_clause = ""

        # Read from database table
        query = f"""
            SELECT user_id,
                   business_id,
                   epoch_ms(CAST(date AS TIMESTAMP)) AS ts,
                   1 AS implicit
            FROM {review_table}
            {join_clause}
            WHERE {where_sql}
        """
        df = con.execute(query, params).fetchdf()
        logger.info("Loaded %d reviews from database", len(df))

        return df
    finally:
        con.close()
