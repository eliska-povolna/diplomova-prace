"""Database setup: Load raw Yelp JSON data into DuckDB or CloudSQL.

This is a one-time setup step. After running this, preprocessing and training
read data directly from the database (no Parquet intermediate).

Usage
-----
    # Load JSON into local DuckDB (default)
    python -m src.setup_database --json-dir path/to/yelp_json

    # Load JSON into CloudSQL
    python -m src.setup_database --json-dir path/to/yelp_json --cloud-sql

    # Check if data already exists and skip if present (default behavior)
    python -m src.setup_database --json-dir path/to/yelp_json --skip-if-exists

Database Schema
---------------
Tables created:
  - yelp_business (id, name, state, review_count, stars, ...)
  - yelp_review (review_id, user_id, business_id, stars, date, text, ...)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# Import from sibling modules
from .utils import load_config, setup_logger

logger = logging.getLogger(__name__)


def _get_db_path_from_config(config: dict) -> str:
    """Return DuckDB path with unified key precedence.

    Preferred key is data.db_path (used by training/evaluation).
    Legacy fallback: database.duckdb_path.
    """
    return (
        config.get("data", {}).get("db_path")
        or config.get("database", {}).get("duckdb_path")
        or "yelp.duckdb"
    )


def load_json_to_duckdb(json_dir: Path, db_path: str | Path) -> bool:
    """Load raw Yelp JSON files into DuckDB tables.

    Parameters
    ----------
    json_dir : Path
        Directory containing raw JSON files (yelp_academic_dataset_*.json)
    db_path : str | Path
        DuckDB database file path

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    from duckdb import connect

    logger.info("=" * 80)
    logger.info("LOADING JSON INTO DUCKDB")
    logger.info("=" * 80)

    con = None
    try:
        con = connect(str(db_path))

        # Load businesses.json
        logger.info("\n1. Loading yelp_academic_dataset_business.json...")
        business_json = json_dir / "yelp_academic_dataset_business.json"
        if not business_json.exists():
            logger.error(f"Not found: {business_json}")
            return False

        # Create table from JSON
        con.execute(
            f"""
            CREATE OR REPLACE TABLE yelp_business AS
            SELECT * FROM read_json_auto('{business_json}')
        """
        )
        n_businesses = con.execute("SELECT COUNT(*) FROM yelp_business").fetchall()[0][
            0
        ]
        logger.info(f"✓ Loaded {n_businesses:,} businesses")

        # Load reviews.json
        logger.info("\n2. Loading yelp_academic_dataset_review.json...")
        review_json = json_dir / "yelp_academic_dataset_review.json"
        if not review_json.exists():
            logger.error(f"Not found: {review_json}")
            return False

        con.execute(
            f"""
            CREATE OR REPLACE TABLE yelp_review AS
            SELECT * FROM read_json_auto('{review_json}')
        """
        )
        n_reviews = con.execute("SELECT COUNT(*) FROM yelp_review").fetchall()[0][0]
        logger.info(f"✓ Loaded {n_reviews:,} reviews")

        # Optional: Load users.json if it exists
        user_json = json_dir / "yelp_academic_dataset_user.json"
        if user_json.exists():
            logger.info("\n3. Loading yelp_academic_dataset_user.json...")
            con.execute(
                f"""
                CREATE OR REPLACE TABLE yelp_user AS
                SELECT * FROM read_json_auto('{user_json}')
            """
            )
            n_users = con.execute("SELECT COUNT(*) FROM yelp_user").fetchall()[0][0]
            logger.info(f"✓ Loaded {n_users:,} users")

        logger.info("\n✓ DuckDB database setup complete")
        return True

    except Exception as e:
        logger.error(f"Failed to load data into DuckDB: {e}")
        return False
    finally:
        if con is not None:
            con.close()


def load_json_to_cloudsql(
    json_dir: Path, connection_string: str, skip_if_exists: bool = True
) -> bool:
    """Load raw Yelp JSON files into CloudSQL.

    Parameters
    ----------
    json_dir : Path
        Directory containing raw JSON files
    connection_string : str
        CloudSQL connection string from config
    skip_if_exists : bool
        If True, check if tables exist and skip if they do

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    import psycopg2

    logger.info("=" * 80)
    logger.info("LOADING JSON INTO CLOUDSQL")
    logger.info("=" * 80)

    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()

        # Check if tables exist
        if skip_if_exists:
            cursor.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_name = 'yelp_business'
                )
            """
            )
            if cursor.fetchone()[0]:
                logger.info("✓ Tables already exist in CloudSQL. Skipping load.")
                cursor.close()
                conn.close()
                return True

        # Load businesses.json
        logger.info("\n1. Loading yelp_academic_dataset_business.json...")
        business_json = json_dir / "yelp_academic_dataset_business.json"
        if not business_json.exists():
            logger.error(f"Not found: {business_json}")
            return False

        # Create table and bulk load
        cursor.execute(
            """
            DROP TABLE IF EXISTS yelp_business CASCADE
        """
        )

        # Load JSON and insert
        with open(business_json) as f:
            businesses = [json.loads(line) for line in f]

        if businesses:
            df_biz = pd.DataFrame(businesses)
            # Convert to CSV format for COPY
            csv_buffer = df_biz.to_csv(index=False)
            cursor.copy_from(csv_buffer, "yelp_business")
            conn.commit()
            logger.info(f"✓ Loaded {len(businesses):,} businesses")

        # Similar for reviews
        logger.info("\n2. Loading yelp_academic_dataset_review.json...")
        review_json = json_dir / "yelp_academic_dataset_review.json"
        if not review_json.exists():
            logger.error(f"Not found: {review_json}")
            return False

        cursor.execute("DROP TABLE IF EXISTS yelp_review CASCADE")

        with open(review_json) as f:
            reviews = [json.loads(line) for line in f]

        if reviews:
            df_rev = pd.DataFrame(reviews)
            csv_buffer = df_rev.to_csv(index=False)
            cursor.copy_from(csv_buffer, "yelp_review")
            conn.commit()
            logger.info(f"✓ Loaded {len(reviews):,} reviews")

        cursor.close()
        conn.close()
        logger.info("\n✓ CloudSQL database setup complete")
        return True

    except Exception as e:
        logger.error(f"Failed to load data into CloudSQL: {e}")
        return False


def main() -> None:
    """Main setup entry point."""
    parser = argparse.ArgumentParser(
        description="Load raw Yelp JSON data into DuckDB or CloudSQL"
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        required=True,
        help="Path to directory containing yelp_academic_dataset_*.json files",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (for CloudSQL connection string)",
    )
    parser.add_argument(
        "--cloud-sql",
        action="store_true",
        help="Load into CloudSQL instead of local DuckDB",
    )
    parser.add_argument(
        "--no-skip-if-exists",
        action="store_false",
        dest="skip_if_exists",
        default=True,
        help="Force reload even if tables already exist (default: skip if exists)",
    )

    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    if not json_dir.exists():
        print(f"Error: JSON directory not found: {json_dir}")
        return

    # Set up logging
    setup_logger(__name__, level=logging.INFO)

    if args.cloud_sql:
        config = load_config(args.config)
        cloud_sql_conn = config.get("database", {}).get("cloud_sql_connection_string")
        if not cloud_sql_conn:
            logger.error(
                "CloudSQL connection string not found in config. "
                "Set database.cloud_sql_connection_string in your config."
            )
            return
        success = load_json_to_cloudsql(json_dir, cloud_sql_conn, args.skip_if_exists)
    else:
        config = load_config(args.config)
        db_path = _get_db_path_from_config(config)
        success = load_json_to_duckdb(json_dir, db_path)

    if not success:
        logger.error("Database setup failed")
        return

    logger.info("\n" + "=" * 80)
    logger.info("✓ Setup complete. Ready for preprocessing!")
    logger.info("=" * 80)
    logger.info(
        "Next step: python -m src.preprocess_data --config configs/default.yaml"
    )


if __name__ == "__main__":
    main()
