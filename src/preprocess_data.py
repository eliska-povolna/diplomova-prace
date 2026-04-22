"""Preprocessing: Build CSR matrix and ID mappings from database.

This script reads Yelp reviews/businesses from DuckDB or CloudSQL
and creates a sparse matrix representation for model training.

Data must already be loaded into the database (see setup_database.py).
Optionally runs database setup if --setup-database flag is provided.

Usage
-----
    # Standard: data already in database
    python -m src.preprocess_data --config configs/default.yaml

    # With auto-setup: load JSON into DB first, then preprocess
    python -m src.preprocess_data --config configs/default.yaml \\
        --setup-database --json-dir path/to/yelp_json

    # Preprocessed data saved to: data/preprocessed_yelp/
    # Contents:
    #   - processed_train.npz (scipy sparse CSR)
    #   - user2index.pkl (user_id → index mapping)
    #   - item2index.pkl (business_id → index mapping)
    #   - preprocessing_info.json (metadata)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from scipy.sparse import save_npz

# Import from sibling modules
from .data.yelp_loader import load_reviews, load_businesses
from .data.preprocessing import build_csr, save_dataset
from .utils import load_config, setup_logger

# Get project root for config/data paths
project_root = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)


def _apply_kcore_filtering_to_reviews(
    reviews,
    *,
    k: int = 5,
    max_iterations: int = 10,
):
    """Iteratively keep only user-item pairs that satisfy k-core filtering."""
    filtered = reviews[["user_id", "business_id"]].dropna().drop_duplicates()

    for iteration in range(max_iterations):
        old_count = len(filtered)

        user_counts = filtered["user_id"].value_counts()
        item_counts = filtered["business_id"].value_counts()

        keep_users = set(user_counts[user_counts >= k].index)
        keep_items = set(item_counts[item_counts >= k].index)

        filtered = filtered[
            filtered["user_id"].isin(keep_users)
            & filtered["business_id"].isin(keep_items)
        ].drop_duplicates()

        logger.info(
            "5-core (k=%d, iter=%d): %d interactions (removed %d)",
            k,
            iteration + 1,
            len(filtered),
            old_count - len(filtered),
        )

        if len(filtered) == old_count:
            logger.info("5-core filtering converged after %d iterations", iteration + 1)
            break

    return filtered.reset_index(drop=True)


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


def setup_database_if_needed(json_dir: Path, config: dict) -> bool:
    """Optionally initialize database from JSON files.

    Parameters
    ----------
    json_dir : Path
        Directory containing raw JSON files
    config : dict
        Config dict with database settings

    Returns
    -------
    bool
        True if successful or skipped, False if error
    """
    logger.info("=" * 80)
    logger.info("STEP 0: DATABASE SETUP (OPTIONAL)")
    logger.info("=" * 80)

    try:
        from .setup_database import load_json_to_duckdb, load_json_to_cloudsql

        use_cloud_sql = config.get("database", {}).get("use_cloud_sql", False)

        if use_cloud_sql:
            cloud_sql_conn = config.get("database", {}).get(
                "cloud_sql_connection_string"
            )
            if not cloud_sql_conn:
                logger.error(
                    "CloudSQL connection string not found in config. "
                    "Set database.cloud_sql_connection_string in your config."
                )
                return False
            return load_json_to_cloudsql(json_dir, cloud_sql_conn, skip_if_exists=True)
        else:
            db_path = _get_db_path_from_config(config)
            return load_json_to_duckdb(json_dir, db_path)

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def build_csr_matrices(config: dict, preprocess_dir: Path) -> bool:
    """Build universal CSR matrix and ID mappings from database.

    Parameters
    ----------
    config : dict
        Configuration dictionary with database settings
    preprocess_dir : Path
        Output directory for preprocessed data

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    logger.info("=" * 80)
    logger.info("STEP 1: BUILD CSR MATRIX AND ID MAPPINGS FROM DATABASE")
    logger.info("=" * 80)

    try:
        db_path = _get_db_path_from_config(config)

        # Load positive reviews only (stars >= 4.0)
        logger.info("\nLoading reviews from database...")
        reviews = load_reviews(
            db_path=db_path,
            pos_threshold=4.0,
            year_min=config["data"].get("year_min"),
            year_max=config["data"].get("year_max"),
        )

        logger.info(f"✓ Loaded {len(reviews):,} positive reviews (stars >= 4.0)")
        logger.info(
            f"  User-item pairs: {reviews[['user_id', 'business_id']].drop_duplicates().shape[0]:,}"
        )

        logger.info("\nApplying 5-core filtering (k=5)...")
        reviews = _apply_kcore_filtering_to_reviews(reviews, k=5)

        if reviews.empty:
            logger.error("No interactions remain after 5-core filtering")
            return False

        logger.info(f"✓ Retained {len(reviews):,} interactions after 5-core filtering")
        logger.info(
            f"  Remaining users: {reviews['user_id'].nunique():,}, items: {reviews['business_id'].nunique():,}"
        )

        # Load businesses for the items that survived 5-core filtering
        logger.info("\nLoading businesses from database...")
        businesses = load_businesses(
            db_path=db_path,
            state_filter=None,  # NO state filtering
            min_review_count=0,  # NO business-level filtering
        )

        businesses = businesses[businesses["business_id"].isin(reviews["business_id"])]

        logger.info(f"✓ Loaded {len(businesses):,} businesses after 5-core filtering")
        logger.info(f"  States: {businesses['state'].nunique()} unique")

        # Build CSR matrix
        logger.info("\nBuilding CSR matrix...")
        dataset = build_csr(reviews)

        logger.info(
            f"✓ Built CSR: {dataset.csr.shape[0]} users × {dataset.csr.shape[1]} items"
        )
        logger.info(f"  Interactions: {dataset.csr.nnz:,}")
        logger.info(
            f"  Density: {100.0 * dataset.csr.nnz / (dataset.csr.shape[0] * dataset.csr.shape[1]):.4f}%"
        )

        # Save to preprocessed directory
        logger.info(f"\nSaving preprocessed data...")
        save_dataset(dataset, preprocess_dir)

        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "n_users": int(dataset.csr.shape[0]),
            "n_items": int(dataset.csr.shape[1]),
            "n_interactions": int(dataset.csr.nnz),
            "n_reviews_loaded": len(reviews),
            "density_percent": float(
                100.0 * dataset.csr.nnz / (dataset.csr.shape[0] * dataset.csr.shape[1])
            ),
            "k_core": 5,
            "pos_threshold": 4.0,
            "note": "Positive interactions (stars >= 4.0) with iterative 5-core filtering.",
        }

        metadata_path = preprocess_dir / "preprocessing_info.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✓ Metadata saved: {metadata_path}")
        logger.info(f"\n{'='*80}")
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Output directory: {preprocess_dir}")
        logger.info(f"  - processed_train.npz (CSR matrix)")
        logger.info(f"  - user2index.pkl ({dataset.csr.shape[0]:,} users)")
        logger.info(f"  - item2index.pkl ({dataset.csr.shape[1]:,} items)")
        logger.info(f"  - preprocessing_info.json (metadata)")

        return True

    except Exception as e:
        logger.error(f"Failed to build CSR: {e}", exc_info=True)
        return False


def main() -> int:
    """Main preprocessing entry point."""
    parser = argparse.ArgumentParser(
        description="Yelp data preprocessing: Database → CSR + ID mappings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--setup-database",
        action="store_true",
        help="Setup database from JSON before preprocessing (optional, one-time only)",
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default=None,
        help="Path to JSON directory (required if --setup-database is set)",
    )

    parsed_args = parser.parse_args()

    # Setup
    config = load_config(parsed_args.config)
    preprocess_dir = project_root / "data" / "preprocessed_yelp"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logger(
        __name__,
        log_dir=preprocess_dir,
        level=logging.INFO,
    )

    logger.info("=" * 80)
    logger.info("YELP DATA PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Config: {parsed_args.config}")
    logger.info(f"Output directory: {preprocess_dir}")

    # Step 0: Optional database setup
    if parsed_args.setup_database:
        if not parsed_args.json_dir:
            logger.error("ERROR: --json-dir is required when using --setup-database")
            return 1

        json_dir = Path(parsed_args.json_dir)
        if not json_dir.exists():
            logger.error(f"JSON directory not found: {json_dir}")
            return 1

        if not setup_database_if_needed(json_dir, config):
            logger.error("Database setup failed")
            return 1

    # Step 1: Build CSR from database
    if not build_csr_matrices(config, preprocess_dir):
        logger.error("CSR matrix building failed")
        return 1

    logger.info("\n✓ All preprocessing steps completed successfully!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
