"""Preprocessing: Build CSR matrix and ID mappings from database.

This script prepares the sparse matrix representation used by the training and
evaluation pipeline. Data must already be loaded into the database
(`setup_database.py` can do that first when requested).

Usage
-----
    # Standard: data already in database
    python -m src.preprocess_data --config configs/default.yaml

    # With auto-setup: load JSON into DB first, then preprocess
    python -m src.preprocess_data --config configs/default.yaml \\
        --setup-database --json-dir path/to/yelp_json

Outputs
-------
    Preprocessed data saved to: data/preprocessed_yelp/
      - processed_train.npz (scipy sparse CSR)
      - user2index.pkl (user_id -> index mapping)
      - item2index.pkl (business_id -> index mapping)
      - preprocessing_info.json (metadata)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .data.preprocessing import save_dataset
from .data.shared_preprocessing_cache import (
    prepare_shared_preprocessing_cache,
    shared_preprocessing_manifest_path,
)
from .utils import load_config, setup_logger

project_root = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _get_db_path_from_config(config: dict) -> str:
    return (
        config.get("data", {}).get("db_path")
        or config.get("database", {}).get("duckdb_path")
        or "yelp.duckdb"
    )


def setup_database_if_needed(json_dir: Path, config: dict) -> bool:
    logger.info("=" * 80)
    logger.info("STEP 0: DATABASE SETUP (OPTIONAL)")
    logger.info("=" * 80)

    try:
        from .setup_database import load_json_to_cloudsql, load_json_to_duckdb

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

        db_path = _get_db_path_from_config(config)
        return load_json_to_duckdb(json_dir, db_path)
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def build_csr_matrices(config: dict, preprocess_dir: Path) -> bool:
    logger.info("=" * 80)
    logger.info("STEP 1: BUILD CSR MATRIX AND ID MAPPINGS FROM DATABASE")
    logger.info("=" * 80)

    try:
        config["data"]["db_path"] = _get_db_path_from_config(config)
        payload, source, cache_dir = prepare_shared_preprocessing_cache(
            config, require_existing=False
        )
        dataset = payload["final_dataset"]
        reviews = payload["reviews"]

        logger.info("Preparing preprocessed data...")
        logger.info("Data source: %s", source)

        save_dataset(dataset, preprocess_dir)

        metadata = {
            "source": source,
            "cache_dir": str(cache_dir),
            "cache_manifest": str(shared_preprocessing_manifest_path(cache_dir)),
            "n_users": int(dataset.csr.shape[0]),
            "n_items": int(dataset.csr.shape[1]),
            "n_interactions": int(dataset.csr.nnz),
            "n_reviews_loaded": len(reviews),
            "density_percent": float(
                100.0 * dataset.csr.nnz / (dataset.csr.shape[0] * dataset.csr.shape[1])
            ),
            "k_core": payload["manifest"]["signature"].get("k_core"),
            "pos_threshold": payload["manifest"]["signature"].get("pos_threshold"),
            "state_filter": payload["manifest"]["signature"].get("state_filter"),
            "year_min": payload["manifest"]["signature"].get("year_min"),
            "year_max": payload["manifest"]["signature"].get("year_max"),
            "signature": payload["manifest"]["signature"],
            "counts": payload["manifest"]["counts"],
            "note": "Preprocessed interaction matrix and mappings for downstream pipeline use.",
        }

        metadata_path = preprocess_dir / "preprocessing_info.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("PREPROCESSING COMPLETE")
        logger.info("Output directory: %s", preprocess_dir)
        logger.info("  - processed_train.npz (CSR matrix)")
        logger.info("  - user2index.pkl (%d users)", dataset.csr.shape[0])
        logger.info("  - item2index.pkl (%d items)", dataset.csr.shape[1])
        logger.info("  - preprocessing_info.json (metadata)")
        return True
    except Exception as e:
        logger.error(f"Failed to build CSR: {e}", exc_info=True)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Yelp data preprocessing: Database -> CSR + ID mappings",
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

    config = load_config(parsed_args.config)
    preprocess_dir = project_root / "data" / "preprocessed_yelp"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(__name__, log_dir=preprocess_dir, level=logging.INFO)

    logger.info("=" * 80)
    logger.info("YELP DATA PREPROCESSING")
    logger.info("=" * 80)
    logger.info("Config: %s", parsed_args.config)
    logger.info("Output directory: %s", preprocess_dir)

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

    if not build_csr_matrices(config, preprocess_dir):
        logger.error("CSR matrix building failed")
        return 1

    logger.info("All preprocessing steps completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
