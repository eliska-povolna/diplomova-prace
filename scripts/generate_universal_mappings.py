#!/usr/bin/env python3
"""
Generate universal item/business mappings from raw Yelp data.

This script creates complete mappings for ALL businesses/items in the dataset,
regardless of what filtering was applied during training. This ensures that
any training run can be analyzed with complete item mappings in the labeling
notebook.

Usage
-----
    python scripts/generate_universal_mappings.py \\
        --parquet_dir /path/to/yelp_parquet \\
        --db_path /path/to/yelp.duckdb \\
        --output_dir /path/to/training/run

Example with defaults:
    python scripts/generate_universal_mappings.py
"""

import argparse
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path so we can import src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.yelp_loader import load_reviews
from src.utils import setup_logger

logger = logging.getLogger(__name__)


def main() -> None:
    """Generate universal mappings from raw Yelp data."""
    parser = argparse.ArgumentParser(
        description="Generate universal item/business mappings from Yelp data"
    )
    parser.add_argument(
        "--parquet_dir",
        type=Path,
        default=Path("../../Yelp-JSON/yelp_parquet"),
        help="Path to Yelp parquet directory",
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=Path("yelp.duckdb"),
        help="DuckDB database path",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for mappings (if None, uses latest outputs/ dir)",
    )
    parser.add_argument(
        "--pos_threshold",
        type=float,
        default=4.0,
        help="Minimum star rating for positive interaction",
    )
    parser.add_argument(
        "--year_min",
        type=int,
        default=None,
        help="Minimum year for reviews",
    )
    parser.add_argument(
        "--year_max",
        type=int,
        default=None,
        help="Maximum year for reviews",
    )

    args = parser.parse_args()

    # If output_dir not specified, use latest training run
    if args.output_dir is None:
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            run_dirs = sorted(
                [d for d in outputs_dir.iterdir() if d.is_dir()], reverse=True
            )
            if run_dirs:
                args.output_dir = run_dirs[0]
                print(f"Using latest training run: {args.output_dir.name}")
            else:
                print("No training runs found in outputs/ directory")
                return
        else:
            print("outputs/ directory not found")
            return

    # Create mappings subdirectory
    mappings_dir = args.output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logger(
        __name__,
        log_dir=mappings_dir,
        level=logging.INFO,
    )

    logger.info("=" * 80)
    logger.info("GENERATING UNIVERSAL ITEM/BUSINESS MAPPINGS")
    logger.info("=" * 80)
    logger.info(f"Parquet dir:    {args.parquet_dir}")
    logger.info(f"DB path:        {args.db_path}")
    logger.info(f"Output dir:     {args.output_dir}")
    logger.info(f"Pos threshold:  {args.pos_threshold}")
    if args.year_min or args.year_max:
        logger.info(f"Year range:     {args.year_min or '?'} - {args.year_max or '?'}")

    try:
        # Load ALL reviews (no filtering other than rating/year)
        logger.info("\n[1/3] Loading all reviews from Parquet...")
        reviews = load_reviews(
            args.parquet_dir,
            db_path=args.db_path,
            pos_threshold=args.pos_threshold,
            year_min=args.year_min,
            year_max=args.year_max,
        )
        logger.info(f"  Loaded {len(reviews):,} positive reviews")

        # Build universal mappings
        logger.info("\n[2/3] Building universal mappings...")

        all_users = reviews["user_id"].unique()
        all_businesses = reviews["business_id"].unique()

        universal_user_map = {uid: idx for idx, uid in enumerate(all_users)}
        universal_business_map = {bid: idx for idx, bid in enumerate(all_businesses)}

        logger.info(f"  Total unique users: {len(universal_user_map):,}")
        logger.info(f"  Total unique businesses: {len(universal_business_map):,}")

        # Save universal mappings
        logger.info("\n[3/3] Saving universal mappings...")

        user_map_path = mappings_dir / "user2index_universal.pkl"
        with open(user_map_path, "wb") as f:
            pickle.dump(universal_user_map, f)
        logger.info(f"  ✓ Saved: {user_map_path}")

        business_map_path = mappings_dir / "business2index_universal.pkl"
        with open(business_map_path, "wb") as f:
            pickle.dump(universal_business_map, f)
        logger.info(f"  ✓ Saved: {business_map_path}")

        logger.info("\n" + "=" * 80)
        logger.info("✓ UNIVERSAL MAPPINGS GENERATED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(
            f"\nThese mappings cover ALL {len(universal_business_map):,} businesses in the dataset."
        )
        logger.info(
            "The labeling notebook will automatically use these instead of filtered mappings."
        )

    except Exception as e:
        logger.error(f"Error generating universal mappings: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
