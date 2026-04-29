#!/usr/bin/env python3
"""
Resume Cloud SQL database reload with checkpoint support.

This script:
1. Checks what's already loaded in Cloud SQL
2. Resumes reviews from where it left off
3. Uses memory-efficient batching

Usage:
    python scripts/reset_and_reload_data_resume.py
    python scripts/reset_and_reload_data_resume.py --skip-businesses  # Only load reviews
"""

import os
import sys
import logging
import json
import gc
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import duckdb
    import pandas as pd
except ImportError:
    print("ERROR: duckdb or pandas not installed")
    print("Install with: pip install duckdb pandas")
    sys.exit(1)

from src.ui.services.cloud_sql_helper import get_cloud_sql_instance

CHECKPOINT_FILE = Path("reset_reload_checkpoint.json")
LOG_FILE = Path("reset_reload_resume.log")

# Setup dual logging: console + file
# Use UTF-8 encoding to support Unicode characters (emoji in messages)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# File handler with UTF-8 encoding
file_handler = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)

# Console handler with UTF-8 support
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Configure encoding for console output
try:
    # Python 3.7+: use reconfigure for UTF-8
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, TypeError):
    # Fallback: no reconfigure available
    pass

# Format for both handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()


def load_checkpoint() -> dict:
    """Load checkpoint data if it exists."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            logger.info(f"✅ Loaded checkpoint: Reviews inserted={data.get('reviews_inserted', 0)}")
            return data
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    return {"reviews_inserted": 0, "total_reviews": 0, "businesses_complete": False}


def save_checkpoint(stats: dict):
    """Save checkpoint data."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def check_db_state(helper):
    """Check what's already in the database."""
    try:
        import sqlalchemy
        
        with helper.engine.connect() as conn:
            # Check business count
            result = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM public.businesses"))
            business_count = result.scalar() or 0
            
            # Check review count
            result = conn.execute(sqlalchemy.text("SELECT COUNT(*) FROM public.reviews"))
            review_count = result.scalar() or 0
            
        logger.info(f"📊 Database state: {business_count} businesses, {review_count} reviews")
        return {"businesses": business_count, "reviews": review_count}
    except Exception as e:
        logger.warning(f"⚠️  Could not check DB state: {e}")
        return {"businesses": 0, "reviews": 0}


def load_review_data_resume(batch_size: int = 500, max_rows: int = None):
    """
    Resume loading Yelp review data from parquet to Cloud SQL.
    
    Args:
        batch_size: Batch size for inserts (reduced for memory efficiency)
        max_rows: Optional limit for testing
    """
    import sqlalchemy

    logger.info("🔄 Loading review data from parquet (RESUME MODE)...")

    try:
        helper = get_cloud_sql_instance()
    except Exception as e:
        logger.error(f"Failed to connect to Cloud SQL: {e}")
        return False

    # Check current state
    db_state = check_db_state(helper)
    checkpoint = load_checkpoint()
    
    # Find parquet data
    possible_paths = [
        Path("Yelp-JSON/yelp_parquet/review"),
        Path(__file__).parent.parent / "Yelp-JSON/yelp_parquet/review",
        Path(__file__).parent.parent.parent / "Yelp-JSON/yelp_parquet/review",
        Path(__file__).parent.parent.parent.parent / "Yelp-JSON/yelp_parquet/review",
    ]

    parquet_dir = None
    for path in possible_paths:
        if path.exists():
            parquet_dir = path
            logger.info(f"✅ Found Yelp parquet data at: {parquet_dir}")
            break

    if not parquet_dir:
        logger.warning(f"⚠️  Review parquet directory not found - skipping review load")
        return True

    # Load and insert data
    try:
        logger.info(f"Reading parquet files from: {parquet_dir}")

        # Check current state
        resume_from = checkpoint.get("reviews_inserted", 0)
        if db_state["reviews"] > resume_from:
            resume_from = db_state["reviews"]
            logger.info(f"⚠️  Database has {db_state['reviews']} reviews, updating resume point")

        logger.info(f"Resuming from review #{resume_from + 1}")

        # Only valid columns for reviews table
        valid_columns = ["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "text", "date"]
        
        logger.info(f"📝 Will use columns: {', '.join(valid_columns)}")
        logger.info(f"🚀 Starting batch insertion - reading parquet in chunks...")
        sys.stdout.flush()

        conn_duckdb = duckdb.connect(":memory:")
        parquet_pattern = str(parquet_dir / "**/*.parquet").replace("\\", "/")
        
        # Get total for progress tracking
        query = f"SELECT * FROM read_parquet('{parquet_pattern}')"
        df_count_result = conn_duckdb.execute(f"SELECT COUNT(*) FROM ({query})").fetchone()
        df_count = df_count_result[0] if df_count_result else 0
        
        if max_rows:
            df_count = min(max_rows, df_count)
        
        logger.info(f"Total reviews in parquet: {df_count:,}")

        if df_count == 0:
            logger.warning("No data found in review parquet files!")
            return True

        # Process in chunks using DuckDB streaming
        total_inserted = db_state["reviews"]
        skipped_count = 0
        failed_count = 0
        chunk_size = batch_size * 5  # 2500-5000 rows per chunk (5 batches per chunk)
        
        for offset in range(0, df_count, chunk_size):
            limit = min(chunk_size, df_count - offset)
            
            # Read chunk from parquet
            chunk_query = query + f" LIMIT {limit} OFFSET {offset}"
            df_chunk = conn_duckdb.execute(chunk_query).df()
            
            if len(df_chunk) == 0:
                break
            
            # Prepare rows in batches for insertion
            batch_rows = []
            
            for _, row in df_chunk.iterrows():
                try:
                    values = {col: row[col] for col in valid_columns if col in row and pd.notna(row[col])}
                    
                    if not values:
                        skipped_count += 1
                        continue
                    
                    batch_rows.append(values)
                    
                except Exception as e:
                    failed_count += 1
                    if failed_count <= 5:
                        logger.debug(f"Row prep error: {str(e)[:100]}")
            
            # Batch insert all rows from this chunk in a single transaction
            if batch_rows:
                try:
                    with helper.engine.begin() as conn:
                        # Build batch insert SQL
                        first_row = batch_rows[0]
                        col_names = list(first_row.keys())
                        placeholders = ', '.join([f':{col}' for col in col_names])
                        
                        insert_sql = (
                            f"INSERT INTO public.reviews ({', '.join(col_names)}) "
                            f"VALUES ({placeholders}) "
                            f"ON CONFLICT (review_id) DO NOTHING"
                        )
                        
                        # Execute batch - SQLAlchemy handles executemany for multiple rows
                        conn.execute(sqlalchemy.text(insert_sql), batch_rows)
                        total_inserted += len(batch_rows)
                        
                except Exception as e:
                    logger.error(f"Batch insert failed: {e}")
                    failed_count += len(batch_rows)
            
            # Progress report
            pct = 100 * (offset + len(df_chunk)) / df_count if df_count > 0 else 0
            logger.info(
                f"Progress: {offset + len(df_chunk):,}/{df_count:,} ({pct:.1f}%) "
                f"| Batch size: {len(batch_rows):,} | Total: {total_inserted:,} | Skipped: {skipped_count:,} | Failed: {failed_count:,}"
            )
            sys.stdout.flush()
            
            # Save checkpoint
            checkpoint = {
                "reviews_inserted": total_inserted,
                "total_reviews": df_count,
                "businesses_complete": True,
            }
            save_checkpoint(checkpoint)
            
            # Cleanup
            gc.collect()

        logger.info(f"✅ Loaded {total_inserted - db_state['reviews']:,} new reviews (Total: {total_inserted:,})")
        logger.info(f"   Skipped: {skipped_count}, Failed: {failed_count}")
        
        # Final checkpoint
        checkpoint = {
            "reviews_inserted": total_inserted,
            "total_reviews": df_count,
            "businesses_complete": True,
        }
        save_checkpoint(checkpoint)
        
        return True

    except Exception as e:
        logger.error(f"Error loading reviews: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Resume database reload")
    parser.add_argument(
        "--skip-businesses",
        action="store_true",
        help="Skip business loading, only load reviews",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for inserts (default: 500)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Limit review loading to N rows (for testing)",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete checkpoint and reset counts",
    )

    args = parser.parse_args()

    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("✓ Checkpoint deleted")

    logger.info("=" * 70)
    logger.info("🔄 RESUME DATABASE RELOAD - STARTED")
    logger.info("=" * 70)
    logger.info(f"📝 Logs being written to: {LOG_FILE.absolute()}")
    logger.info(f"💾 Checkpoint file: {CHECKPOINT_FILE.absolute()}")
    logger.info("=" * 70)
    sys.stdout.flush()

    # Check current state
    try:
        logger.info("🔌 Connecting to Cloud SQL...")
        helper = get_cloud_sql_instance()
        logger.info("✅ Connected to Cloud SQL!")
        db_state = check_db_state(helper)
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step: Load review data
    logger.info("=" * 70)
    logger.info("📤 Starting review data load...")
    logger.info("=" * 70)
    sys.stdout.flush()
    
    if not load_review_data_resume(batch_size=args.batch_size, max_rows=args.max_rows):
        logger.error("⚠️  Failed to load review data")
        return False

    logger.info("\n" + "=" * 70)
    logger.info("✅ Database reload resume complete!")
    logger.info(f"📝 Full logs at: {LOG_FILE.absolute()}")
    logger.info("=" * 70)
    sys.stdout.flush()
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
