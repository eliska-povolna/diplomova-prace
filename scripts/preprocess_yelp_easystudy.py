#!/usr/bin/env python3
"""
Preprocess Yelp dataset to match the format of EasyStudy recommender algorithm.

This script converts Yelp rating data into the exact same format as the
reference preprocessing script (preprocess.py from EasyStudy project), ensuring
compatibility with the ELSA/EASE recommendation algorithms.

The output format matches:
  - processed_train.pkl: scipy.sparse.csr_matrix (users x items)
  - processed_test.pkl:  scipy.sparse.csr_matrix (users x items)
  - item2index.pkl:      {business_id: int} mapping

Generated data can be used directly in recommendation algorithms that expect:
  - Implicit feedback (binary 0/1 interactions)
  - Contiguous user/item indices (0 to N-1)
  - Disjoint train/test user sets
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

# Try to import from src.data or add fallback
try:
    from src.data.yelp_loader import load_reviews, load_businesses
except ImportError:
    print("Warning: Could not import from src.data.yelp_loader")
    print("Make sure you run from repo root or adjust PYTHONPATH")
    import sys
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Yelp data for EasyStudy recommendation algorithm"
    )
    parser.add_argument(
        "--parquet_dir",
        type=Path,
        required=True,
        help="Path to Yelp parquet directory (contains business/, review/ subdirs)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/processed_yelp_easystudy"),
        help="Output directory for processed data (default: data/processed_yelp_easystudy)",
    )
    parser.add_argument(
        "--db_path",
        type=Path,
        default=Path("yelp.duckdb"),
        help="DuckDB database path (default: yelp.duckdb)",
    )
    parser.add_argument(
        "--pos_threshold",
        type=float,
        default=4.0,
        help="Minimum star rating to treat as positive interaction (default: 4.0)",
    )
    parser.add_argument(
        "--min_business_reviews",
        type=int,
        default=100,
        help="Minimum number of reviews per business (default: 100)",
    )
    parser.add_argument(
        "--min_user_interactions",
        type=int,
        default=100,
        help="Minimum number of interactions per user (default: 100)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.1,
        help="Test set proportion (default: 0.1)",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--state_filter",
        type=str,
        default=None,
        help="Optional: filter to specific US state code (e.g. 'CA', 'TX')",
    )
    parser.add_argument(
        "--year_min",
        type=int,
        default=None,
        help="Optional: minimum year for reviews",
    )
    parser.add_argument(
        "--year_max",
        type=int,
        default=None,
        help="Optional: maximum year for reviews",
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Yelp Data Preprocessing for EasyStudy")
    print("=" * 80)
    print(f"Parquet dir:        {args.parquet_dir}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Positive threshold: {args.pos_threshold}")
    print(f"Min business revs:  {args.min_business_reviews}")
    print(f"Min user actions:   {args.min_user_interactions}")
    print(f"Test size:          {args.test_size}")
    print(f"Random seed:        {args.random_state}")
    if args.state_filter:
        print(f"State filter:       {args.state_filter}")
    if args.year_min or args.year_max:
        print(f"Year range:         {args.year_min or '?'} - {args.year_max or '?'}")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Load reviews from Parquet
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/6] Loading reviews from Parquet...")
    df = load_reviews(
        args.parquet_dir,
        db_path=args.db_path,
        pos_threshold=args.pos_threshold,
        year_min=args.year_min,
        year_max=args.year_max,
    )
    print(f"  Loaded {len(df):,} positive reviews")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    if df.empty:
        print("  ERROR: No interactions after filtering!")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Filter businesses by minimum review count
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[2/6] Filtering businesses with <{args.min_business_reviews} reviews...")
    before_filter = len(df)

    business_counts = df.groupby("business_id").size()
    popular_businesses = business_counts[
        business_counts >= args.min_business_reviews
    ].index
    df = df[df["business_id"].isin(popular_businesses)]

    print(f"  Before: {before_filter:,} reviews, {business_counts.shape[0]:,} businesses")
    print(f"  After:  {len(df):,} reviews, {len(popular_businesses):,} businesses")
    print(f"  Removed: {before_filter - len(df):,} reviews")

    if df.empty:
        print("  ERROR: No interactions after business filtering!")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # 2b. Filter users by minimum interaction count
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n[2b/6] Filtering users with <{args.min_user_interactions} interactions...")
    before_user_filter = len(df)

    user_counts = df.groupby("user_id").size()
    active_users = user_counts[user_counts >= args.min_user_interactions].index
    df = df[df["user_id"].isin(active_users)]

    print(f"  Before: {before_user_filter:,} reviews, {user_counts.shape[0]:,} users")
    print(f"  After:  {len(df):,} reviews, {len(active_users):,} users")
    print(f"  Removed: {before_user_filter - len(df):,} reviews")

    if df.empty:
        print("  ERROR: No interactions after user filtering!")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Remap user_id and business_id to contiguous indices
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[3/6] Remapping IDs to contiguous indices...")

    user_ids = df["user_id"].unique()
    business_ids = df["business_id"].unique()

    user2index = {uid: idx for idx, uid in enumerate(user_ids)}
    business2index = {bid: idx for idx, bid in enumerate(business_ids)}

    df["user_idx"] = df["user_id"].map(user2index)
    df["business_idx"] = df["business_id"].map(business2index)

    # Remove duplicates (same user rating same business multiple times)
    df = df.drop_duplicates(subset=["user_id", "business_id"])

    num_users = len(user2index)
    num_businesses = len(business2index)

    print(f"  Users:      {num_users:,}")
    print(f"  Businesses: {num_businesses:,}")
    print(f"  Interactions: {len(df):,}")

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Create CSR matrix (users x businesses)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[4/6] Creating CSR interaction matrix...")

    X = csr_matrix(
        (np.ones(len(df)), (df["user_idx"], df["business_idx"])),
        shape=(num_users, num_businesses),
        dtype=np.float32,
    )

    density = 100.0 * X.nnz / (num_users * num_businesses)
    print(f"  Shape:   {X.shape[0]} × {X.shape[1]}")
    print(f"  NNZ:     {X.nnz:,}")
    print(f"  Density: {density:.4f}%")

    # ──────────────────────────────────────────────────────────────────────────
    # 5. Remove users with fewer than K interactions
    # ──────────────────────────────────────────────────────────────────────────
    print(
        f"\n[5/6] Removing users with <{args.min_user_interactions} interactions..."
    )

    before_filter_users = X.shape[0]
    user_activity = np.array(X.sum(axis=1)).flatten()
    active_users = np.where(user_activity >= args.min_user_interactions)[0]
    X = X[active_users]

    print(f"  Before: {before_filter_users:,} users")
    print(f"  After:  {X.shape[0]:,} users")
    print(f"  Removed: {before_filter_users - X.shape[0]:,} users")

    if X.shape[0] == 0:
        print("  ERROR: No users left after filtering!")
        return

    # ──────────────────────────────────────────────────────────────────────────
    # 6. Split into train/test with disjoint user sets
    # ──────────────────────────────────────────────────────────────────────────
    print(
        f"\n[6/6] Splitting train/test ({100*(1-args.test_size):.0f}/{100*args.test_size:.0f})..."
    )

    all_users = np.arange(X.shape[0])
    train_users, test_users = train_test_split(
        all_users,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    X_train = X[train_users]
    X_test = X[test_users]

    print(f"  Train users: {X_train.shape[0]:,}")
    print(f"  Test users:  {X_test.shape[0]:,}")
    print(f"  Items:       {X_train.shape[1]:,}")
    print(f"  Train NNZ:   {X_train.nnz:,}")
    print(f"  Test NNZ:    {X_test.nnz:,}")

    # ──────────────────────────────────────────────────────────────────────────
    # 7. Save processed data
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[SAVE] Saving processed data...")

    with open(args.output_dir / "processed_train.pkl", "wb") as f:
        pickle.dump(X_train, f)
    print(f"  ✓ {args.output_dir / 'processed_train.pkl'}")

    with open(args.output_dir / "processed_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    print(f"  ✓ {args.output_dir / 'processed_test.pkl'}")

    # Note: using business2index instead of item2index to match Yelp terminology
    with open(args.output_dir / "business2index.pkl", "wb") as f:
        pickle.dump(business2index, f)
    print(f"  ✓ {args.output_dir / 'business2index.pkl'}")

    # Also save as item2index.pkl for compatibility with EasyStudy
    with open(args.output_dir / "item2index.pkl", "wb") as f:
        pickle.dump(business2index, f)
    print(f"  ✓ {args.output_dir / 'item2index.pkl'}")

    # Save user2index mapping as well for reference
    with open(args.output_dir / "user2index.pkl", "wb") as f:
        pickle.dump(user2index, f)
    print(f"  ✓ {args.output_dir / 'user2index.pkl'}")

    # ──────────────────────────────────────────────────────────────────────────
    # 8. Summary
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
    print(f"Train users: {X_train.shape[0]:,}, Test users: {X_test.shape[0]:,}, Items: {X_train.shape[1]:,}")
    print(f"Output directory: {args.output_dir.absolute()}")
    print("\nFiles created:")
    print("  - processed_train.pkl  (train interaction matrix)")
    print("  - processed_test.pkl   (test interaction matrix)")
    print("  - item2index.pkl       (business_id → index mapping)")
    print("  - user2index.pkl       (user_id → index mapping)")
    print("  - business2index.pkl   (same as item2index.pkl)")
    print("\nReady for training with EasyStudy-compatible algorithms!")
    print("=" * 80)


if __name__ == "__main__":
    main()
