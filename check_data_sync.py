#!/usr/bin/env python3
"""Check data sync between local storage, GCS, and CloudSQL."""

import sys
from pathlib import Path
import duckdb
from google.cloud import storage
import os

try:
    from cloud_sql_python_connector import Connector
    import sqlalchemy
    HAS_SQL_CONNECTOR = True
except ImportError:
    HAS_SQL_CONNECTOR = False

print("=" * 80)
print("DATA SYNC CHECK: Local vs Cloud Storage vs CloudSQL")
print("=" * 80)

# ===========================================================================
# LOCAL DATA COUNTS
# ===========================================================================
print("\n[LOCAL DATA]")
print("-" * 80)

# Count local JPG files
photos_dir = Path("yelp_photos/photos")
local_jpg_count = len(list(photos_dir.glob("*.jpg")))
print(f"✓ Local JPG files: {local_jpg_count:,}")

# Count rows in local photos.json
if Path("yelp_photos/photos.json").exists():
    with open("yelp_photos/photos.json", "r") as f:
        photos_json_lines = len(f.readlines())
    print(f"✓ Local photos.json lines: {photos_json_lines:,}")
else:
    photos_json_lines = 0
    print(f"✗ Local photos.json NOT FOUND")

# Count rows in local parquet files using DuckDB
parquet_base = Path("../../Yelp-JSON/yelp_parquet").resolve()
parquet_counts = {}

if parquet_base.exists():
    try:
        con = duckdb.connect()
        
        # Count review rows (recursive)
        review_files = list(parquet_base.glob("review/**/*.parquet"))
        if review_files:
            # Use glob pattern for DuckDB
            review_pattern = str((parquet_base / "review" / "**" / "*.parquet").resolve()).replace("\\", "/")
            result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{review_pattern}', union_by_name=true)").fetchall()
            parquet_counts['review'] = result[0][0]
        
        # Count business rows (recursive)
        business_files = list(parquet_base.glob("business/**/*.parquet"))
        if business_files:
            business_pattern = str((parquet_base / "business" / "**" / "*.parquet").resolve()).replace("\\", "/")
            result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{business_pattern}', union_by_name=true)").fetchall()
            parquet_counts['business'] = result[0][0]
        
        # Count user rows (recursive)
        user_files = list(parquet_base.glob("user/**/*.parquet"))
        if user_files:
            user_pattern = str((parquet_base / "user" / "**" / "*.parquet").resolve()).replace("\\", "/")
            result = con.execute(f"SELECT COUNT(*) FROM read_parquet('{user_pattern}', union_by_name=true)").fetchall()
            parquet_counts['user'] = result[0][0]
        
        con.close()
        
        for table, count in sorted(parquet_counts.items()):
            print(f"✓ Parquet {table} rows: {count:,}")
            
    except Exception as e:
        print(f"✗ Error reading parquet files: {e}")
else:
    print(f"✗ Parquet directory NOT FOUND: {parquet_base}")

# ===========================================================================
# GCS BUCKET COUNTS
# ===========================================================================
print("\n[CLOUD STORAGE (GCS)]")
print("-" * 80)

try:
    client = storage.Client()
    bucket = client.bucket("diplomova-prace")
    blobs = list(bucket.list_blobs(prefix="photos/"))

    jpg_count = sum(1 for blob in blobs if blob.name.endswith(".jpg"))
    json_count = sum(1 for blob in blobs if blob.name.endswith(".json"))
    total_size_gb = sum(blob.size for blob in blobs) / (1024**3)

    print(f"✓ GCS JPG files: {jpg_count:,}")
    print(f"✓ GCS JSON metadata files: {json_count:,}")
    print(f"✓ GCS total size: {total_size_gb:.2f} GB")
    print(f"✓ GCS total objects: {len(blobs):,}")
except Exception as e:
    print(f"✗ Error connecting to GCS: {e}")
    jpg_count = None

# ===========================================================================
# CLOUDSQL DATABASE COUNTS
# ===========================================================================
print("\n[CLOUD SQL DATABASE]")
print("-" * 80)

if not HAS_SQL_CONNECTOR:
    print("✗ cloud-sql-python-connector not installed")
    print("  Run: pip install cloud-sql-python-connector")
else:
    try:
        connector = Connector()

        def getconn():
            return connector.connect(
                os.environ.get("CLOUDSQL_INSTANCE", "project-875345c9-e860-4df7-b06:us-central1:yelp-sae-db"),
                "pg8000",
                user=os.environ.get("CLOUDSQL_USER", "postgres"),
                password=os.environ.get("CLOUDSQL_PASSWORD"),
                db=os.environ.get("CLOUDSQL_DATABASE", "postgres"),
            )

        engine = sqlalchemy.create_engine(
            "postgresql+pg8000://",
            creator=getconn,
        )

        with engine.connect() as conn:
            # Get table list
            result = conn.execute(
                sqlalchemy.text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"
                )
            )
            tables = [row[0] for row in result]
            
            if not tables:
                print("✗ No tables found in CloudSQL")
            else:
                print(f"✓ Tables in CloudSQL: {', '.join(tables)}")
                print()

                # Count rows in each table
                cloudsql_counts = {}
                for table in tables:
                    try:
                        result = conn.execute(sqlalchemy.text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        cloudsql_counts[table] = count
                        print(f"  - {table}: {count:,} rows")
                    except Exception as e:
                        print(f"  - {table}: ERROR - {e}")

        connector.close()

    except Exception as e:
        print(f"✗ Error connecting to CloudSQL: {e}")

# Initialize cloudsql_counts if it doesn't exist
if 'cloudsql_counts' not in locals():
    cloudsql_counts = {}

# ===========================================================================
# COMPARISON SUMMARY
# ===========================================================================
print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

print(f"\nPhotos:")
print(f"  Local JPGs:         {local_jpg_count:,}")
print(f"  Local photos.json:  {photos_json_lines:,} lines")
if jpg_count is not None:
    print(f"  GCS JPGs:           {jpg_count:,}")
    if jpg_count == local_jpg_count:
        print(f"  ✓ MATCH: All {local_jpg_count:,} photos uploaded to GCS")
    else:
        missing = local_jpg_count - jpg_count
        print(f"  ✗ MISMATCH: {missing:,} photos missing in GCS (~{missing/local_jpg_count*100:.1f}%)")
else:
    print(f"  GCS JPGs:           UNABLE TO CHECK")

print(f"\nDatabase Tables (CloudSQL):")
if cloudsql_counts:
    for table, count in sorted(cloudsql_counts.items()):
        print(f"  - {table}: {count:,} rows")
else:
    print("  ✗ Could not retrieve CloudSQL table counts")

print(f"\nParquet Data (Local - Row Counts):")
if parquet_counts:
    for table, count in sorted(parquet_counts.items()):
        print(f"  - {table}: {count:,} rows")
else:
    print("  ✗ Could not count parquet rows")
