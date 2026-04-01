#!/usr/bin/env python3
"""Simple test to verify data loading with state filter."""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ui.services.data_service import DataService

# Set up paths - from the repo root
duckdb_path = Path("../../Yelp-JSON/yelp.duckdb")
parquet_dir = Path("../../Yelp-JSON/yelp_parquet")

print(f"Working directory: {Path.cwd()}")
print(f"DuckDB path (relative): {duckdb_path}")
print(f"DuckDB path (absolute): {duckdb_path.resolve()}")
print(f"Parquet dir (absolute): {parquet_dir.resolve()}")

# Test config
config = {"state_filter": "CA"}

try:
    print("\nLoading DataService with CA filter...")
    data = DataService(duckdb_path, parquet_dir, config)
    print(f"✅ Loaded successfully!")
    print(f"   POIs: {data.num_pois}")
    print(f"   State filter: {data.state_filter}")

    if data.num_pois > 0:
        print("\n✅ SUCCESS: State filter is working!")
        print(f"   Expected ~2212 POIs (CA), got {data.num_pois}")
    else:
        print("\n❌ ERROR: No POIs loaded!")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
