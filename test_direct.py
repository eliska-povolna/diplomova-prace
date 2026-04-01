#!/usr/bin/env python3
"""Direct test of DataService with state filter."""
import sys
import logging
# Suppress logging to see just our output
logging.getLogger().setLevel(logging.CRITICAL)

from pathlib import Path
from src.ui.services.data_service import DataService

try:
    # Test with state_filter
    config_ca = {"state_filter": "CA"}
    data_ca = DataService(
        duckdb_path=Path("../../Yelp-JSON/yelp.duckdb"),
        parquet_dir=Path("../../Yelp-JSON/yelp_parquet"),
        config=config_ca
    )
    print(f"CA filter: {data_ca.num_pois} POIs, state_filter={data_ca.state_filter}")
except Exception as e:
    print(f"ERROR with CA filter: {e}")
    sys.exit(1)

try:
    # Test without state_filter
    config_none = {}
    data_none = DataService(
        duckdb_path=Path("../../Yelp-JSON/yelp.duckdb"),
        parquet_dir=Path("../../Yelp-JSON/yelp_parquet"),
        config=config_none
    )
    print(f"No filter: {data_none.num_pois} POIs, state_filter={data_none.state_filter}")
except Exception as e:
    print(f"ERROR without filter: {e}")
    sys.exit(1)

print("✅ Success!")

