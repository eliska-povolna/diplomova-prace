#!/usr/bin/env python3
"""Quick check - count POIs with and without state filter."""
import duckdb
from pathlib import Path

duckdb_path = "../../Yelp-JSON/yelp.duckdb"
parquet_dir = "../../Yelp-JSON/yelp_parquet"

with open("count_results.txt", "w") as f:
    conn = duckdb.connect(str(duckdb_path))
    pattern = str(Path(parquet_dir) / 'business' / '**' / '*.parquet').replace('\\', '/')

    result_all = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{pattern}")').fetchall()
    all_count = result_all[0][0]
    
    result_ca = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{pattern}") WHERE state = "CA"').fetchall()
    ca_count = result_ca[0][0]

    f.write(f"All POIs: {all_count}\n")
    f.write(f"CA POIs: {ca_count}\n")
    
    conn.close()

print("✅ Done - check count_results.txt")
