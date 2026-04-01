#!/usr/bin/env python3
import duckdb
from pathlib import Path

duckdb_path = "../../Yelp-JSON/yelp.duckdb"
parquet_dir = "../../Yelp-JSON/yelp_parquet"

conn = duckdb.connect(str(duckdb_path))
pattern = str(Path(parquet_dir) / 'business' / '**' / '*.parquet').replace('\\', '/')

# Get schema
result = conn.execute(f'SELECT * FROM read_parquet("{pattern}") LIMIT 1').description
print("Columns in business parquet:")
for col in result:
    print(f"  {col[0]}")

# Check sample states
result = conn.execute(f'SELECT DISTINCT state FROM read_parquet("{pattern}") LIMIT 10').fetchall()
print("\nSample states:</value>")
for row in result:
    print(f"  {row[0]}")

conn.close()
