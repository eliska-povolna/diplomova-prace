#!/usr/bin/env python3
"""Check if 'state' column exists and has 'CA' values."""
import duckdb
from pathlib import Path

duckdb_path = "../../Yelp-JSON/yelp.duckdb"
parquet_dir = "../../Yelp-JSON/yelp_parquet"

conn = duckdb.connect(str(duckdb_path))
pattern = str(Path(parquet_dir) / 'business' / '**' / '*.parquet').replace('\\', '/')

# Check columns
print("=== TABLE SCHEMA ===")
info = conn.execute(f'SELECT * FROM read_parquet("{pattern}") LIMIT 1').description
for col in info:
    print(f"  {col[0]}")

# Check for 'state' column and CA values
try:
    result = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{pattern}") WHERE state = "CA"').fetchall()
    print(f"\n✅ 'state' column exists")
    print(f"   CA count: {result[0][0]}")
except Exception as e:
    print(f"\n❌ 'state' column issue: {e}")

# Show actual state values
print("\n=== ACTUAL STATE VALUES ===")
result = conn.execute(f'SELECT DISTINCT state FROM read_parquet("{pattern}") ORDER BY state').fetchall()
for row in result[:5]:
    print(f"  {row[0]}")

conn.close()
