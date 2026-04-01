#!/usr/bin/env python3
"""Debug state filter in DuckDB queries."""
import duckdb
from pathlib import Path
import os

# Go to project root
os.chdir(Path(__file__).parent)

# Paths from project root - Yelp-JSON is a sibling folder outside Github/
duckdb_path = '../../Yelp-JSON/yelp.duckdb'
parquet_dir = '../../Yelp-JSON/yelp_parquet'

print(f"CWD: {os.getcwd()}")
print(f"DuckDB path (relative): {duckdb_path}")
print(f"DuckDB path (absolute): {os.path.abspath(duckdb_path)}")
print(f"Parquet dir (absolute): {os.path.abspath(parquet_dir)}")

if not os.path.exists(duckdb_path):
    print(f"❌ DuckDB file not found: {os.path.abspath(duckdb_path)}")
    exit(1)

if not os.path.exists(parquet_dir):
    print(f"❌ Parquet dir not found: {os.path.abspath(parquet_dir)}")
    exit(1)

conn = duckdb.connect(str(duckdb_path))
# Use forward slashes for glob pattern - DuckDB expects posix-style paths
parquet_pattern = str(Path(parquet_dir) / 'business' / '**' / '*.parquet').replace('\\', '/')

print(f"\nParquet pattern: {parquet_pattern}\n")

# Test 1: Count all businesses
print('Test 1: All businesses')
result = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{parquet_pattern}")').fetchall()
print(f'  Total: {result[0][0]}')

# Test 2: Count CA businesses
print('\nTest 2: CA businesses')
result = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{parquet_pattern}") WHERE state = "CA"').fetchall()
print(f'  CA: {result[0][0]}')

# Test 3: Check actual state values
print('\nTest 3: State distribution')
result = conn.execute(f'SELECT state, COUNT(*) as cnt FROM read_parquet("{parquet_pattern}") GROUP BY state ORDER BY cnt DESC').fetchall()
for row in result[:10]:
    print(f'  {row[0]}: {row[1]}')

conn.close()
print("\n✅ Done")


# Test 1: Count all businesses
print('Test 1: All businesses')
result = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{parquet_pattern}")').fetchall()
print(f'  Total: {result[0][0]}')

# Test 2: Count CA businesses
print('\nTest 2: CA businesses')
result = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{parquet_pattern}") WHERE state = "CA"').fetchall()
print(f'  CA: {result[0][0]}')

# Test 3: Check actual state values
print('\nTest 3: State distribution')
result = conn.execute(f'SELECT state, COUNT(*) as cnt FROM read_parquet("{parquet_pattern}") GROUP BY state ORDER BY cnt DESC').fetchall()
for row in result[:10]:
    print(f'  {row[0]}: {row[1]}')

conn.close()
print("\n✅ Done")
