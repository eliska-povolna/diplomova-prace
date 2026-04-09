from pathlib import Path
import duckdb

parquet_dir = Path('../../Yelp-JSON/yelp_parquet')
pattern = str(parquet_dir / 'business' / 'state=*' / '*.parquet').replace('\\', '/')

print(f"Pattern: {pattern}\n")

conn = duckdb.connect(':memory:')
result = conn.execute(f"SELECT * FROM read_parquet('{pattern}') LIMIT 1").description
print("Columns available:")
for col in result:
    print(f"  - {col[0]}")

# Try to count with state filter
try:
    result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}') WHERE state = 'CA'").fetchall()
    print(f"\nWHERE state='CA': {result[0][0]} rows")
except Exception as e:
    print(f"\nError with WHERE state='CA': {e}")

# Try without WHERE to see if it loads
try:
    result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{pattern}')").fetchall()
    print(f"Total rows (no filter): {result[0][0]} rows")
except Exception as e:
    print(f"Error counting: {e}")

conn.close()
