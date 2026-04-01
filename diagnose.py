#!/usr/bin/env python3
"""Diagnose the exact problem with data loading."""
import sys
import traceback

output = []

try:
    output.append("=" * 60)
    output.append("DIAGNOSTIC: Data Service State Filter")
    output.append("=" * 60)
    
    # Step 1: Check paths exist
    from pathlib import Path
    db = "../../Yelp-JSON/yelp.duckdb"
    pq_dir = "../../Yelp-JSON/yelp_parquet"
    
    output.append(f"\n1. Checking paths:")
    output.append(f"   CWD: {Path.cwd()}")
    output.append(f"   DB exists: {Path(db).exists()}")
    output.append(f"   Parquet dir exists: {Path(pq_dir).exists()}")
    
    if not Path(db).exists() or not Path(pq_dir).exists():
        output.append("   ❌ Paths don't exist!")
        with open("diagnose_output.txt", "w") as f:
            f.write("\n".join(output))
        sys.exit(1)
    
    # Step 2: Connect to DuckDB
    output.append(f"\n2. Connecting to DuckDB...")
    import duckdb
    conn = duckdb.connect(str(db))
    output.append(f"   ✅ Connected")
    
    # Step 3: Test glob pattern
    output.append(f"\n3. Testing glob pattern:")
    pattern = str(Path(pq_dir) / 'business' / '**' / '*.parquet').replace('\\', '/')
    output.append(f"   Pattern: {pattern}")
    
    # Try to read files
    try:
        result = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{pattern}")').fetchall()
        all_count = result[0][0]
        output.append(f"   ✅ All POIs: {all_count}")
    except Exception as e:
        output.append(f"   ❌ Error: {str(e)}")
        with open("diagnose_output.txt", "w") as f:
            f.write("\n".join(output))
        sys.exit(1)
    
    # Step 4: Test state filter
    output.append(f"\n4. Testing state filter (WHERE state = 'CA'):")
    try:
        result = conn.execute(f'SELECT COUNT(*) FROM read_parquet("{pattern}") WHERE state = "CA"').fetchall()
        ca_count = result[0][0]
        output.append(f"   ✅ CA POIs: {ca_count}")
    except Exception as e:
        output.append(f"   ❌ Error: {str(e)}")
        with open("diagnose_output.txt", "w") as f:
            f.write("\n".join(output))
        sys.exit(1)
    
    # Step 5: Summary
    output.append(f"\n5. Summary:")
    if ca_count == 0:
        output.append(f"   ❌ PROBLEM: CA filter returned 0 POIs!")
        output.append(f"   This means the WHERE state = 'CA' clause isn't working")
        
        # Check what states exist
        output.append(f"\n6. Checking available states:")
        result = conn.execute(f'SELECT DISTINCT state FROM read_parquet("{pattern}") LIMIT 10').fetchall()
        output.append(f"   Sample states: {[r[0] for r in result]}")
    elif ca_count < 1000:
        output.append(f"   ⚠️  WARNING: CA returned {ca_count} POIs (expected ~2212)")
    else:
        output.append(f"   ✅ SUCCESS: CA filter working ({ca_count} POIs)")
    
    conn.close()
    output.append("\n" + "=" * 60)
    
except Exception as e:
    output.append(f"\n❌ FATAL ERROR:")
    output.append(f"   {str(e)}")
    output.append(traceback.format_exc())

# Write all output to file
with open("diagnose_output.txt", "w") as f:
    f.write("\n".join(output))

# Also print first 50 lines
for line in output[:50]:
    print(line)

