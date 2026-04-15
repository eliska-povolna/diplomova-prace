#!/usr/bin/env python3
"""Convert Yelp JSON files to Parquet format.

This script streams large Yelp JSON Lines files into Parquet format,
breaking them into chunks to avoid memory overflow.

Usage:
    python src/data/convert_json_to_parquet.py \\
        --json_dir /path/to/yelp_dataset \\
        --parquet_dir /path/to/yelp_parquet \\
        --max_rows 500000 (optional, default: None = full dataset)

Expected input structure:
    yelp_dataset/
        yelp_academic_dataset_business.json
        yelp_academic_dataset_review.json
        yelp_academic_dataset_user.json
        yelp_academic_dataset_tip.json
        yelp_academic_dataset_checkin.json

Output structure:
    yelp_parquet/
        business.parquet
        review.parquet
        user.parquet
        (other tables as desired)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterator, Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Iterate over JSONL file."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def convert_jsonl_to_parquet(
    json_path: Path,
    output_path: Path,
    max_rows: Optional[int] = None,
    chunk_size: int = 50_000,
    dtype_overrides: Optional[Dict[str, str]] = None,
) -> Path:
    """Stream JSONL to Parquet in chunks to manage memory.

    Parameters
    ----------
    json_path : Path
        Input JSONL file path.
    output_path : Path
        Output Parquet file path.
    max_rows : int, optional
        Maximum number of rows to convert. If None, convert all.
    chunk_size : int, default=50_000
        Number of rows to accumulate before writing a chunk.
    dtype_overrides : dict, optional
        Column name → dtype mappings to apply after converting.

    Returns
    -------
    Path
        Output Parquet file path.
    """
    if not json_path.exists():
        logger.warning(f"Skipping {json_path}: file not found")
        return output_path

    logger.info(f"Converting {json_path.name} → {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    total_rows = 0

    for i, rec in enumerate(iter_jsonl(json_path)):
        rows.append(rec)

        # Write chunk
        if len(rows) >= chunk_size:
            df = pd.DataFrame(rows)
            if dtype_overrides:
                for col, dt in dtype_overrides.items():
                    if col in df.columns:
                        try:
                            df[col] = df[col].astype(dt)
                        except (ValueError, TypeError):
                            logger.debug(f"Could not cast {col} to {dt}, skipping")

            # Write each chunk to a separate part file to avoid append issues
            if not output_path.exists():
                df.to_parquet(
                    output_path,
                    engine="pyarrow",
                    compression="zstd",
                    index=False,
                )
            else:
                # For incremental writes, append to existing file
                import pyarrow as pa
                from pyarrow import parquet as pq

                # Read existing file and concatenate with new data
                existing_table = pq.read_table(str(output_path))
                new_table = pa.Table.from_pandas(df)
                combined_table = pa.concat_tables([existing_table, new_table])
                pq.write_table(combined_table, str(output_path), compression="zstd")
            total_rows += len(rows)
            rows = []
            logger.info(f"  ... wrote {total_rows} rows so far")

        # Check max_rows limit
        if max_rows and i + 1 >= max_rows:
            logger.info(f"Reached max_rows={max_rows}, stopping")
            break

    # Write remaining rows using same PyArrow strategy
    if rows:
        df = pd.DataFrame(rows)
        if dtype_overrides:
            for col, dt in dtype_overrides.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dt)
                    except (ValueError, TypeError):
                        logger.debug(f"Could not cast {col} to {dt}, skipping")

        import pyarrow as pa
        from pyarrow import parquet as pq

        new_table = pa.Table.from_pandas(df)
        if output_path.exists():
            existing_table = pq.read_table(str(output_path))
            combined_table = pa.concat_tables([existing_table, new_table])
            pq.write_table(combined_table, str(output_path), compression="zstd")
        else:
            pq.write_table(new_table, str(output_path), compression="zstd")
        total_rows += len(rows)

    logger.info(f"✓ Converted {total_rows} rows → {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Yelp JSON dataset to Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        required=True,
        help="Directory containing Yelp JSON files (yelp_academic_dataset_*.json)",
    )
    parser.add_argument(
        "--parquet_dir",
        type=Path,
        required=True,
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Max rows per file (None = all)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=50_000,
        help="Rows per write chunk (default: 50k)",
    )
    parser.add_argument(
        "--tables",
        type=str,
        nargs="+",
        default=["business", "review", "user"],
        help="Which tables to convert (default: business review user)",
    )

    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    parquet_dir = Path(args.parquet_dir)
    parquet_dir.mkdir(parents=True, exist_ok=True)

    table_map = {
        "business": json_dir / "yelp_academic_dataset_business.json",
        "review": json_dir / "yelp_academic_dataset_review.json",
        "user": json_dir / "yelp_academic_dataset_user.json",
        "tip": json_dir / "yelp_academic_dataset_tip.json",
        "checkin": json_dir / "yelp_academic_dataset_checkin.json",
    }

    logger.info("=" * 70)
    logger.info("CONVERTING YELP JSON → PARQUET")
    logger.info("=" * 70)

    for table_name in args.tables:
        if table_name not in table_map:
            logger.warning(f"Unknown table: {table_name}")
            continue

        json_file = table_map[table_name]
        parquet_file = parquet_dir / f"{table_name}.parquet"
        convert_jsonl_to_parquet(
            json_file,
            parquet_file,
            max_rows=args.max_rows,
            chunk_size=args.chunk_size,
        )

    logger.info("=" * 70)
    logger.info(f"✓ All conversions complete. Output: {parquet_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
