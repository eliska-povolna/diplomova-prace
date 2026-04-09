# Data directory

This directory holds raw and processed Yelp dataset files.
**None of the data files are tracked in git** (see `.gitignore`).

## Download and convert instructions

### 1. Download the Yelp dataset

1. Go to <https://www.yelp.com/dataset> and request the dataset.
2. Download the archive and extract it.
3. You should have these JSON files:

```
yelp_dataset/
    yelp_academic_dataset_business.json      (~120 MB)
    yelp_academic_dataset_review.json        (~6 GB)
    yelp_academic_dataset_user.json          (~3 GB)
    yelp_academic_dataset_checkin.json       (optional)
    yelp_academic_dataset_tip.json           (optional)
```

### 2. Convert JSON to Parquet

Use the dedicated conversion script to convert Yelp JSON files to Parquet format (for faster loading and reduced memory usage):

```bash
cd <repo_root>
python src/data/convert_json_to_parquet.py \
    --json_dir /path/to/yelp_dataset \
    --parquet_dir /path/to/yelp_parquet
```

**Options:**
- `--max_rows N`: Convert only the first N rows per file (useful for testing)
- `--chunk_size N`: Adjust memory usage (default: 50,000)
- `--tables business review user`: Convert specific tables (default: all)

**Example** (testing with first 100k rows):
```bash
python src/data/convert_json_to_parquet.py \
    --json_dir /path/to/yelp_dataset \
    --parquet_dir /path/to/yelp_parquet \
    --max_rows 100000
```

This produces:
```
yelp_parquet/
    business.parquet
    review.parquet
    user.parquet
```

## Directory layout

| Directory | Contents |
|---|---|
| `data/raw/` | Original Yelp JSON files (if downloaded) |
| `data/processed/` | Cleaned, filtered interaction matrices |
| `data/interim/` | Intermediate artefacts from preprocessing steps |

## Using the converted Parquet files

After conversion, update your config to point to the Parquet directory:

**In `configs/default.yaml`:**
```yaml
data:
  parquet_dir: "/path/to/yelp_parquet"  # Point to where convert_json_to_parquet.py output files are
  db_path: "yelp.duckdb"
```

Then training will automatically load from Parquet files instead of JSON.
