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

### 2. Load JSON into DuckDB

**Setup (one-time only)**

Load raw Yelp JSON files into DuckDB:

```bash
python -m src.setup_database --json-dir /path/to/yelp_dataset
```

This creates:
- `yelp.duckdb` with three tables: `yelp_business`, `yelp_review`, `yelp_user`
- Enables efficient SQL queries directly on the data
- No file conversion needed - DuckDB reads JSON natively

**Next steps**

After setup, run preprocessing:

```bash
cd <repo_root>
python -m src.preprocess_data --config configs/default.yaml
```

This reads from DuckDB and creates:
    --parquet_dir /path/to/yelp_parquet
```

This produces the same partitioned structure:
```
yelp_parquet/
    business/state=AZ/part-0.parquet
    business/state=CA/part-0.parquet
    ...
    review/year=2005/part-0.parquet
    review/year=2006/part-0.parquet
    ...
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
  parquet_dir: "Yelp-JSON/yelp_parquet"  # Point to where Parquet files are stored
  db_path: "yelp.duckdb"
```

## Preprocessing Pipeline

### Complete workflow: JSON → Parquet → CSR matrices

The **preprocessing notebook** (`notebooks/00_preprocessing.ipynb`) orchestrates the full end-to-end pipeline:

1. **JSON → Parquet conversion** (if needed)
   - Reads raw Yelp JSON files
   - Converts to DataFrames using pandas
   - Saves partitioned Parquet files

2. **Build global ID mappings**
   - Extracts all unique user_ids → creates `user2index.pkl`
   - Extracts all unique business_ids → creates `item2index.pkl`
   - Ensures consistent indices across all experiments

3. **Create CSR matrices**
   - Full matrix: `R_full.npz` (all states, all users)
   - Per-state variants: `R_{STATE}_compact.npz` (only active users/items)
   - Statistics: `state_statistics.csv`

**Outputs saved to:** `data/preprocessed_yelp/`

### Why separate preprocessing?

- **Consistent indices**: Fixed user/item mappings enable reproducible neuron labeling
- **One-time conversion**: Parquet created once, then reused for multiple experiments
- **Explicit filtering**: Business filtering (k-core, state selection) happens during training, not preprocessing
- **Traceability**: `metadata.json` documents preprocessing decisions

### Running the preprocessing pipeline

```python
# Open and run: notebooks/00_preprocessing.ipynb
# The notebook will:
# - Check if Parquet files exist
# - Convert JSON → Parquet if needed (automatic)
# - Build CSR matrices and ID mappings
# - Save all preprocessed data to data/preprocessed_yelp/
```

After preprocessing completes, you can run the **training notebook** (`notebooks/02_training.ipynb`):
- Loads preprocessed CSR matrices
- Applies k-core filtering (k=5)
- Trains models on the filtered data
