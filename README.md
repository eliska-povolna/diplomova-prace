# Explainable and Steerable POI Recommendations

> **Diploma thesis** — Eliška Povolná, 2025/2026

Modern recommender systems can be accurate while still being hard to inspect or influence. This project combines a collaborative-filtering model (**ELSA**) with a **Sparse Autoencoder (SAE)** so user preferences become sparse, readable features that can be interpreted and steered. The data source is the Yelp academic dataset, loaded into **DuckDB** locally or **CloudSQL** when using a cloud backend.

## What data was used in the thesis?

- Final run used in thesis: [outputs/20260427_030923/summary.json](outputs/20260427_030923/summary.json)
- Final filtered dataset stats: 37,323 users, 12,793 items, 525,257 positive interactions, 99.89% sparsity
- Final SAE metrics: NDCG@10=0.0667, Recall@10=0.1087, HR@10=0.2401, Coverage=0.7924, Entropy=0.7907
- Results page implementation: [src/ui/pages/results.py](src/ui/pages/results.py)
- Dataset page implementation: [src/ui/pages/dataset_statistics.py](src/ui/pages/dataset_statistics.py)
- Chart export script for thesis: [scripts/generate_thesis_charts.py](scripts/generate_thesis_charts.py)

### Artifact Snapshot Notes

- Current canonical evaluation artifacts in the latest strict run use K={10,20,50}.
- Steering diagnostics are primarily exposed in the Live Demo through rank deltas, score deltas, and activation shifts.
- Dataset-level and model-level charts used in the thesis are exported by [scripts/generate_thesis_charts.py](scripts/generate_thesis_charts.py).

## What Is This?

This repository contains the full pipeline for an interpretable POI recommender system:

- It loads Yelp JSON into a database.
- It preprocesses user-item interactions into a CSR matrix.
- It trains ELSA to learn dense latent user representations.
- It trains a TopK SAE to turn those dense latents into sparse features.
- It labels those features so they can be inspected and steered in the UI.
- It evaluates ranking quality and model behavior on held-out data.
- It exposes the results in a Streamlit dashboard.

The main idea is simple: instead of keeping recommendations inside an opaque latent vector, the SAE turns them into a set of features that can be named, inspected, and adjusted.

## Contents

- [Quick Start](#quick-start) — Try the demo or run locally
- [System Architecture](#system-architecture) — How data flows through the system
- [Core Algorithms](#core-algorithms) — ELSA and Sparse Autoencoders explained
- [Pipeline: Step-by-Step](#pipeline-step-by-step) — Database setup through evaluation
- [Data: Yelp Dataset](#data-yelp-dataset) — Structure and download
- [Configuration & Customization](#configuration--customization) — Hyperparameters
- [Evaluation Metrics](#evaluation-metrics) — What is measured and how to interpret results
- [Interpretability & Steering](#interpretability--steering) — Feature labeling and interactive control
- [Interactive Dashboard](#interactive-dashboard) — Using the Streamlit UI
- [Notebooks for Exploration](#notebooks-for-exploration) — Analysis and visualization
- [Deployment](#deployment) — Local, Streamlit Cloud, and GCP options
- [References](#references)

## Quick Start

### Option 1: Try the Streamlit Demo Online

Open the live demo (no setup required):  
**[https://explainable-steerable-poi-recommendations.streamlit.app/](https://explainable-steerable-poi-recommendations.streamlit.app/)**

### Option 2: Run the Interactive Dashboard Locally

```bash
# 1. Clone and setup
git clone https://github.com/eliska-povolna/diplomova-prace.git
cd Diplomov-pr-ce
python -m venv .venv
source .venv/bin/activate                          # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt

# 2. Configure secrets (see Configuration & Secrets section below)
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit both files with your credentials

# 3. Download Yelp data (if not present)
# Download from https://www.yelp.com/dataset (JSON format)

# 4. Setup database (one-time)
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset

# 5. Run the UI
streamlit run src/ui/main.py
```

The app will be available at `http://localhost:8501`

### Option 3: Run the Full Training Pipeline

```bash
# Setup database (one-time)
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset

# Preprocess data
python -m src.preprocess_data --config configs/default.yaml

# Train models (ELSA + TopK SAE)
python -m src.train --config configs/default.yaml

# Label neurons for interpretability (auto-detects the latest complete run)
python -m src.label

# Evaluate on test set (auto-detects the latest complete run)
python -m src.evaluate

# View results in interactive dashboard
streamlit run src/ui/main.py
```

Locally, output is saved to `outputs/YYYYMMDD_HHMMSS/` with models, metrics, and interpretations. On Streamlit Cloud, those artifacts are uploaded to GCS and loaded from there instead of relying on a local `outputs/` directory.

**Note:** See [Configuration & Secrets](#configuration--secrets) for required setup.

---

## System Architecture

### High-Level Data Flow

```
Raw Yelp JSON Files
    ↓
[Setup Database] → DuckDB (yelp_business, yelp_review tables) or CloudSQL
    ↓
[Preprocess] → CSR matrix + ID mappings
    ↓
[Train] → ELSA (dense latents) + TopK SAE (sparse features)
    ↓
[Label] → Neuron interpretations (weighted-category, TF-IDF, LLM)
    ↓
[Evaluate] → Ranking metrics, model comparison
    ↓
[Interactive UI] → Metrics, neuron browser, steering interface
```

### Components

**Data Storage:**
- **DuckDB** (local, default): Fast in-process SQL database
  - Tables: `yelp_business`, `yelp_review`
  - File: `yelp.duckdb`
- **CloudSQL** (optional): PostgreSQL backend for cloud deployment
  - Same schema as DuckDB
  - Configured via `CLOUDSQL_*` environment variables
  - Used by the live demo on **[https://explainable-steerable-poi-recommendations.streamlit.app/](https://explainable-steerable-poi-recommendations.streamlit.app/)** (available until June 2026)

**Models:**
- **ELSA** (Scalable Linear Shallow Autoencoder):
  - Input: User interaction history (CSR matrix)
  - Output: dense latent vectors (dimension defined in config)
  - Purpose: Compress user preferences into dense space
  
- **TopK SAE** (Sparse Autoencoder): Sparse feature decomposer
  - Input: latent vectors from ELSA
  - Output: Top-k active neuron codes
  - Purpose: Decompose dense vectors into interpretable, sparse features

**UI & Inference:**
- **Streamlit app** (`src/ui/main.py`): Interactive dashboard
- **Services**: Data loading, model inference, steering, metrics visualization
- **Local + Cloud deployment**: Works on laptop or Streamlit Cloud

### Yelp Data Schema

**yelp_business table:**
```
- business_id (string, primary key)
- name (string): Business name
- state (string): US state abbreviation
- review_count (int): Total reviews
- stars (float): Average rating
- categories (string): Comma-separated categories
- ... (other fields like city, address, coordinates)
```

**yelp_review table:**
```
- review_id (string, primary key)
- user_id (string): Foreign key to user
- business_id (string): Foreign key to business
- stars (int): Rating 1-5
- text (string): Review text
- date (string): Review date (YYYY-MM-DD)
- useful/funny/cool (int): Votes
```

**yelp_user table (optional):**
```
- user_id (string, primary key)
- name (string): User name
- review_count (int): Total reviews by user
- ... (other fields like yelping_since, friends)
```

---

## Core Algorithms

### ELSA (Scalable Linear Shallow Autoencoder)

A linear shallow autoencoder designed for collaborative filtering:

```
Input: x ∈ ℝ^n (user interaction vector, n = # items)
       ↓
Encoder: z = W_e @ x (latent representation, latent_dim-dimensional)
       ↓
Decoder: x̂ = W_d @ z (reconstructed interactions)
       ↓
Loss: MSE(x, x̂)
```

**Implementation:** [src/models/collaborative_filtering.py](src/models/collaborative_filtering.py)

**Configuration:** [configs/default.yaml](configs/default.yaml#L20)
```yaml
elsa:
  latent_dim: 512        # Latent dimension (can be adjusted)
  learning_rate: 0.0003
  batch_size: 1024
  num_epochs: 25
  patience: 10           # Early stopping
```

### TopK SAE (Sparse Autoencoder)

A sparse feature decomposer that identifies the most important directions in the latent space:

```
Input: z ∈ ℝ^latent_dim (dense latent from ELSA)
       ↓
Encoder: h = ReLU(W_encode @ z + b)  (hidden layer, width_ratio * latent_dim)
       ↓
Top-K Selection: h_sparse = TopK(h, k)  (keep only top-k neurons)
       ↓
Decoder: ẑ = W_decode @ h_sparse (reconstruct latent)
       ↓
Loss: MSE(z, ẑ) + L1(h_sparse)
```

**Implementation:** [src/models/sparse_autoencoder.py](src/models/sparse_autoencoder.py)

**Configuration:** [configs/default.yaml](configs/default.yaml#L28)
```yaml
sae:
  width_ratio: 2         # hidden_dim = width_ratio * latent_dim
  k: 32                  # Number of active neurons (sparsity target)
  l1_coef: 0.0003        # L1 regularization strength
  learning_rate: 0.0003
  num_epochs: 50
```

### How They Work Together

1. **User history** → ELSA → e.g. **512-dim latent**
2. **Latent** → SAE → e.g. **32 active neurons** (interpretable features)
3. **Neurons** → labeled semantically (e.g., "likes_coffee_shops", "prefers_outdoor")
4. **User can steer**: Increase/decrease neuron activation → get different recommendations

---

## Pipeline: Step-by-Step

### Stage 0: Setup Database (One-Time)

Load raw Yelp JSON into DuckDB or CloudSQL:

```bash
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset
```

**What it does:**
- Reads `yelp_academic_dataset_*.json` files
- Creates tables: `yelp_business`, `yelp_review`, `yelp_user`
- Creates indices for fast querying

**Output:**
- `yelp.duckdb` (~2-3 GB) in the project root
- Tables are indexed and ready for preprocessing

### Stage 1: Preprocess

Build CSR matrix and ID mappings:

```bash
python -m src.preprocess_data --config configs/default.yaml
```

If you have not run database setup yet, you can combine both steps:

```bash
python -m src.preprocess_data --config configs/default.yaml --setup-database --json-dir ~/Downloads/yelp_dataset
```

**What it does:**
- Loads positive user-item interactions from DuckDB with `stars >= 4.0`
- Applies iterative 5-core filtering so every user and item has at least 5 interactions
- Builds the filtered user→item interaction CSR matrix
- Creates ID mappings: user_id ↔ index, business_id ↔ index
- Splits into train/val/test sets (80/10/10)

**Configuration:** [configs/default.yaml](configs/default.yaml#L6-L14)
```yaml
data:
  db_path: "../../Yelp-JSON/yelp.duckdb"
  state_filter: "PA"           # Set to null for all states
  min_review_count: 20         # Filter users with < 20 reviews
  pos_threshold: 4.0           # Stars >= threshold are positive feedback
  train_test_split: 0.8
  val_split: 0.1
```

**Output:**
- `data/preprocessed_yelp/processed_train.npz` (5-core filtered CSR matrix)
- `data/preprocessed_yelp/user2index.pkl` (filtered user ID mapping)
- `data/preprocessed_yelp/item2index.pkl` (filtered business ID mapping)

### Stage 2: Train

Train ELSA and TopK SAE models:

```bash
python -m src.train --config configs/default.yaml
```

**What it does:**
- Loads preprocessed CSR matrix
- Trains ELSA on train/val sets
- Trains TopK SAE on ELSA latents
- Saves best models and training metrics
- Evaluates on test set (Recall@K, NDCG@K, etc.)

**Configuration:**
```yaml
elsa:
  latent_dim: 512
  learning_rate: 0.0003
  batch_size: 1024
  num_epochs: 25
  device: "cpu"              # or "cuda"

sae:
  width_ratio: 2
  k: 32
  l1_coef: 0.0003
  batch_size: 1024
  num_epochs: 50
```

**Output:**
- Locally, the pipeline writes to `outputs/YYYYMMDD_HHMMSS/`.
- In cloud runs, the same artifacts are uploaded to GCS automatically when `GCS_BUCKET_NAME` is configured.
- `checkpoints/` — Trained models
- `metrics/` — Training logs
- `summary.json` — Training config + the canonical evaluation metrics used by the Results page
- `data/shared_data_manifest.json` — Pointer to the shared preprocessing cache used by the run
- `data/train_user_ids.json`, `data/test_user_ids.json`, `data/val_user_ids.json` — Run-specific split artifacts
- `data/test_holdout_input.npz`, `data/test_holdout_target.npz` — Exact replay artifacts for deterministic post-hoc evaluation
- `precomputed_ui_cache/` — Optional UI cache for word clouds, neuron stats, and test user embeddings

Heavy shared preprocessing artifacts are stored once under `outputs/_shared_preprocessed/<cache_key>/` and referenced from each run, rather than duplicated into every run directory.

**Expected duration:** 5-30 minutes (depends on data size and hardware)

### Stage 3: Label (Optional)

Generate semantic labels for neurons:

```bash
python -m src.label
```

**What it does:**
- Loads the latest complete training run automatically unless you pass `--training-dir`
- If the latest run came from an experiment sweep, the UI and labeling defaults follow the best run from that latest experiment by **NDCG@10**
- Supports label generation with a weighted-category baseline, matrix-based TF-IDF, metadata-only LLM, and review-based LLM methods
- Supports `--method non-llm`, `--skip-coactivation`, and `--coactivation-only`
- Generates neuron labels, neuron embeddings, co-activation data, and interpretability metadata

**Labeling methods:**
- **Weighted-category baseline**: looks at the businesses that activate a neuron most strongly and aggregates their categories weighted by activation strength. Very generic parent categories such as `Restaurants` and `Food` are deprioritized unless they are the only available categories. For backwards compatibility, older artifacts may still call this `tag-based`.
- **Matrix-based**: builds a category-neuron association matrix from the real sparse activations and applies TF-IDF to find the categories that best characterize each neuron while staying grounded in that neuron's top-activating businesses. This is the statistical baseline.
- **LLM-based**: sends the strongest examples and their categories to Google Gemini, which returns a semantic label in plain language. This usually produces the most natural descriptions, but it depends on the API key and is slower than the other methods. The prompt templates live in [src/interpret/prompts.py](src/interpret/prompts.py#L3) for neuron labels and [src/interpret/prompts.py](src/interpret/prompts.py#L17) for superfeature labels.
- **Review-based LLM**: extends the LLM prompt with top useful review snippets from the highest-activating businesses, so the label can capture atmosphere and user-experience details that categories alone miss.
  This method requires the saved review artifact to contain `business_id`, `text`, `useful`, and `stars`.

**Optional LLM API:**
- `GOOGLE_API_KEY` in `.env` or `.streamlit/secrets.toml` (for Gemini-based neuron labeling)

**Output:**
- Locally, the artifacts are written under `outputs/YYYYMMDD_HHMMSS/neuron_interpretations/`.
- In cloud runs, they are uploaded to GCS automatically so Streamlit Cloud can load them.
- `neuron_labels.json` — Selected labels, per-method labels, aliases, and comparison payload
- `labels_weighted-category.pkl` — Dictionary of neuron → activation-weighted category label
- `labels_tag-based.pkl` — Legacy compatibility alias for the weighted-category baseline
- `labels_matrix-based.pkl` — Dictionary of neuron → TF-IDF label
- `labels_llm-based.pkl` — Dictionary of neuron → LLM description
- `labels_llm-review-based.pkl` — Dictionary of neuron → review-enriched LLM description
- `concept_mapping.pkl` — Saved matrix-based concept → neuron mapping for concept steering
- `neuron_embeddings.pt` — Embedding vectors for each neuron
- `neuron_category_metadata.json` — Per-neuron category metadata for the UI
- `neuron_coactivation.json` — Co-activation correlations and labels

### Stage 4: Evaluate

Re-run deterministic ranking and model quality evaluation for a saved run:

```bash
python -m src.evaluate
```

This auto-detects the latest complete run and validates it on the saved test split. The canonical metrics shown in the Results page still come from the training-time `summary.json`; `src.evaluate` is the post-hoc verifier / refresher.

**What it computes:**
- **Ranking metrics** @ K=5, 10, 20:
  - Recall@K, Precision@K, Hit Rate, NDCG, MRR, MAP
- **Model quality**:
  - Coverage (% unique items recommended)
  - Entropy (diversity of recommendations)
  - Sparsity (avg active neurons)
- **Comparison**:
  - ELSA-only vs SAE+ELSA metrics side-by-side
  - % performance difference

**Advanced usage:**
```bash
python -m src.evaluate --checkpoint outputs/YYYYMMDD_HHMMSS --split test
python -m src.evaluate --checkpoint outputs/YYYYMMDD_HHMMSS --split test --sync-summary --sync-experiment-manifest
```

**Output:**
- `outputs/YYYYMMDD_HHMMSS/evaluation_test.json` — Post-hoc evaluation results
- Optionally refreshes `summary.json` and the experiment manifest when `--sync-summary` / `--sync-experiment-manifest` are used

`src.evaluate` prefers the exact saved holdout replay artifacts from the training run. If those artifacts are missing, it falls back to reconstructed holdout evaluation and marks the result as non-canonical.

**Analysis:**
See [Evaluation Analysis Notebook](#notebooks-for-exploration) for detailed visualizations and interpretation.

---

## Data: Yelp Dataset

### What is Yelp Data?

The Yelp Academic Dataset contains:
- **~200K businesses** (restaurants, shops, services in US/Canada/etc)
- **~7M reviews** (user ratings 1-5 and text)
- **~1.5M users** (reviewers)

### Expected Schema After Setup

| Table | Rows | Columns | Key Fields |
|-------|------|---------|-----------|
| `yelp_business` | 150K-200K | 20+ | business_id, name, state, stars, categories |
| `yelp_review` | 5M-7M | 9 | review_id, user_id, business_id, stars, text, date |
| `yelp_user` | 1M-1.5M | 9 | user_id, name, review_count, yelping_since |

### Download Instructions

1. Go to [https://www.yelp.com/dataset](https://www.yelp.com/dataset)
2. Download the **JSON format** (do not use Parquet)
3. Extract to a folder (e.g., `~/Downloads/yelp_dataset/`)
4. You'll have 4 files:
   - `yelp_academic_dataset_business.json`
   - `yelp_academic_dataset_review.json`
   - `yelp_academic_dataset_user.json`
   - `yelp_academic_dataset_checkin.json` (optional)

### Data Filtering

The preprocessing step filters data by:
- **5-core filtering**: Keep only users and items that have at least 5 interactions, applied iteratively until convergence.
- **Rating threshold**: Stars ≥ 4.0 are treated as positive interactions.

This reduces the full dataset to a manageable subset for faster training.



## Configuration & Secrets

Use the quick-start commands above to clone the repo, install dependencies, and initialize the database. This section only covers the secret files and what each part of the system reads.

### Secrets Templates

Copy these templates before running the UI or training scripts locally:

```bash
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

- **`.env`**: For training scripts and utilities
- **`.streamlit/secrets.toml`**: For the Streamlit UI
- Both templates are gitignored so you can keep local credentials out of the repo

### Setup Reminder

- Make sure the Yelp JSON files are available before running `src.setup_database`
- The local database file is `yelp.duckdb`
- If you use CloudSQL or Streamlit Cloud, configure the corresponding secrets before starting the app

---

## Configuration & Customization

### Training Configuration File: [configs/default.yaml](configs/default.yaml)

All training hyperparameters are defined here. Copy to create custom configs:

```bash
cp configs/default.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml
python -m src.train --config configs/my_experiment.yaml
```

**Key Parameters:**

**Data:**
```yaml
data:
  db_path: "../../Yelp-JSON/yelp.duckdb"
  state_filter: "PA"        # null = all states, "CA", "PA", etc.
  min_review_count: 20      # Filter sparse users
  pos_threshold: 4.0        # What counts as positive feedback
  train_test_split: 0.8
  val_split: 0.1
  seed: 42
```

**Model Architecture:**
```yaml
elsa:
  latent_dim: 512           # ELSA latent dimension
  learning_rate: 0.0003
  batch_size: 1024
  num_epochs: 25
  patience: 10              # Early stopping patience

sae:
  width_ratio: 2            # Hidden dim = width_ratio * latent_dim
  k: 32                     # Number of active neurons (sparsity)
  l1_coef: 0.0003          # L1 regularization strength
  learning_rate: 0.0003
  num_epochs: 50
  patience: 10
```

**Training:**
```yaml
output:
  base_dir: "outputs"
  log_level: "INFO"         # DEBUG, INFO, WARNING, ERROR
  save_frequency: 5         # Save checkpoint every N epochs

evaluation:
  k_values: [10, 20, 50]    # Evaluate metrics @ these k values
  device: "cuda"            # or "cpu" (fast on GPU)
```


## Evaluation Metrics

After training completes, metrics are saved to `outputs/YYYYMMDD_HHMMSS/evaluation_test.json`.

### Ranking Metrics (@ K=5, 10, 20)

| Metric | Range | Meaning |
|--------|-------|---------|
| **Recall@K** | 0-1 | % of user's true items that appeared in top-K recommendations |
| **Precision@K** | 0-1 | % of top-K recommendations that were actually relevant |
| **Hit Rate@K** | 0-1 | % of users who got ≥ 1 relevant item in top-K |
| **NDCG@K** | 0-1 | Ranking quality (penalizes irrelevant items in top positions) |
| **MRR@K** | 0-1 | Mean reciprocal rank (1/position of first relevant item) |
| **MAP@K** | 0-1 | Mean average precision (area under precision curve) |

### Model Quality

| Metric | Range | Meaning |
|--------|-------|---------|
| **Coverage** | 0-1 | % of all items that were recommended to at least one user |
| **Entropy** | 0+ | How diverse the recommendations are (higher = more variety) |
| **Sparsity** | 0-K | Average number of active neurons (should be ≈ K) |

### Model Comparison

Results include **ELSA-only** (baseline) vs **SAE+ELSA** (sparse):

```json
{
  "model": "elsa_only",
  "recall@10": 0.28,
  "ndcg@10": 0.32
}
{
  "model": "sae_elsa",
  "recall@10": 0.26,
  "ndcg@10": 0.30,
  "% difference": -7.1%
}
```

**Typical trade-off:** SAE often has slightly lower accuracy (~5-10%) but gains interpretability and steerability.

### For Detailed Analysis

See the **Evaluation Analysis Notebook** (`notebooks/04_evaluation_analysis.ipynb`) for visualizations, metric comparisons, and deeper interpretation.

### Outputs & Artifacts

GCS is the source of truth for cloud runs. Local `outputs/<run_id>/` is the offline mirror and fallback.

- `summary.json` records the run config and train/test metrics.
- `evaluation_test.json` stores ranking and quality metrics for the test split.
- `checkpoints/` stores the best ELSA and SAE weights.
- `neuron_interpretations/` stores labels, co-activation data, and embeddings.
- `precomputed_ui_cache/` stores cached UI assets.

These artifacts are uploaded automatically during the pipeline when `GCS_BUCKET_NAME` is configured.

---

## Interpretability & Steering

### Neuron Labeling Process

Each SAE neuron is labeled to identify what "feature" it represents:

#### Weighted-Category Labeling
- For each neuron, find the businesses with the highest activation scores
- Extract Yelp categories from those top-activating businesses
- Aggregate category evidence weighted by activation strength
- De-prioritize generic parent categories (e.g., Restaurants/Food) unless they are the only available signal
- Best when you want a fast, reproducible label without any API calls

#### Matrix-Based Labeling
- Build a neuron-tag matrix from real sparse activations
- Apply TF-IDF style scoring so labels reflect discriminative concepts, not just frequent tags
- Export concept-neuron mappings used directly for concept steering in Live Demo
- Best when you want a more global, dataset-level view of what a neuron represents

#### LLM-Based Labeling
- Send the strongest activating businesses and their categories to Google Gemini
- Ask the model to summarize the pattern in natural language
- Result: More descriptive labels such as "casual dining with outdoor seating" or "coffee and breakfast spots"
- Best when you want the clearest human-readable interpretation and can afford the extra API step
- Prompt template: [neuron label prompt](src/interpret/prompts.py#L3)
- Superfeature prompt: [superfeature synthesis prompt](src/interpret/prompts.py#L17)

#### Review-Based LLM Labeling
- Extends LLM labeling with top useful review snippets per business
- Uses the same activation-driven top businesses plus review text context
- Produces labels that can capture nuance beyond taxonomy (atmosphere/service cues)
- Requires review artifacts with business_id, text, useful, and stars columns

**See:** `src/interpret/neuron_labeling.py`

### Co-Activation Analysis

Neurons don't work in isolation. Co-activation is computed from saved sparse test activations:

- Load `h_sparse_test.pt` for the run
- Center activations and compute neuron-neuron Pearson correlation matrix
- For each neuron, persist top positive and top negative related neurons (thresholded)
- Save as `neuron_coactivation.json` for direct UI consumption

The UI shows both:
- Frequently co-activated features (positive correlation)
- Rarely co-activated features (negative correlation)

```
Neuron A (coffee)  ──────────┐
                             ├──→ Similar users
Neuron B (outdoor) ──────────┘
```

Available in UI: **Interpretability Tab → Co-Activation Heatmap**

### Top Activations and Wordcloud Methodology

For each neuron, the labeling stage precomputes category metadata:

- `top_items`: top activating businesses with activation value, name, categories, city/state, stars, review_count
- `category_weights`: category → list of activation values contributed by top items

Interpretability UI then renders:

- **Top Activating Businesses**: top businesses sorted by activation
- **Top Activating Categories**: categories ranked by mean activation with frequency and range diagnostics
- **Wordcloud**: category token size is driven by combined frequency and activation strength (`frequency * mean_activation`, scaled)

Wordcloud payloads are precomputed and cached under `precomputed_ui_cache/neuron_wordclouds/` to avoid runtime recomputation.

### Feature Steering in UI

**Live Demo Tab** supports two steering modes that share the same inference pipeline:

1. **Neuron Steering**
  - User directly sets target values for selected neuron activations.
2. **Concept / Superfeature Steering**
  - Query text is matched by cosine similarity against saved concept mappings (matrix-based) or superfeatures/neuron labels (LLM-based).
  - The selected concept is converted to a neuron-weight patch, optionally similarity-scaled.

Both modes are merged into a single steering vector before inference.

Steering inference path:

- Encode user latent to sparse features
- Apply neuron overrides in sparse space
- Decode back to latent
- Interpolate with baseline latent via global alpha
- Re-rank items and compute rank/score deltas

In short, recommendations are not rebuilt by retraining; they are recomputed from saved models with controlled latent-space edits.

**Live workflow:**

1. Select a user (or create synthetic profile)
2. Build steering draft from neuron sliders and/or concept search
3. Apply steering (single action merges all draft sources)
4. See updated top-K recommendations, map updates, and diagnostics
5. Inspect activation shift and rank movement

**Example:**
- Increase "coffee_shops" neuron → get more coffee shop recommendations
- Decrease "expensive" neuron → filter out fine dining

---

## Interactive Dashboard

### Local Access

```bash
streamlit run src/ui/main.py
```

Opens at `http://localhost:8501`

### Cloud Access

Deployed version: [https://explainable-steerable-poi-recommendations.streamlit.app/](https://explainable-steerable-poi-recommendations.streamlit.app/)

### Pages

**🏠 Home**
- Quick stats
- Debug info (data size, model status)
- Cache management

**📊 Results**
- Evaluation metrics table
- Model comparison (ELSA vs SAE+ELSA)
- Performance charts

**🎛️ Live Demo** (main page)
- Select a user or create synthetic profile
- Steering sliders for each neuron
- Concept steering from saved matrix-based concept mappings
- Superfeature steering from saved LLM-grouped feature families
- Steering updates the existing activation chart in place
- Steering updates the recommendation list and map from the same saved recommendation state
- Top-K recommendations list
- POI details (name, category, rating, reviews)
- Photo gallery (if Yelp photos available)

**🔍 Interpretability**
- Neuron browser (sorted by activation frequency)
- Neuron labels (weighted-category baseline, TF-IDF, and LLM-based)
- Superfeature browser for LLM-based labeling runs
- Co-activation heatmap
- Feature visualization

The UI reads saved artifacts from the selected run and does not retrain models or regenerate labels at runtime.

---

## Notebooks for Exploration

Interactive Jupyter notebooks for analysis and visualization:

| Notebook | Purpose |
|----------|---------|
| `00_preprocessing.ipynb` | Build the CSR interaction matrix and ID mappings |
| `01_data_exploration.ipynb` | Explore Yelp data distribution, user/item statistics |
| `02_training.ipynb` | Train ELSA and SAE models interactively |
| `03_neuron_labeling_demo.ipynb` | Browse neuron labels and feature semantics |
| `04_evaluation_analysis.ipynb` | Detailed metrics analysis and comparison |

**To run:**
```bash
jupyter notebook notebooks/
```

**Recommended for:**
- Understanding data before training
- Debugging model behavior
- Writing thesis visualizations
- Exploring edge cases

---

## Deployment

### Local Development

Use the Quick Start steps for local setup, then run the Streamlit app with `streamlit run src/ui/main.py`.

### Streamlit Cloud

1. Push code to GitHub
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
3. Select repository and deploy
4. In app settings, link GitHub Secrets:
   - `CLOUDSQL_INSTANCE`, `CLOUDSQL_USER`, `CLOUDSQL_PASSWORD`
   - `GOOGLE_API_KEY`
   - Streamlit Cloud automatically maps them to `st.secrets`

### Cloud SQL Backend (Optional)

Use PostgreSQL instead of DuckDB for cloud scalability:

1. Create CloudSQL instance in Google Cloud Console
2. Set environment variables:
   ```bash
   export CLOUDSQL_INSTANCE=project:region:instance
   export CLOUDSQL_USER=postgres
   export CLOUDSQL_PASSWORD=...
   export CLOUDSQL_DATABASE=postgres
   ```

3. Setup uses same schema as DuckDB:
   ```bash
   python -m src.setup_database --json-dir ~/Downloads/yelp_dataset --cloud-sql
   ```

### GCS Integration (Optional)

Store models and results on Google Cloud Storage:

```bash
export GCS_BUCKET_NAME=my-bucket
export GOOGLE_CLOUD_PROJECT=my-project
```

Models auto-upload to GCS when configured.

### Current Data Layout

- Local development uses DuckDB as the default database.
- Cloud deployments use the same logical schema through CloudSQL.
- The active state filter, review-count filter, model dimensions, and evaluation settings all come from `configs/default.yaml`.
- Notebook, labeling, and UI outputs are read from the latest completed run in GCS first, with local `outputs/` used only as an offline fallback.

---

## References

License: See [LICENSE](LICENSE).

- Spišák, M. et al. (2024). *From Knots to Knobs: Towards Steerable Collaborative Filtering Using Sparse Autoencoders.*
- Wang, J. et al. (2024). *Understanding Internal Representations of Recommendation Models with Sparse Autoencoders (RecSAE).*
- Vančura, V. et al. (2022). *Scalable Linear Shallow Autoencoder for Collaborative Filtering (ELSA).*
- Bostandjiev, S. et al. (2012). *TasteWeights: a visual interactive hybrid recommender system.*

Cite this thesis: Povolná, E. (2026). *Explainable and Steerable POI Recommendations.* Thesis, Charles University.
