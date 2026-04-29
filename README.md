# Explainable and Steerable POI Recommendations

> **Diploma thesis** - Eliška Povolná, 2025

Modern recommender systems can be accurate while still being hard to inspect or influence. This project combines a collaborative-filtering model (**ELSA**) with a **Sparse Autoencoder (SAE)** so user preferences become sparse, readable features that can be interpreted and steered. The data source is the Yelp academic dataset, loaded into **DuckDB** locally or **Cloud SQL** when using a cloud backend.

## What data was used in the thesis?

- Final run used in the thesis: [outputs/20260427_030923/summary.json](outputs/20260427_030923/summary.json)
- Final filtered dataset stats: 37,323 users, 12,793 items, 525,257 positive interactions, 99.89% sparsity
- Final SAE metrics: NDCG@10=0.0667, Recall@10=0.1087, HR@10=0.2401, Coverage=0.7924, Entropy=0.7907
- Results page implementation: [src/ui/pages/results.py](src/ui/pages/results.py)
- Dataset page implementation: [src/ui/pages/dataset_statistics.py](src/ui/pages/dataset_statistics.py)
- Chart export script for thesis: [scripts/generate_thesis_charts.py](scripts/generate_thesis_charts.py)

### Artifact Snapshot Notes

- Current canonical evaluation artifacts in the latest strict run use K={10,20,50}.
- Steering diagnostics are primarily exposed in the Live Demo through rank deltas, score deltas, activation shifts, CPR, and NDCG@K.
- Dataset-level and model-level charts used in the thesis are exported by [scripts/generate_thesis_charts.py](scripts/generate_thesis_charts.py).

### Steering Evaluation

- Steering changes are appended to [outputs/steering_eval.csv](outputs/steering_eval.csv) from the Live Demo.
- Charts are generated with [scripts/plot_steering_eval.py](scripts/plot_steering_eval.py) and written to [img/generated/](img/generated/).
- The Results page includes a dedicated Steering Eval tab with log filters, recent rows, and generated charts.
- Plot generation keeps only the latest 500 steering rows by default so the UI stays responsive.

## What Is This?

This repository contains the full pipeline for an interpretable POI recommender system:

- It loads Yelp JSON files into a database.
- It preprocesses user-item interactions into a CSR matrix.
- It trains ELSA to learn dense latent user representations.
- It trains a Top-K SAE to turn those dense latents into sparse features.
- It labels those features so they can be inspected and steered in the UI.
- It evaluates ranking quality and model behavior on held-out data.
- It exposes the results in a Streamlit dashboard.

The main idea is simple: instead of keeping recommendations inside an opaque latent vector, the SAE turns them into a set of features that can be named, inspected, and adjusted.

## Contents

- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Core Algorithms](#core-algorithms)
- [Pipeline: Step-by-Step](#pipeline-step-by-step)
- [Data: Yelp Dataset](#data-yelp-dataset)
- [Configuration & Secrets](#configuration--secrets)
- [Configuration & Customization](#configuration--customization)
- [Evaluation Metrics](#evaluation-metrics)
- [Interpretability & Steering](#interpretability--steering)
- [Interactive Dashboard](#interactive-dashboard)
- [Notebooks for Exploration](#notebooks-for-exploration)
- [Deployment](#deployment)
- [References](#references)

## Quick Start

### Option 1: Try the Streamlit Demo Online

Open the live demo, no setup required:

**[https://explainable-steerable-poi-recommendations.streamlit.app/](https://explainable-steerable-poi-recommendations.streamlit.app/)**

### Option 2: Run the Interactive Dashboard Locally

```bash
# 1. Clone and setup
git clone https://github.com/eliska-povolna/diplomova-prace.git
cd diplomova-prace
python -m venv .venv
source .venv/bin/activate                          # Windows: .venv\Scripts\activate
pip install -r src/requirements.txt

# 2. Configure secrets
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit both files with your credentials

# 3. Download Yelp data if not present
# Download from https://www.yelp.com/dataset in JSON format

# 4. Setup database once
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset

# 5. Run the UI
streamlit run src/ui/main.py
````

The app will be available at `http://localhost:8501`.

### Option 3: Run the Full Training Pipeline

```bash
# Setup database once
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset

# Preprocess data
python -m src.preprocess_data --config configs/default.yaml

# Train models
python -m src.train --config configs/default.yaml

# Label neurons for interpretability
python -m src.label

# Evaluate on test set
python -m src.evaluate

# View results in interactive dashboard
streamlit run src/ui/main.py
```

Locally, output is saved to `outputs/YYYYMMDD_HHMMSS/` with models, metrics, and interpretation artifacts. On Streamlit Cloud, selected artifacts are uploaded to GCS and loaded from there instead of relying on a local `outputs/` directory.

## System Architecture

### High-Level Data Flow

```text
Raw Yelp JSON Files
    |
    v
[Setup Database] -> DuckDB or Cloud SQL
    |
    v
[Preprocess] -> CSR matrix and ID mappings
    |
    v
[Train] -> ELSA dense latents + Top-K SAE sparse features
    |
    v
[Label] -> Neuron interpretations
    |
    v
[Evaluate] -> Ranking metrics and model comparison
    |
    v
[Interactive UI] -> Metrics, neuron browser, steering interface
```

### Components

**Data Storage:**

* **DuckDB** for local use:

  * Tables: `yelp_business`, `yelp_review`, optionally `yelp_user`
  * File: `yelp.duckdb`
* **Cloud SQL** for cloud deployment:

  * PostgreSQL backend with the same logical schema
  * Configured via `CLOUDSQL_*` environment variables
  * Used by the live demo while the cloud resources are available

**Models:**

* **ELSA** (Scalable Linear Shallow Autoencoder):

  * Input: user interaction history as a CSR matrix
  * Output: dense latent user vectors
  * Purpose: compress user preferences into dense space

* **Top-K SAE** (Sparse Autoencoder):

  * Input: latent vectors from ELSA
  * Output: sparse feature activations
  * Purpose: decompose dense vectors into interpretable sparse features

**UI & Inference:**

* **Streamlit app**: `src/ui/main.py`
* Services for data loading, model inference, steering, metrics, and visualization
* Local and cloud deployment support

### Yelp Data Schema

**yelp_business table:**

```text
business_id (string, primary key)
name (string)
state (string)
city (string)
address (string)
latitude, longitude (float)
review_count (int)
stars (float)
categories (string)
attributes (object/string)
hours (object/string)
```

**yelp_review table:**

```text
review_id (string, primary key)
user_id (string)
business_id (string)
stars (int)
text (string)
date (string)
useful, funny, cool (int)
```

**yelp_user table, optional:**

```text
user_id (string, primary key)
name (string)
review_count (int)
average_stars (float)
yelping_since (string)
friends (string/list)
```

## Core Algorithms

### ELSA

ELSA is a linear shallow autoencoder used for collaborative filtering.

```text
Input: x, user interaction vector
Encoder: z = W_e x
Decoder: x_hat = W_d z
Loss: reconstruction error
```

**Implementation:** [src/models/collaborative_filtering.py](src/models/collaborative_filtering.py)

Example configuration:

```yaml
elsa:
  latent_dim: 512
  learning_rate: 0.0003
  batch_size: 1024
  num_epochs: 25
  patience: 10
```

### Top-K SAE

The Top-K SAE decomposes ELSA latent vectors into sparse activations.

```text
Input: z, dense latent vector from ELSA
Encoder: h = ReLU(W_encode z + b)
Top-K: keep only the k highest positive activations
Decoder: z_hat = W_decode h_sparse
Loss: reconstruction error + L1 regularization
```

**Implementation:** [src/models/sparse_autoencoder.py](src/models/sparse_autoencoder.py)

Example configuration:

```yaml
sae:
  width_ratio: 2
  k: 32
  l1_coef: 0.0003
  learning_rate: 0.0003
  num_epochs: 50
```

### How They Work Together

1. User history is encoded by ELSA into a dense latent vector.
2. The SAE transforms this dense vector into sparse feature activations.
3. Sparse features are labeled semantically.
4. The user can steer recommendations by increasing or decreasing selected features.
5. The modified sparse profile is decoded back and used to re-rank items.

## Pipeline: Step-by-Step

### Stage 0: Setup Database

Load raw Yelp JSON into DuckDB or Cloud SQL:

```bash
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset
```

What it does:

* Reads `yelp_academic_dataset_*.json` files.
* Creates tables such as `yelp_business`, `yelp_review`, and optionally `yelp_user`.
* Creates indices for faster querying.

Output:

* `yelp.duckdb` in the project root for local runs
* Indexed tables ready for preprocessing

### Stage 1: Preprocess

Build CSR matrix and ID mappings:

```bash
python -m src.preprocess_data --config configs/default.yaml
```

If the database has not been created yet:

```bash
python -m src.preprocess_data --config configs/default.yaml --setup-database --json-dir ~/Downloads/yelp_dataset
```

What it does:

* Loads positive user-item interactions from DuckDB or Cloud SQL.
* Uses `stars >= 4.0` as positive implicit feedback.
* Applies iterative 5-core filtering.
* Builds the filtered user-item CSR matrix.
* Creates user and item ID mappings.
* Splits users into train, validation, and test sets.

Example configuration:

```yaml
data:
  db_path: "../../Yelp-JSON/yelp.duckdb"
  state_filter: "PA"
  min_review_count: 20
  pos_threshold: 4.0
  train_test_split: 0.8
  val_split: 0.1
```

Output:

* `data/preprocessed_yelp/processed_train.npz`
* `data/preprocessed_yelp/user2index.pkl`
* `data/preprocessed_yelp/item2index.pkl`

### Stage 2: Train

Train ELSA and Top-K SAE models:

```bash
python -m src.train --config configs/default.yaml
```

What it does:

* Loads the preprocessed CSR matrix.
* Trains ELSA on train and validation sets.
* Trains Top-K SAE on ELSA latents.
* Saves models, metrics, split artifacts, and UI cache.
* Evaluates the model on the test split.

Output:

* `outputs/YYYYMMDD_HHMMSS/checkpoints/`
* `outputs/YYYYMMDD_HHMMSS/metrics/`
* `outputs/YYYYMMDD_HHMMSS/summary.json`
* `outputs/YYYYMMDD_HHMMSS/data/`
* `outputs/YYYYMMDD_HHMMSS/precomputed_ui_cache/`

Shared preprocessing artifacts may be stored under:

```text
outputs/_shared_preprocessed/<cache_key>/
```

and referenced from individual runs.

Expected duration: approximately 5 to 30 minutes, depending on dataset size and hardware.

### Stage 3: Label

Generate semantic labels for neurons:

```bash
python -m src.label
```

What it does:

* Loads the latest complete training run unless `--training-dir` is specified.
* If the latest run came from an experiment sweep, defaults follow the best run by NDCG@20.
* Supports weighted-category, matrix-based, metadata-only LLM, and review-based LLM labeling.
* Generates neuron labels, neuron embeddings, co-activation data, and interpretation metadata.

Labeling methods:

* **Weighted-category baseline**:

  * Uses top-activating businesses for a neuron.
  * Aggregates their categories weighted by activation strength.
  * De-prioritizes generic parent categories such as `Restaurants` and `Food`.

* **Matrix-based**:

  * Builds a category-neuron association matrix from sparse activations.
  * Applies TF-IDF style scoring.
  * Saves concept-neuron mappings used for concept steering.

* **LLM-based**:

  * Sends strongest examples and categories to Google Gemini.
  * Produces a natural-language label for each neuron.

* **Review-based LLM**:

  * Extends LLM labeling with useful review snippets.
  * Can capture atmosphere and user-experience signals beyond taxonomy.

Optional LLM setup:

```text
GOOGLE_API_KEY
```

stored in `.env` or `.streamlit/secrets.toml`.

Output:

* `neuron_labels.json`
* `labels_weighted-category.pkl`
* `labels_tag-based.pkl`
* `labels_matrix-based.pkl`
* `labels_llm-based.pkl`
* `labels_llm-review-based.pkl`
* `concept_mapping.pkl`
* `neuron_embeddings.pt`
* `neuron_category_metadata.json`
* `neuron_coactivation.json`

### Stage 4: Evaluate

Re-run deterministic ranking and model-quality evaluation for a saved run:

```bash
python -m src.evaluate
```

What it computes:

* Ranking metrics for K={10,20,50}:

  * Recall@K
  * Precision@K
  * Hit Rate@K
  * NDCG@K
  * MRR
  * MAP
* Model quality:

  * Coverage
  * Entropy
  * Sparsity
* Comparison:

  * ELSA-only vs ELSA+SAE metrics

Advanced usage:

```bash
python -m src.evaluate --checkpoint outputs/YYYYMMDD_HHMMSS --split test
python -m src.evaluate --checkpoint outputs/YYYYMMDD_HHMMSS --split test --sync-summary --sync-experiment-manifest
```

Output:

* `outputs/YYYYMMDD_HHMMSS/evaluation_test.json`
* Optional refreshed `summary.json` and experiment manifest

`src.evaluate` prefers exact saved holdout replay artifacts from the training run. If those artifacts are missing, it falls back to reconstructed holdout evaluation and marks the result as non-canonical.

## Data: Yelp Dataset

### What is Yelp Data?

The Yelp Academic Dataset contains businesses, reviews, users, check-ins, tips, and photos. In this project, the main training signal comes from the `business` and `review` files, while other files may be used for context, analysis, and UI.

### Download Instructions

1. Go to [https://www.yelp.com/dataset](https://www.yelp.com/dataset).
2. Download the JSON format.
3. Extract it to a folder, for example `~/Downloads/yelp_dataset/`.

Typical files include:

```text
yelp_academic_dataset_business.json
yelp_academic_dataset_review.json
yelp_academic_dataset_user.json
yelp_academic_dataset_checkin.json
yelp_academic_dataset_tip.json
yelp_academic_dataset_photo.json
```

The minimum required files for the main recommendation pipeline are `business` and `review`. Other files are optional depending on the analysis or UI features used.

### Data Filtering

The preprocessing step filters data by:

* **State filter**: the final thesis run uses `PA`.
* **Rating threshold**: reviews with stars >= 4.0 are treated as positive interactions.
* **5-core filtering**: users and items with fewer than 5 interactions are removed iteratively until convergence.

Final thesis run after filtering:

```text
Users: 37,323
Items: 12,793
Positive interactions: 525,257
Sparsity: 99.89%
```

## Configuration & Secrets

Copy templates before running the UI or scripts locally:

```bash
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

* `.env`: used by training scripts and utilities
* `.streamlit/secrets.toml`: used by the Streamlit UI
* Both templates are gitignored

Setup reminders:

* Make sure the Yelp JSON files are available before running `src.setup_database`.
* The local database file is `yelp.duckdb`.
* If using Cloud SQL or Streamlit Cloud, configure the corresponding secrets before starting the app.

## Configuration & Customization

### Training Configuration File

Default configuration:

```text
configs/default.yaml
```

Create a custom experiment:

```bash
cp configs/default.yaml configs/my_experiment.yaml
python -m src.train --config configs/my_experiment.yaml
```

Example configuration:

```yaml
data:
  db_path: "../../Yelp-JSON/yelp.duckdb"
  state_filter: "PA"
  min_review_count: 20
  pos_threshold: 4.0
  train_test_split: 0.8
  val_split: 0.1
  seed: 42

elsa:
  latent_dim: 512
  learning_rate: 0.0003
  batch_size: 1024
  num_epochs: 25
  patience: 10

sae:
  width_ratio: 2
  k: 32
  l1_coef: 0.0003
  learning_rate: 0.0003
  num_epochs: 50
  patience: 10

evaluation:
  k_values: [10, 20, 50]
  device: "cpu"
```

## Evaluation Metrics

Metrics are saved to the run directory, primarily in `summary.json` and optionally in `evaluation_test.json`.

### Ranking Metrics

| Metric      | Range | Meaning                                                 |
| ----------- | ----- | ------------------------------------------------------- |
| Recall@K    | 0-1   | Share of relevant held-out items retrieved in top-K     |
| Precision@K | 0-1   | Share of top-K recommendations that are relevant        |
| Hit Rate@K  | 0-1   | Share of users with at least one relevant item in top-K |
| NDCG@K      | 0-1   | Ranking quality with higher weight for top positions    |
| MRR         | 0-1   | Mean reciprocal rank of the first relevant item         |
| MAP         | 0-1   | Mean average precision                                  |

### Model Quality Metrics

| Metric   | Range | Meaning                                          |
| -------- | ----- | ------------------------------------------------ |
| Coverage | 0-1   | Share of catalog items recommended at least once |
| Entropy  | 0+    | Diversity of recommendations                     |
| Sparsity | 0-K   | Average number of active SAE neurons             |

### Thesis Final Run

Final run:

```text
20260427_030923
```

Selected metrics:

```text
NDCG@10 = 0.0667
Recall@10 = 0.1087
HR@10 = 0.2401
Coverage = 0.7924
Entropy = 0.7907
```

The thesis also compares ELSA, ELSA+SAE, selected baselines, and a random recommender reference. The comparison is intended as an experimental reference under the same data setting, not as a fully tuned benchmark study for every baseline model.

## Interpretability & Steering

### Neuron Labeling Process

Each SAE neuron is labeled to identify what type of feature it may represent.

#### Weighted-Category Labeling

* Finds businesses with the highest activation scores for a neuron.
* Extracts their Yelp categories.
* Aggregates category evidence weighted by activation strength.
* De-prioritizes generic categories such as `Restaurants` and `Food`.

#### Matrix-Based Labeling

* Builds a neuron-tag matrix from sparse activations.
* Applies TF-IDF style scoring.
* Exports concept-neuron mappings used directly for concept steering.

#### LLM-Based Labeling

* Sends strongest activating businesses and their categories to Google Gemini.
* Produces natural-language labels.
* Prompt templates are in [src/interpret/prompts.py](src/interpret/prompts.py).

#### Review-Based LLM Labeling

* Extends LLM labeling with selected useful review snippets.
* Uses activation-driven top businesses plus review context.
* Can capture information such as atmosphere, service, or user-experience cues.

### Co-Activation Analysis

Co-activation is computed from saved sparse test activations:

* Load saved sparse activations for the run.
* Center activations and compute neuron-neuron correlations.
* Save top positive and negative related neurons.
* Display the results in the Interpretability page.

### Top Activations and Wordcloud Methodology

For each neuron, the labeling stage precomputes category metadata:

* `top_items`: top activating businesses with activation value, name, categories, city/state, stars, and review count
* `category_weights`: category to activation evidence

The Interpretability UI renders:

* Top Activating Businesses
* Top Activating Categories
* Wordcloud
* Related Features, when available

Wordcloud payloads are cached under:

```text
precomputed_ui_cache/neuron_wordclouds/
```

### Feature Steering in UI

The Live Demo page supports two steering modes:

1. **Neuron Steering**

   * The user directly changes selected neuron activations.

2. **Concept / Superfeature Steering**

   * Query text is matched against saved concept mappings or LLM-derived labels.
   * The selected concept is converted to a neuron-weight patch.

Both modes are merged into a single steering vector before inference.

Steering inference path:

1. Encode user history to dense latent representation.
2. Encode dense latent representation to sparse SAE activations.
3. Apply neuron overrides in sparse space.
4. Decode back to dense latent space.
5. Interpolate with the baseline latent representation.
6. Re-rank items and compute rank/score deltas.

Recommendations are not rebuilt by retraining. They are recomputed from saved models with controlled latent-space edits.

### Live Workflow

1. Select a user.
2. Build a steering draft from neuron sliders and/or concept search.
3. Apply steering.
4. Inspect updated recommendations, map, and diagnostics.
5. Compare activation shifts and rank movement.

## Interactive Dashboard

### Local Access

```bash
streamlit run src/ui/main.py
```

The app opens at:

```text
http://localhost:8501
```

### Cloud Access

Deployed version:

[https://explainable-steerable-poi-recommendations.streamlit.app/](https://explainable-steerable-poi-recommendations.streamlit.app/)

### Pages

**Home**

* Overview of the thesis demo
* Quick stats
* Links to main app sections

**Dataset**

* Yelp dataset statistics
* Distribution by state, city, categories, ratings, and time
* Optional filters

**Results**

* Evaluation metrics table
* Model comparison
* Steering evaluation tab
* Generated plots

**Live Demo**

* User selection
* Neuron steering
* Concept steering
* Recommendation cards
* Map visualization
* Rank deltas, score deltas, activation shifts, CPR, and NDCG@K where available

**Interpretability**

* Neuron browser
* Label source selection
* Top activating categories and businesses
* Wordcloud
* Related features and co-activation analysis where available

The UI reads saved artifacts from the selected run and does not retrain models or regenerate labels at runtime.

## Notebooks for Exploration

| Notebook                        | Purpose                                          |
| ------------------------------- | ------------------------------------------------ |
| `00_preprocessing.ipynb`        | Build the CSR interaction matrix and ID mappings |
| `01_data_exploration.ipynb`     | Explore Yelp data distribution and statistics    |
| `02_training.ipynb`             | Train ELSA and SAE models interactively          |
| `03_neuron_labeling_demo.ipynb` | Browse neuron labels and feature semantics       |
| `04_evaluation_analysis.ipynb`  | Analyze metrics and comparisons                  |

Run notebooks with:

```bash
jupyter notebook notebooks/
```

## Deployment

### Local Development

Use the Quick Start steps and then run:

```bash
streamlit run src/ui/main.py
```

### Streamlit Cloud

1. Push code to GitHub.
2. Go to [https://share.streamlit.io/](https://share.streamlit.io/).
3. Select the repository and deploy.
4. Configure secrets for Cloud SQL, GCS, and LLM labeling if needed.

Typical secrets:

```text
CLOUDSQL_INSTANCE
CLOUDSQL_USER
CLOUDSQL_PASSWORD
CLOUDSQL_DATABASE
GCS_BUCKET_NAME
GOOGLE_CLOUD_PROJECT
GOOGLE_API_KEY
```

### Cloud SQL Backend

Use PostgreSQL instead of DuckDB:

```bash
export CLOUDSQL_INSTANCE=project:region:instance
export CLOUDSQL_USER=postgres
export CLOUDSQL_PASSWORD=...
export CLOUDSQL_DATABASE=postgres
```

Database setup:

```bash
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset --cloud-sql
```

### GCS Integration

Store models and results on Google Cloud Storage:

```bash
export GCS_BUCKET_NAME=my-bucket
export GOOGLE_CLOUD_PROJECT=my-project
```

Artifacts are uploaded automatically when configured.

### Current Data Layout

* Local development uses DuckDB by default.
* Cloud deployments use the same logical schema through Cloud SQL.
* The active state filter, review-count filter, model dimensions, and evaluation settings come from `configs/default.yaml`.
* Notebook, labeling, and UI outputs are read from the latest completed run in GCS first, with local `outputs/` used as an offline fallback.

## Notes on Reproducibility

This repository contains the implementation and configuration needed to reproduce the pipeline. Some large artifacts, raw Yelp data, cloud credentials, and API keys are not included in the repository because of size, licensing, and security reasons.

The thesis text is the primary source for the complete explanation of the method, architecture, implementation decisions, and evaluation. The repository documentation is intended mainly as a practical guide for navigating and running the code.

## References

* Spišák, M. et al. (2026). *From Knots to Knobs: Towards Steerable Collaborative Filtering Using Sparse Autoencoders.*
* Wang, J. et al. (2026). *Understanding Internal Representations of Recommendation Models with Sparse Autoencoders.*
* Vančura, V. et al. (2022). *Scalable Linear Shallow Autoencoder for Collaborative Filtering.*
* Bostandjiev, S. et al. (2012). *TasteWeights: A Visual Interactive Hybrid Recommender System.*

Cite this thesis:

Povolná, E. (2025). *Explainable and Steerable POI Recommendations.* Diploma thesis, Charles University.
