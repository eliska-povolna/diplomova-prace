# Interpretable POI Recommender System with Sparse Autoencoders

> **Diploma thesis** — Eliška Povolná, 2025/2026


Modern recommender systems achieve high accuracy through complex latent representations, but these are often opaque and hard for users to influence. This project integrates **Sparse Autoencoders (SAE)** into a collaborative-filtering architecture (**ELSA**) to transform dense latent vectors into **interpretable, user-controllable preference representations** — applied to **Points of Interest (POI)** recommendation using **Yelp** data.

The resulting model learns sparse, semantically meaningful features that serve as interactive "knobs" — users can directly steer recommendations by adjusting their preference profile, rather than treating the system as a black box.

---

## Contents

- [Quick Start](#quick-start) — Try the demo or run locally
- [System Architecture](#system-architecture) — How data flows through the system
- [Core Algorithms](#core-algorithms) — ELSA and Sparse Autoencoders explained
- [Pipeline: Step-by-Step](#pipeline-step-by-step) — Database setup through evaluation
- [Data: Yelp Dataset](#data-yelp-dataset) — Structure and download
- [Getting Started](#getting-started) — Developer setup guide
- [Configuration & Customization](#configuration--customization) — Hyperparameters
- [Evaluation Metrics](#evaluation-metrics) — What is measured and how to interpret results
- [Interpretability & Steering](#interpretability--steering) — Feature labeling and interactive control
- [Interactive Dashboard](#interactive-dashboard) — Using the Streamlit UI
- [Notebooks for Exploration](#notebooks-for-exploration) — Analysis and visualization
- [Deployment](#deployment) — Local, Streamlit Cloud, and GCP options
- [License & Citation](#license--citation)

---

## Quick Start

### Option 1: Try the Streamlit Demo Online

Open the live demo (no setup required):  
**[https://explainable-steerable-poi-recommendations.streamlit.app/](https://explainable-steerable-poi-recommendations.streamlit.app/)**

### Option 2: Run the Interactive Dashboard Locally

```bash
# 1. Clone and setup
git clone https://github.com/eliska-povolna/Diplomov-pr-ce.git
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

# Label neurons for interpretability (auto-detects latest checkpoint)
python -m src.label

# Evaluate on test set (auto-detects latest checkpoint)
python -m src.evaluate

# View results in interactive dashboard
streamlit run src/ui/main.py
```

Output is saved to `outputs/YYYYMMDD_HHMMSS/` with models, metrics, and interpretations.

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
[Label] → Neuron interpretations (tag-based, LLM-based)
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

**What it does:**
- Loads user-item interactions from DuckDB
- Filters by state and review count (configurable in config)
- Builds user→item interaction CSR matrix
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
- `data/preprocessed_yelp/processed_train.npz` (CSR matrix)
- `data/preprocessed_yelp/user2index.pkl` (user ID mapping)
- `data/preprocessed_yelp/item2index.pkl` (business ID mapping)

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
- `outputs/YYYYMMDD_HHMMSS/checkpoints/` — Trained models
- `outputs/YYYYMMDD_HHMMSS/metrics/` — Training logs
- `outputs/YYYYMMDD_HHMMSS/summary.json` — Training config + results

**Expected duration:** 5-30 minutes (depends on data size and hardware)

### Stage 3: Label (Optional)

Generate semantic labels for neurons:

```bash
python -m src.label
```

**What it does:**
- Loads trained SAE from latest checkpoint (auto-detected)
- For each neuron, extracts its activation patterns
- Generates semantic labels using:
  - **Tag-based**: Categories from businesses with high activation
  - **LLM-based**: Sends patterns to LLM (Gemini or GitHub Models)
- Stores labels and neuron embeddings

**Optional LLM APIs (auto-detects which is available):**
- `GOOGLE_API_KEY` in `.env` (for Google Gemini)
- `GITHUB_TOKEN` in `.env` (for GitHub Models / GPT-4o)

**Output:**
- `outputs/YYYYMMDD_HHMMSS/neuron_interpretations/`
  - `labels_tag-based.pkl` — Dictionary of neuron → tags
  - `labels_llm-based.pkl` — Dictionary of neuron → LLM description
  - `neuron_embeddings.pt` — Embedding vectors for each neuron

### Stage 4: Evaluate

Compute ranking and model quality metrics on test set:

```bash
python -m src.evaluate
```

This auto-detects the latest checkpoint and evaluates on the test set.

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
```

**Output:**
- `outputs/YYYYMMDD_HHMMSS/evaluation_test.json` — All metrics

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
- **State**: Default "PA" (Pennsylvania). Set to `null` in config for all states.
- **Review count**: Keep users with ≥ 20 reviews (configurable)
- **Rating threshold**: Stars ≥ 4.0 = positive feedback (configurable)

This reduces the full dataset to a manageable subset for faster training.



## Configuration & Secrets

### Prerequisites

- Python 3.10+
- ~10GB free disk space (for data + models)
- GPU optional (training works on CPU, but slower)

### Environment Setup

1. **Clone and setup:**
   ```bash
   git clone https://github.com/eliska-povolna/Diplomov-pr-ce.git
   cd Diplomov-pr-ce
   python -m venv .venv
   source .venv/bin/activate           # Linux/Mac
   # or
   .venv\Scripts\activate              # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r src/requirements.txt
   ```

3. **Configure secrets** — Copy templates and add your credentials:
   ```bash
   cp .env.example .env
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   - **`.env`**: For training scripts (GOOGLE_API_KEY, GITHUB_TOKEN optional, Cloud SQL credentials optional)
   - **`.streamlit/secrets.toml`**: For Streamlit UI (same variables as .env)
   - Both are automatically excluded from git for security

4. **Download and setup Yelp data:**
   ```bash
   # Download from https://www.yelp.com/dataset (JSON format)
   python -m src.setup_database --json-dir ~/Downloads/yelp_dataset
   ```
   This creates `yelp.duckdb` (~5-10 minutes for full dataset).

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

---

## Interpretability & Steering

### Neuron Labeling Process

Each SAE neuron is labeled to identify what "feature" it represents:

#### Tag-Based Labeling
- For each neuron, find businesses with highest activation
- Extract categories (tags) from those businesses
- Aggregate: neuron ≈ "coffee_shops + casual" (most common tags)

#### LLM-Based Labeling
- Send activation patterns to Google Gemini API
- Prompt: "These are the top business categories for this neuron: ... What does this neuron represent?"
- Result: Human-readable label (e.g., "user likes casual dining and outdoor seating")

**See:** `src/interpret/neuron_labeling.py`

### Co-Activation Analysis

Neurons don't work in isolation. Co-activation matrices show which neurons tend to activate together:

```
Neuron A (coffee)  ──────────┐
                             ├──→ Similar users
Neuron B (outdoor) ──────────┘
```

Available in UI: **Interpretability Tab → Co-Activation Heatmap**

### Feature Steering in UI

**Live Demo Tab** allows interactive steering:

1. Select a user (or create synthetic profile)
2. Adjust neuron sliders (increase/decrease activation)
3. See updated top-K recommendations
4. Attribution shows which neurons influenced the ranking

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
- Top-K recommendations list
- POI details (name, category, rating, reviews)
- Photo gallery (if Yelp photos available)

**🔍 Interpretability**
- Neuron browser (sorted by activation frequency)
- Neuron labels (tag-based and LLM-based)
- Co-activation heatmap
- Feature visualization

---

## Notebooks for Exploration

Interactive Jupyter notebooks for analysis and visualization:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Explore Yelp data distribution, user/item statistics |
| `02_coactivation_analysis.ipynb` | Analyze co-activation patterns between neurons |
| `03_neuron_interpretability.ipynb` | Browse neuron labels and feature semantics |
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

```bash
# Setup (one-time)
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
cp .env.example .env
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
python -m src.setup_database --json-dir ~/Downloads/yelp_dataset

# Run UI
streamlit run src/ui/main.py
```

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

---

## License & Citation

License: See [LICENSE](LICENSE)

**Citation (thesis):**
```bibtex
@thesis{povolna2026,
  title={Interpretable POI Recommender System with Sparse Autoencoders},
  author={Povolná, Eliška},
  year={2026},
  school={Charles University}
}
```

**Key Papers:**
- Sparse Autoencoders: [Sharkey et al., 2022](https://arxiv.org/abs/2309.08600)
- ELSA: Collaborative filtering with linear autoencoders
- TopK SAE: Sparse feature decomposition with activation patterns
