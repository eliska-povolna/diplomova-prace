# Neural Interpretation Module (`src/interpret/`)

This module provides tools for interpreting and labeling neurons in sparse autoencoders (SAEs) trained on collaborative filtering embeddings.

## Module Overview

### Files

| File | Purpose |
|------|---------|
| **`activations.py`** | Low-level activation extraction and analysis utilities |
| **`neuron_labeling.py`** | Weighted-category baseline labels, Gemini labels, embeddings, and superfeatures |
| **`matrix_based_labeling.py`** | TF-IDF concept-neuron mapping and grounded statistical labels |

## Current Workflow

- Labeling runs from the latest complete training output by default.
- `python -m src.label` uses Google Gemini via `GOOGLE_API_KEY` for LLM labels.
- Batch labeling and co-activation generation are the current supported paths.
- `GITHUB_TOKEN` is not used for current labeling.
- Label outputs are written under `outputs/<run_id>/neuron_interpretations/`.

## Current Outputs

- `neuron_labels.json` or equivalent label mapping files for each neuron.
- `neuron_embeddings.pkl` / embedding cache for semantic comparison.
- `similarity_matrix.pkl` for label similarity and clustering.
- `superfeatures.json` for grouped feature families.
- `coactivation_matrix.json` for neuron co-activation analysis.

## Quick Start

### 1. Extract Neuron Activations

```python
from src.interpret.activations import extract_sparse_activations

# Get sparse activations for all items through the SAE
h_sparse = extract_sparse_activations(
    embeddings=Z_train,        # Latent vectors from ELSA (num_items, 512)
    sae=sae_model,             # Trained SAE model
    batch_size=256,
    device="cpu"
)
# Returns: (num_items, num_neurons) tensor of sparse activations
```

### 2. Find Max/Zero Activating Items

```python
from src.interpret.activations import get_max_activating_items, get_zero_activating_items

# Create index mappings
item2index = {business_id: idx for idx, business_id in enumerate(business_ids)}
index2item = {idx: bid for bid, idx in item2index.items()}

# Get top 10 items per neuron with highest activation
max_activating = get_max_activating_items(h_sparse, item2index, num_examples=10)
# Returns: {neuron_id: [(business_id, activation_value), ...]}

# Get bottom 5 items with lowest activation
zero_activating = get_zero_activating_items(h_sparse, item2index, num_examples=5)
# Returns: {neuron_id: [business_id, ...]}
```

### 3. Collect Business Metadata

```python
from src.interpret.activations import collect_business_metadata

# Get metadata for specific businesses
business_ids = [bid for bids in max_activating.values() for bid, _ in bids[:5]]
metadata = collect_business_metadata(business_ids, businesses_df)
# Returns: {business_id: {"name": ..., "categories": [...], ...}}
```

### 4. Build Complete Neuron Profile

```python
from src.interpret.activations import build_neuron_profile

# Build profile for a single neuron
profile = build_neuron_profile(
    neuron_idx=42,
    max_activating_items=max_activating[42],
    zero_activating_items=zero_activating[42],
    business_metadata=metadata
)
```

## What Each Module Does

### `activations.py` - Low-level Utilities
✓ Extract sparse activations from SAE  
✓ Find max/min activating items per neuron  
✓ Collect business metadata  
✓ Build neuron profiles  

**When to use:** For batch processing, custom analysis pipelines, or when you need fine-grained control.

### `neuron_labeling.py` - Label Generation and Superfeatures
✓ Weighted-category baseline labels  
✓ Gemini-based neuron naming  
✓ Review-enriched Gemini labels  
✓ Embeddings and superfeature clustering  

**When to use:** For the saved labeling pipeline used by `python -m src.label`.

### `neuron_labeling.py` - Tag-Based Labeling
✓ Extract categories from max-activating items  
✓ Generate labels using business metadata  
✓ Count category frequencies  

**When to use:** For quick category-based labels without API calls.

## Data Flow

```
Input Data (CSR sparse matrix)
    ↓
[ELSA Encoding] (collaborative_filtering.py)
    ↓
Latent vectors (512-dim)
    ↓
extract_sparse_activations() ← src/interpret/activations.py
    ↓
Sparse activations (num_items, num_neurons)
    ↓
get_max/zero_activating_items()
    ↓
Item lists with activation values
    ↓
collect_business_metadata()
    ↓
Business details (categories, names, etc.)
    ↓
build_neuron_profile() + the saved labeling pipeline from `python -m src.label`
    ↓
Labeled neuron profiles (with tags and interpretations)
```

## Examples

See [notebooks/03_neuron_labeling_demo.ipynb](../../notebooks/03_neuron_labeling_demo.ipynb) for working examples from the notebook workflow.

## Notes

- All functions expect PyTorch tensors for activation data
- Business metadata requires a DataFrame with columns: `business_id`, `categories`, `name`, `city`, `stars`, `review_count`
- Item2index mapping should map business IDs to their position in the CSR matrix
- Batch processing is recommended for large datasets to avoid memory issues

