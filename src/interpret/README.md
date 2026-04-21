# Neural Interpretation Module (`src/interpret/`)

This module provides tools for interpreting and labeling neurons in sparse autoencoders (SAEs) trained on collaborative filtering embeddings.

## Module Overview

### Files

| File | Purpose |
|------|---------|
| **`activations.py`** | Low-level activation extraction and analysis utilities |
| **`neuron_interpreter.py`** | High-level LLM-based neuron interpretation (uses Google Gemini) |
| **`neuron_labeling.py`** | Tag-based neuron labeling from business categories |
| **`USAGE_GUIDE.md`** | Complete usage examples and patterns |

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

## Alternative: High-Level LLM-Based Interpretation

For semantic interpretation using LLMs:

```python
from src.interpret.neuron_interpreter import NeuronInterpreter

# Initialize with Google Gemini (requires GOOGLE_API_KEY)
interpreter = NeuronInterpreter()

# Interpret a neuron
label = interpreter.label_neuron(neuron_profiles[neuron_id])
```

## What Each Module Does

### `activations.py` - Low-level Utilities
✓ Extract sparse activations from SAE  
✓ Find max/min activating items per neuron  
✓ Collect business metadata  
✓ Build neuron profiles  

**When to use:** For batch processing, custom analysis pipelines, or when you need fine-grained control.

### `neuron_interpreter.py` - LLM-Based Interpretation
✓ Generate semantic descriptions using LLMs  
✓ Uses Google Gemini for semantic interpretation  
✓ Extract tags and reasons from LLM responses  

**When to use:** For generating human-readable interpretations of neuron behavior.

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
build_neuron_profile() + neuron_interpreter.label_neuron()
    ↓
Labeled neuron profiles (with tags and interpretations)
```

## Examples

See [USAGE_GUIDE.md](USAGE_GUIDE.md) for complete working examples from the notebooks.

## Notes

- All functions expect PyTorch tensors for activation data
- Business metadata requires a DataFrame with columns: `business_id`, `categories`, `name`, `city`, `stars`, `review_count`
- Item2index mapping should map business IDs to their position in the CSR matrix
- Batch processing is recommended for large datasets to avoid memory issues

