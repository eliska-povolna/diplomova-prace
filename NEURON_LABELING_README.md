# Neuron Labeling and Interpretation

This module provides neural network neuron interpretation using multiple complementary approaches.

## Features

### 1. Tag-Based Labeling (`TagBasedLabeler`)
Analyzes the categories and tags of businesses that maximally activate each neuron.

**How it works:**
- Extracts categories and tags from top-activating businesses
- Uses TF-IDF-like weighting to find the most discriminative features
- Generates human-readable labels directly from data

**Pros:**
- Fast (no API calls)
- Directly interpretable (based on explicit tags)
- No cost

**Cons:**
- Limited to tag vocabulary in data
- May miss conceptual similarities

### 2. LLM-Based Labeling (`LLMBasedLabeler`)
Uses Google's Gemini API (free tier) to analyze max/zero activating examples with deep semantic understanding.

**How it works:**
- Extracts max-activating and zero-activating examples for each neuron
- Sends formatted examples to Gemini with a specialized prompt
- Applies Occam's razor principle to find simplest explanation
- Handles rate limiting and retries automatically

**Pros:**
- Semantic understanding beyond explicit tags
- Flexible label length and complexity
- Captures implicit concepts

**Cons:**
- Requires API calls (but uses free Gemini tier)
- Slower than tag-based
- May hallucinate with poor examples

### 3. Neuron Embeddings (`NeuronEmbedder`)
Creates semantic embeddings of neuron labels using sentence transformers.

**How it works:**
- Encodes generated labels into fixed-dimensional vectors
- Uses `sentence-transformers/all-distilroberta-v1` model
- Computes cosine similarity between label embeddings

**Use cases:**
- Finding conceptually similar neurons
- Visualizing neuron relationships
- Clustering neurons by semantic similarity

### 4. Superfeature Generation (`SuperfeatureGenerator`)
Groups similar neurons into "feature families" and generates abstract super-labels.

**How it works:**
1. Clusters neurons based on embedding similarity (threshold: 0.7 default)
2. For each cluster, sends member labels to Gemini
3. Generates a unified super-label capturing the common concept

**Example:**
- Individual labels: "Italian restaurants", "French bistros", "Spanish tapas bars"
- Cluster: "European cuisine"
- Super-label: "Fine dining and international cuisine"

## Usage

### Basic Usage

```python
from src.interpret.neuron_labeling import (
    TagBasedLabeler,
    LLMBasedLabeler,
    NeuronEmbedder,
    SuperfeatureGenerator,
)

# 1. Label neurons using tag-based approach
labeler = TagBasedLabeler()
labels = labeler.label_neurons(neuron_profiles, business_metadata)

# 2. Or use LLM-based approach
labeler = LLMBasedLabeler(api_key="your-gemini-key")
labels = labeler.label_neurons(neuron_profiles, business_metadata)

# 3. Create embeddings
embedder = NeuronEmbedder()
embeddings, neuron_indices = embedder.embed_labels(labels)
similarity_matrix = embedder.compute_similarity_matrix(embeddings)

# 4. Generate superfeatures
generator = SuperfeatureGenerator()
clusters = generator.cluster_neurons(similarity_matrix, neuron_indices)
superfeatures = generator.create_superfeatures(clusters, labels)
```

### Using the CLI Script

```bash
# Run complete pipeline with both methods
python label_neurons.py \
    --model_path models/sae_model.pt \
    --data_path data/processed_yelp_easystudy \
    --business_metadata data/business_metadata.pkl \
    --output_dir data/neuron_interpretations \
    --method both \
    --gemini_api_key YOUR_API_KEY \
    --similarity_threshold 0.7
```

**Arguments:**
- `--model_path`: Path to trained SAE model checkpoint
- `--data_path`: Path to preprocessed data directory
- `--business_metadata`: Pickle file with business metadata
- `--output_dir`: Where to save results
- `--method`: `tag-based`, `llm-based`, or `both`
- `--gemini_api_key`: Gemini API key (or use GOOGLE_API_KEY env var)
- `--similarity_threshold`: Threshold for clustering (0-1)
- `--top_k`: Number of max/zero examples per neuron

### Output Files

The script saves:
- `labels_tag-based.pkl`: Tag-based neuron labels (if method includes)
- `labels_llm-based.pkl`: LLM-based neuron labels (if method includes)
- `neuron_embeddings.pt`: Label embeddings and similarity matrix
- `superfeatures.pkl`: Clustered superfeatures with parent labels
- `summary.pkl`: Summary metrics and configuration

## API Key Setup

### Google Gemini API (Free)

1. Get free API key: https://aistudio.google.com/app/apikey
2. Set environment variable:
   ```bash
   export GOOGLE_API_KEY="your-key-here"
   ```
3. Or pass directly:
   ```python
   labeler = LLMBasedLabeler(api_key="your-key-here")
   ```

### Free Tier Limits
- 60 requests per minute
- Built-in rate limiting handles this automatically

## Example: Comparative Analysis

```python
# Load data
neuron_profiles = ...  # From SAE analysis
business_metadata = ... # Business data with categories/tags

# Method 1: Tag-based
tag_labeler = TagBasedLabeler(max_tags_per_neuron=3)
tag_labels = tag_labeler.label_neurons(neuron_profiles, business_metadata)

# Method 2: LLM-based
llm_labeler = LLMBasedLabeler()
llm_labels = llm_labeler.label_neurons(neuron_profiles, business_metadata)

# Compare
for nid in list(tag_labels.keys())[:5]:
    print(f"Neuron {nid}:")
    print(f"  Tag-based:  {tag_labels[nid]}")
    print(f"  LLM-based:  {llm_labels[nid]}")
```

## Prompts Used

### Phase 1: Neuron Interpretation
Analyzes activation patterns with two key principles:
1. **Max-activating examples**: What excites this neuron?
2. **Zero-activating examples**: What suppresses it?
3. **Occam's razor**: Select simplest explanation

### Phase 2: Superfeature Synthesis
Given a cluster of similar labels, finds the abstract parent concept that unifies them.

## Implementation Details

### Tag-Based Algorithm
1. Extract categories/tags from max-activating items
2. Weight by item arrival position (higher position = higher weight)
3. Aggregate across top-10 items
4. Sort by total weight and select top-3
5. Join with "and" to form label

### LLM-Based Algorithm
1. Format examples: name, categories, city, activation value
2. Send to Gemini with system prompt
3. Extract label from response (format: "LABEL: <text>")
4. Retry with exponential backoff on failure
5. Rate-limit to ~1 req/sec

### Embedding Algorithm
1. Use pre-trained sentence-transformers model
2. Encode each neuron label
3. Compute cosine similarity between all pairs
4. Store both embeddings and similarity matrix

### Superfeature Clustering
1. Greedy clustering: start with first neuron, find all similar ones
2. Mark clustered neurons, move to next unclustered
3. Only keep clusters with 2+ neurons
4. Send cluster labels to Gemini for super-label generation
5. Store cluster composition and super-label

## Troubleshooting

### "google-generativeai not installed"
```bash
pip install google-generativeai
```

### "sentence-transformers not installed"
```bash
pip install sentence-transformers
```

### API rate limiting
- Script automatically handles 1 req/sec rate limiting
- Gemini free tier allows 60 requests/min (safe window)

### Poor LLM labels
- Check max-activating examples are meaningful
- Ensure business metadata has good category data
- Try increasing `top_k` for more context examples

### Low embedding quality
- Use phrase-based labels (not single words)
- Ensure labels describe actual concepts
- Can swap model name: `all-mpnet-base-v2` (slower, better)

## Extending the Module

### Custom Labeler
```python
from src.interpret.neuron_labeling import NeuronLabeler

class MyCustomLabeler(NeuronLabeler):
    def __init__(self):
        super().__init__("my-method")
    
    def label_neurons(self, neuron_profiles, business_metadata):
        # Your implementation
        return {neuron_id: label, ...}
```

### Custom Clustering
```python
from sklearn.cluster import DBSCAN

# Replace greedy clustering with DBSCAN
clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
cluster_labels = clustering.fit_predict(similarity_matrix)
```

## References

This module is inspired by:
- Max-Activating and Zero-Activating examples (neuron interpretation technique)
- Occam's Razor principle (simplest explanation principle)
- Sentence-BERT embeddings for semantic similarity
- Superfeature concepts from mechanistic interpretability research
