# NeuronInterpreter & SuperfeatureGenerator Usage Guide

## Overview

Both `neuron_interpreter.py` classes now support **two LLM providers**:
- **GitHub Models (GPT-4o)** - Recommended for quality and availability
- **Google Gemini** - Free tier with rate limiting

## Quick Start

### 1. Set up your API credentials

**For GitHub Models (GPT-4o):**
```bash
export GITHUB_TOKEN=<your_github_pat>
# Get PAT from: https://github.com/settings/tokens
```

**For Gemini:**
```bash
export GOOGLE_API_KEY=<your_gemini_api_key>
# Get key from: https://aistudio.google.com/app/apikey
```

### 2. Use in code

#### Auto-detect provider (uses whatever is available)
```python
from src.interpret.neuron_interpreter import NeuronInterpreter

interpreter = NeuronInterpreter()
# Automatically uses GITHUB_TOKEN if available, falls back to GOOGLE_API_KEY
```

#### Explicitly specify provider
```python
# Use GitHub Models
interpreter = NeuronInterpreter(provider="github_models")

# Use Gemini
interpreter = NeuronInterpreter(provider="gemini")

# Custom model name
interpreter = NeuronInterpreter(
    provider="github_models",
    model_name="gpt-4o"
)
```

#### Label neurons
```python
label = interpreter.label_neuron(
    neuron_idx=42,
    max_activating=[{"name": "Starbucks", "category": "Coffee"}],
    zero_activating=[{"name": "Gym", "category": "Sports"}]
)
print(f"Neuron 42: {label}")
```

### 3. SuperfeatureGenerator (same API)

```python
from src.interpret.neuron_interpreter import SuperfeatureGenerator

# Works exactly like NeuronInterpreter
generator = SuperfeatureGenerator(provider="github_models")

# Cluster and generate superlabels
clusters = generator.cluster_labels_by_similarity(labels)
superlabels = generator.generate_superlabels(clusters)
```

## In Jupyter Notebooks

### Complete example

```python
from src.interpret.neuron_interpreter import NeuronInterpreter
import os

# Setup
provider = "github_models" if os.getenv("GITHUB_TOKEN") else "gemini"
interpreter = NeuronInterpreter(provider=provider)

# Label neurons from neuron_profiles
labels = {}
for neuron_id, profile in neuron_profiles.items():
    max_items = profile['max_activating']['indices'][:5]
    zero_items = profile['zero_activating']['indices'][:5]
    
    # Format as business objects (example)
    max_examples = [{"name": f"Item {i}", "category": "POI"} for i in max_items]
    zero_examples = [{"name": f"Item {i}", "category": "Other"} for i in zero_items]
    
    label = interpreter.label_neuron(neuron_id, max_examples, zero_examples)
    if label:
        labels[neuron_id] = label

print(f"Labeled {len(labels)} neurons")
```

## Provider Comparison

| Feature | GitHub Models | Gemini |
|---------|---------------|--------|
| **Quality** | ⭐⭐⭐⭐⭐ (GPT-4o) | ⭐⭐⭐⭐ |
| **Speed** | ⭐⭐⭐⭐ | ⭐⭐⭐ (with rate limiting) |
| **Cost** | Free on GitHub | Free tier: 5 req/min, 10/day |
| **Setup** | Requires GitHub account | Free API key |
| **Availability** | Reliable | Rate-limited |

## Error Handling

```python
try:
    interpreter = NeuronInterpreter(provider="github_models")
except ValueError as e:
    print(f"Invalid provider or missing API key: {e}")
except ImportError as e:
    print(f"Required library not installed: {e}")
    # pip install openai (for GitHub Models)
    # pip install google-generativeai (for Gemini)

try:
    label = interpreter.label_neuron(...)
except Exception as e:
    print(f"Labeling failed: {e}")
```

## Rate Limiting

GitHub Models and Gemini have different rate limits. The code includes:
- Automatic retry logic with exponential backoff (3 retries)
- Configurable delay between attempts: `rate_limit_delay` parameter

```python
label = interpreter.label_neuron(
    neuron_idx=0,
    max_activating=[...],
    zero_activating=[...],
    max_attempts=3,          # Retries
    rate_limit_delay=0.5     # Delay between retries (seconds)
)
```

## Environment Variables

The module reads from:

```bash
# GitHub Models
GITHUB_TOKEN=<your_github_pat>

# Gemini
GOOGLE_API_KEY=<your_api_key>
```

Or pass directly:
```python
interpreter = NeuronInterpreter(
    provider="github_models",
    api_key="your_token_here"
)
```

## Support Both in Production

Use provider detection to fall back gracefully:

```python
import os
from src.interpret.neuron_interpreter import NeuronInterpreter

def get_interpreter():
    if os.getenv("GITHUB_TOKEN"):
        return NeuronInterpreter(provider="github_models")
    elif os.getenv("GOOGLE_API_KEY"):
        return NeuronInterpreter(provider="gemini")
    else:
        raise ValueError("No LLM API credentials configured")

interpreter = get_interpreter()
```
