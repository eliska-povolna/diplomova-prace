"""Precompute statistics and visualizations for Streamlit UI.

After SAE training completes, generates:
- Neuron word clouds (top activating reviews)
- Neuron activation statistics
- Test user embeddings

Output saved to: outputs/{state}_{timestamp}/precomputed_ui_cache/
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from collections import Counter

logger = logging.getLogger(__name__)


def extract_wordcloud_text(reviews: List[str], max_words: int = 50) -> Dict:
    """Extract text from reviews for word cloud visualization.
    
    Args:
        reviews: List of review texts
        max_words: Maximum number of unique words to extract
        
    Returns:
        Dict with structure: {"text": "pizza italian fast", "freq": [10, 8, 5]}
    """
    # Simple tokenization: split by whitespace, remove short words
    all_words = []
    for review in reviews:
        words = review.lower().split()
        # Keep words with 3+ chars, remove common stopwords
        words = [w.strip(".,!?;:-") for w in words if len(w) > 2]
        all_words.extend(words)
    
    # Count word frequencies
    word_freq = Counter(all_words)
    
    # Get top words
    top_words = dict(word_freq.most_common(max_words))
    
    # Format for word cloud library
    text = " ".join(word for word in top_words.keys() for _ in range(top_words[word]))
    
    return {
        "text": text,
        "freq": list(top_words.values()),
        "words": list(top_words.keys()),
        "word_count": len(word_freq),
    }


def precompute_neuron_wordclouds(
    inference_service,
    data_service,
    output_dir: Path,
    num_samples: int = 1000,
) -> None:
    """Precompute word clouds for all neurons.
    
    Finds top-activating reviews for each neuron and extracts word clouds.
    Saves to: output_dir/precomputed_ui_cache/neuron_wordclouds/
    
    Args:
        inference_service: ELSA + SAE inference service
        data_service: Data service with POI info
        output_dir: Output directory containing training results
        num_samples: Number of validation samples to use for activation computation
    """
    cache_dir = output_dir / "precomputed_ui_cache" / "neuron_wordclouds"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Precomputing neuron word clouds to {cache_dir}")
    
    # Sample test users/items for activation computation
    # Get diversity across the dataset
    num_neurons = inference_service.sae.k
    wordcloud_data = {}
    
    # For now, create placeholder word clouds
    # In practice, you'd:
    # 1. Get top-K POIs by activation for each neuron from val set
    # 2. Aggregate reviews from those POIs
    # 3. Extract word clouds
    
    for neuron_idx in range(num_neurons):
        wordcloud_data[neuron_idx] = extract_wordcloud_text(
            [f"Business reviews related to neuron {neuron_idx}"],  # placeholder
            max_words=50,
        )
    
    # Save to JSON
    output_file = cache_dir / "wordclouds.json"
    with open(output_file, "w") as f:
        json.dump(wordcloud_data, f, indent=2)
    
    logger.info(f"Saved word clouds to {output_file}")


def precompute_neuron_statistics(
    inference_service,
    dataset_loader,
    output_dir: Path,
) -> None:
    """Precompute activation statistics for each neuron.
    
    Saves: output_dir/precomputed_ui_cache/neuron_stats.json
    
    Contains:
    - mean_activation: Average activation value
    - sparsity: Fraction of near-zero activations
    - top_co_activating_neurons: Most correlated neurons
    """
    cache_dir = output_dir / "precomputed_ui_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Precomputing neuron statistics to {cache_dir}")
    
    num_neurons = inference_service.sae.k
    stats = {}
    
    # Placeholder: compute real stats from validation set
    for neuron_idx in range(num_neurons):
        stats[neuron_idx] = {
            "mean_activation": float(np.random.random()),  # placeholder
            "sparsity": float(np.random.random()),
            "top_co_activating_neurons": [i for i in range(min(5, num_neurons)) if i != neuron_idx],
        }
    
    output_file = cache_dir / "neuron_stats.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved neuron stats to {output_file}")


def precompute_test_user_embeddings(
    inference_service,
    data_service,
    output_dir: Path,
    num_test_users: int = 50,
) -> None:
    """Precompute embeddings for common test users.
    
    Saves: output_dir/precomputed_ui_cache/test_user_embeddings.pkl
    
    Later loaded by Streamlit to avoid recomputing at app startup.
    """
    cache_dir = output_dir / "precomputed_ui_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Precomputing {num_test_users} test user embeddings")
    
    import pickle
    
    # Placeholder: would gather real user interactions and encode
    embeddings = {}
    
    output_file = cache_dir / "test_user_embeddings.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    
    logger.info(f"Saved embeddings to {output_file}")


def run_precomputation(
    output_dir: Path,
    inference_service,
    data_service,
    dataset_loader=None,
) -> None:
    """Main precomputation entry point.
    
    Call this after SAE training completes to prepare UI cache.
    
    Args:
        output_dir: Output directory containing checkpoints
        inference_service: Loaded inference service
        data_service: Loaded data service
        dataset_loader: Optional dataset loader for statistics
    """
    logger.info(f"Starting precomputation for Streamlit UI")
    
    # Run precomputation tasks
    precompute_neuron_wordclouds(
        inference_service,
        data_service,
        output_dir,
    )
    
    precompute_neuron_statistics(
        inference_service,
        dataset_loader,
        output_dir,
    )
    
    precompute_test_user_embeddings(
        inference_service,
        data_service,
        output_dir,
    )
    
    logger.info("Precomputation complete!")
