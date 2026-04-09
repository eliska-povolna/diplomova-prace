"""Extract and analyze neuron activations from trained models.

This module computes which businesses/items most strongly activate each neuron
in the sparse autoencoder, and vice versa.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def extract_sparse_activations(
    embeddings: torch.Tensor,
    sae: torch.nn.Module,
    batch_size: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Extract sparse hidden activations for all items.

    Parameters
    ----------
    embeddings : torch.Tensor
        Dense embeddings from ELSA model (num_items, latent_dim)
    sae : torch.nn.Module
        Trained sparse autoencoder
    batch_size : int
        Batch size for processing
    device : str
        Device to use for computation

    Returns
    -------
    torch.Tensor
        Sparse hidden activations (num_items, num_neurons)
    """
    sae.to(device)
    sae.eval()

    embeddings = embeddings.to(device)
    num_items = embeddings.shape[0]

    h_list = []
    with torch.no_grad():
        for start in range(0, num_items, batch_size):
            end = min(start + batch_size, num_items)
            batch_emb = embeddings[start:end]

            # Normalize embeddings
            batch_emb_norm = torch.nn.functional.normalize(batch_emb, dim=1)

            # Get sparse activations
            _, h_sparse_batch, _ = sae(batch_emb_norm)
            h_list.append(h_sparse_batch.cpu())

    h_sparse = torch.cat(h_list, dim=0)
    return h_sparse


def get_max_activating_items(
    h_sparse: torch.Tensor,
    item2index: dict,
    num_examples: int = 10,
) -> dict:
    """Get items with highest activation for each neuron.

    Parameters
    ----------
    h_sparse : torch.Tensor
        Sparse activations (num_items, num_neurons)
    item2index : dict
        Mapping from item_id to index
    num_examples : int
        Number of top examples per neuron

    Returns
    -------
    dict
        {neuron_idx: [(item_id, activation), ...]}
    """
    index2item = {v: k for k, v in item2index.items()}
    num_neurons = h_sparse.shape[1]

    max_activating = {}
    for neuron_idx in range(num_neurons):
        activations = h_sparse[:, neuron_idx].numpy()

        # Get top activations
        top_indices = np.argsort(-activations)[:num_examples]
        top_items = [
            (index2item[idx], activations[idx]) for idx in top_indices
        ]

        max_activating[neuron_idx] = top_items

    return max_activating


def get_zero_activating_items(
    h_sparse: torch.Tensor,
    item2index: dict,
    num_examples: int = 10,
) -> dict:
    """Get items with zero or minimal activation for each neuron.

    Parameters
    ----------
    h_sparse : torch.Tensor
        Sparse activations (num_items, num_neurons)
    item2index : dict
        Mapping from item_id to index
    num_examples : int
        Number of examples per neuron

    Returns
    -------
    dict
        {neuron_idx: [item_id, ...]}
    """
    index2item = {v: k for k, v in item2index.items()}
    num_neurons = h_sparse.shape[1]

    zero_activating = {}
    for neuron_idx in range(num_neurons):
        activations = h_sparse[:, neuron_idx].numpy()

        # Get items with zero/minimal activation (sort ascending)
        zero_indices = np.argsort(activations)[:num_examples]
        zero_items = [index2item[idx] for idx in zero_indices]

        zero_activating[neuron_idx] = zero_items

    return zero_activating


def collect_business_metadata(
    item_ids: list[str],
    businesses_df,
) -> dict:
    """Collect metadata (category, tags, reviews) for items.

    Parameters
    ----------
    item_ids : list[str]
        List of business IDs
    businesses_df : pd.DataFrame
        DataFrame with business metadata (must contain business_id, categories, etc.)

    Returns
    -------
    dict
        {business_id: {"categories": [...], "attributes": {...}, ...}}
    """
    metadata = {}

    for bid in item_ids:
        # Find business in dataframe
        business_data = businesses_df[businesses_df["business_id"] == bid]

        if len(business_data) == 0:
            metadata[bid] = {
                "categories": [],
                "name": "Unknown",
                "city": "Unknown",
            }
            continue

        row = business_data.iloc[0]
        categories = []
        if "categories" in row and pd.notna(row["categories"]):
            categories = [c.strip() for c in str(row["categories"]).split(",")]

        metadata[bid] = {
            "name": row.get("name", "Unknown"),
            "city": row.get("city", "Unknown"),
            "categories": categories,
            "stars": row.get("stars", None),
            "review_count": row.get("review_count", None),
        }

    return metadata


def build_neuron_profile(
    neuron_idx: int,
    max_activating_items: list[tuple[str, float]],
    zero_activating_items: list[str],
    business_metadata: dict,
) -> dict:
    """Build a profile of a single neuron based on its activations.

    Parameters
    ----------
    neuron_idx : int
        Neuron index
    max_activating_items : list[tuple[str, float]]
        [(business_id, activation_value), ...]
    zero_activating_items : list[str]
        [business_id, ...]
    business_metadata : dict
        {business_id: metadata_dict}

    Returns
    -------
    dict
        Complete neuron profile
    """
    max_categories = []
    max_names = []

    for bid, _ in max_activating_items:
        meta = business_metadata.get(bid, {})
        max_categories.extend(meta.get("categories", []))
        max_names.append(meta.get("name", "Unknown"))

    zero_categories = []
    for bid in zero_activating_items:
        meta = business_metadata.get(bid, {})
        zero_categories.extend(meta.get("categories", []))

    return {
        "neuron_idx": neuron_idx,
        "max_activating": {
            "items": max_activating_items,
            "top_names": max_names[:5],
            "categories": max_categories,
        },
        "zero_activating": {
            "items": zero_activating_items,
            "categories": zero_categories,
        },
    }


if __name__ == "__main__":
    import pandas as pd

    print("Activation extraction module loaded")
