"""
Matrix-based neuron labeling using joint distribution approach.

Implements the methodology from semestral_project:
1. Build joint distribution matrix [tags × items]
2. Multiply against sparse activations [items × neurons] → [tags × neurons]
3. Apply TF-IDF on neuron documents to find distinctive neurons per tag
4. Normalize for final labels
"""

import torch
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy
from typing import Dict, List, Tuple, Any


def extract_tags_and_items(
    business_metadata: Dict[str, Dict[str, Any]],
    item_index_to_business_id: Dict[int, str],
) -> Dict[str, List[int]]:
    """
    Extract both categories and attributes as "tags" with their item indices.

    Parameters
    ----------
    business_metadata : Dict[str, Dict[str, Any]]
        Can be {item_idx: {name, categories, ...}, ...} OR
        {business_id: {name, categories, ...}, ...}
    item_index_to_business_id : Dict[int, str]
        Maps item index to business_id

    Returns
    -------
    Dict[str, List[int]]
        {tag_label: [item_idx, ...], ...}
        where tag_label is formatted as "category:X" or "attribute:Y=Z"
    """
    tag_items = defaultdict(set)

    # Check if business_metadata is keyed by item_idx (int) or business_id (str)
    first_key = next(iter(business_metadata.keys())) if business_metadata else None
    is_indexed_by_item = isinstance(first_key, int)

    if is_indexed_by_item:
        # Direct iteration: keys are already item indices
        for item_idx, metadata in business_metadata.items():
            if not isinstance(metadata, dict):
                continue

            # Extract categories
            if "categories" in metadata and metadata["categories"]:
                categories = metadata["categories"]
                if isinstance(categories, str):
                    categories = [c.strip() for c in categories.split(",")]
                elif isinstance(categories, list):
                    categories = [str(c).strip() for c in categories]
                else:
                    categories = []

                for cat in categories:
                    if cat:
                        tag_items[f"category:{cat}"].add(item_idx)

            # Extract attributes
            if "attributes" in metadata and isinstance(metadata["attributes"], dict):
                attrs = metadata["attributes"]
                for attr_key, attr_val in attrs.items():
                    if attr_val and str(attr_val).lower() not in (
                        "none",
                        "false",
                        "0",
                        "n/a",
                        "",
                    ):
                        tag_label = f"attribute:{attr_key}={attr_val}"
                        tag_items[tag_label].add(item_idx)
    else:
        # Keyed by business_id: need to map to item indices
        business_id_to_index = {
            bid: idx for idx, bid in item_index_to_business_id.items()
        }

        for business_id, metadata in business_metadata.items():
            if business_id not in business_id_to_index or not isinstance(
                metadata, dict
            ):
                continue

            item_idx = business_id_to_index[business_id]

            # Extract categories
            if "categories" in metadata and metadata["categories"]:
                categories = metadata["categories"]
                if isinstance(categories, str):
                    categories = [c.strip() for c in categories.split(",")]
                elif isinstance(categories, list):
                    categories = [str(c).strip() for c in categories]
                else:
                    categories = []

                for cat in categories:
                    if cat:
                        tag_items[f"category:{cat}"].add(item_idx)

            # Extract attributes
            if "attributes" in metadata and isinstance(metadata["attributes"], dict):
                attrs = metadata["attributes"]
                for attr_key, attr_val in attrs.items():
                    if attr_val and str(attr_val).lower() not in (
                        "none",
                        "false",
                        "0",
                        "n/a",
                        "",
                    ):
                        tag_label = f"attribute:{attr_key}={attr_val}"
                        tag_items[tag_label].add(item_idx)

    # Convert sets to lists
    tag_items_dict = {tag: list(items) for tag, items in tag_items.items()}

    return tag_items_dict


def build_joint_distribution_matrix(
    tag_items: Dict[str, List[int]],
    num_items: int,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Build joint distribution matrix [tags × items].

    Parameters
    ----------
    tag_items : Dict[str, List[int]]
        {tag: [item_idx, ...], ...}
    num_items : int
        Total number of items

    Returns
    -------
    Tuple[torch.Tensor, List[str]]
        (joint_distribution matrix [tags × items], tag_names)
    """
    tag_names = sorted(tag_items.keys())
    num_tags = len(tag_names)

    joint_distribution = torch.zeros(num_tags, num_items)

    for tag_idx, tag_name in enumerate(tag_names):
        items = tag_items[tag_name]
        for item_idx in items:
            if item_idx < num_items:
                joint_distribution[tag_idx, item_idx] = 1.0

    return joint_distribution, tag_names


def compute_tag_neuron_activations(
    joint_distribution: torch.Tensor,
    sparse_activations: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute tag-neuron activation matrix via matrix multiplication.

    joint_distribution: [tags × items]
    sparse_activations: [items × neurons]
    → tag_neuron_matrix: [tags × neurons]

    Parameters
    ----------
    joint_distribution : torch.Tensor
        [tags × items] - binary matrix
    sparse_activations : torch.Tensor
        [items × neurons] - sparse activations from SAE
    normalize : bool
        Whether to normalize by number of items per tag

    Returns
    -------
    torch.Tensor
        [tags × neurons] - activation scores
    """
    tag_neuron_matrix = torch.mm(joint_distribution.float(), sparse_activations.float())

    if normalize:
        # Normalize by number of items per tag to get average activation
        tag_counts = joint_distribution.sum(dim=1, keepdim=True)
        tag_neuron_matrix = tag_neuron_matrix / (tag_counts + 1e-8)

    return tag_neuron_matrix


def apply_tfidf_on_neurons(
    tag_neuron_matrix: torch.Tensor,
    tag_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Apply TF-IDF on neuron documents to find distinctive neurons per tag.

    Each tag is treated as a "document" and neurons are "words".

    Parameters
    ----------
    tag_neuron_matrix : torch.Tensor
        [tags × neurons] - activation scores
    tag_names : List[str]
        List of tag names

    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (label_neuron_tfidf [tags × neurons], tag_names)
    """
    num_tags, num_neurons = tag_neuron_matrix.shape

    # Step 1: Create documents where each neuron is a "word"
    # Scale activations to word counts (1-50 repetitions based on activation strength)
    label_documents = []
    for tag_idx in range(num_tags):
        neuron_activations = tag_neuron_matrix[tag_idx]
        doc_words = []

        for neuron_idx, activation in enumerate(neuron_activations):
            if activation > 1e-6:
                # Scale activation to word count
                word_count = max(1, min(50, int(np.sqrt(activation.item()) * 100)))
                doc_words.extend([f"n{neuron_idx}"] * word_count)

        label_documents.append(" ".join(doc_words) if doc_words else "empty")

    # Step 2: Apply TF-IDF
    tfidf = TfidfVectorizer(
        token_pattern=r"n\d+",
        max_features=num_neurons,
        lowercase=False,
        min_df=1,
        max_df=1.0,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True,
    )
    tfidf_matrix = tfidf.fit_transform(label_documents)
    feature_names = tfidf.get_feature_names_out()

    # Step 3: Convert back to tag-neuron matrix
    label_neuron_tfidf = np.zeros((num_tags, num_neurons))
    for tag_idx in range(num_tags):
        for feature_idx, tfidf_score in enumerate(
            tfidf_matrix[tag_idx].toarray().flatten()
        ):
            if tfidf_score > 0:
                feature_name = feature_names[feature_idx]
                neuron_idx = int(feature_name[1:])
                if neuron_idx < num_neurons:
                    label_neuron_tfidf[tag_idx, neuron_idx] = tfidf_score

    # Step 4: L2 normalize rows
    row_norms = np.linalg.norm(label_neuron_tfidf, axis=1, keepdims=True)
    label_neuron_tfidf = label_neuron_tfidf / (row_norms + 1e-8)

    return label_neuron_tfidf, tag_names


def label_neurons_from_tags(
    label_neuron_tfidf: np.ndarray,
    tag_names: List[str],
    neuron_profiles: Dict[int, Dict],
) -> Dict[int, str]:
    """
    Generate neuron labels by finding top tags per neuron.

    Parameters
    ----------
    label_neuron_tfidf : np.ndarray
        [tags × neurons] - TF-IDF scores
    tag_names : List[str]
        List of tag names corresponding to rows
    neuron_profiles : Dict[int, Dict]
        Neuron profiles (for filtering neurons with max_activating items)

    Returns
    -------
    Dict[int, str]
        {neuron_id: label_string, ...}
    """
    neuron_labels = {}

    # For each neuron, find top scoring tags
    for neuron_idx in range(label_neuron_tfidf.shape[1]):
        scores = label_neuron_tfidf[:, neuron_idx]
        top_indices = np.argsort(-scores)[:3]  # Top 3 tags

        top_tags = []
        for idx in top_indices:
            if scores[idx] > 1e-6:
                tag_name = tag_names[idx]
                # Remove prefix for cleaner label
                if tag_name.startswith("category:"):
                    tag_name = tag_name[9:]  # Remove "category:"
                elif tag_name.startswith("attribute:"):
                    tag_name = tag_name[10:]  # Remove "attribute:"

                top_tags.append(tag_name)

        # Build label from top tags
        if top_tags:
            label = " + ".join(top_tags[:2])  # Top 2 tags
        else:
            label = "unlabeled"

        neuron_labels[neuron_idx] = label

    return neuron_labels


def matrix_based_neuron_labeling(
    business_metadata: Dict[str, Dict[str, Any]],
    item_index_to_business_id: Dict[int, str],
    neuron_profiles: Dict[int, Dict],
    sparse_activations: torch.Tensor,
) -> Tuple[Dict[int, str], Dict[str, Any]]:
    """
    Complete matrix-based neuron labeling pipeline.

    Parameters
    ----------
    business_metadata : Dict[str, Dict[str, Any]]
        Business metadata with categories and attributes
    item_index_to_business_id : Dict[int, str]
        Item index to business ID mapping
    neuron_profiles : Dict[int, Dict]
        Neuron profiles from SAE
    sparse_activations : torch.Tensor
        [items × neurons] sparse activation matrix from SAE

    Returns
    -------
    Tuple[Dict[int, str], Dict[str, Any]]
        (neuron_labels, analysis_results)
    """
    num_items = sparse_activations.shape[0]
    num_neurons = sparse_activations.shape[1]

    # Step 1: Extract tags from categories and attributes
    print("\n1. Extracting tags from categories and attributes...")
    tag_items = extract_tags_and_items(business_metadata, item_index_to_business_id)
    print(f"   ✓ Extracted {len(tag_items)} unique tags")

    # Step 2: Build joint distribution matrix
    print("\n2. Building joint distribution matrix...")
    joint_distribution, tag_names = build_joint_distribution_matrix(
        tag_items, num_items
    )
    print(f"   ✓ Matrix shape: {joint_distribution.shape}")

    # Step 3: Compute tag-neuron activations
    print("\n3. Computing tag-neuron activation matrix...")
    tag_neuron_matrix = compute_tag_neuron_activations(
        joint_distribution, sparse_activations, normalize=True
    )
    print(f"   ✓ Activation matrix shape: {tag_neuron_matrix.shape}")

    # Step 4: Apply TF-IDF
    print("\n4. Applying TF-IDF on neuron documents...")
    label_neuron_tfidf, tag_names_sorted = apply_tfidf_on_neurons(
        tag_neuron_matrix, tag_names
    )
    print(f"   ✓ TF-IDF scoring complete")

    # Step 5: Generate neuron labels
    print("\n5. Generating neuron labels...")
    neuron_labels = label_neurons_from_tags(
        label_neuron_tfidf, tag_names_sorted, neuron_profiles
    )
    print(f"   ✓ Generated {len(neuron_labels)} neuron labels")

    # Analysis results
    analysis_results = {
        "num_tags": len(tag_names),
        "num_items": num_items,
        "num_neurons": num_neurons,
        "tag_names": tag_names,
        "joint_distribution": joint_distribution,
        "tag_neuron_matrix": tag_neuron_matrix,
        "label_neuron_tfidf": label_neuron_tfidf,
        "tag_items": tag_items,
    }

    return neuron_labels, analysis_results
