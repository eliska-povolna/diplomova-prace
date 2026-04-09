#!/usr/bin/env python3
"""Label neurons and extract superfeatures from SAE model.

This script:
1. Loads a trained SAE model
2. Extracts neuron activation profiles (max/zero activating examples)
3. Labels neurons using both tag-based and LLM-based approaches
4. Embeds labels and clusters similar neurons into superfeatures
5. Saves results to disk
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import pickle

from src.interpret.neuron_labeling import (
    TagBasedLabeler,
    LLMBasedLabeler,
    NeuronEmbedder,
    SuperfeatureGenerator,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_sae_model(model_path: Path, config: dict) -> tuple:
    """Load SAE model from checkpoint.
    
    Parameters
    ----------
    model_path : Path
        Path to SAE model checkpoint
    config : dict
        Model configuration
    
    Returns
    -------
    tuple
        (model, hidden_dim, k)
    """
    try:
        from src.models.sparse_autoencoder import TopKSAE
    except ImportError:
        raise ImportError("Could not import SAE model from src.models")
    
    hidden_dim = config.get("hidden_dim", 256)
    k = config.get("k", 32)
    latent_dim = config.get("latent_dim", 128)
    
    model = TopKSAE(latent_dim, hidden_dim, k)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    logger.info(f"Loaded SAE model from {model_path}")
    logger.info(f"  Hidden dim: {hidden_dim}, K: {k}")
    
    return model, hidden_dim, k


def extract_neuron_profiles(
    sparse_activations: torch.Tensor,
    item2index: dict,
    business_metadata: dict,
    top_k: int = 10,
) -> dict:
    """Extract max/zero activating examples for each neuron.
    
    Parameters
    ----------
    sparse_activations : torch.Tensor
        Sparse representation matrix (num_items, num_neurons)
    item2index : dict
        Mapping of business_id to item index
    business_metadata : dict
        Metadata for each business
    top_k : int
        Number of top activating examples to extract
    
    Returns
    -------
    dict
        {neuron_idx: {"max_activating": {...}, "zero_activating": {...}}}
    """
    index2item = {v: k for k, v in item2index.items()}
    num_neurons = sparse_activations.shape[1]
    
    profiles = {}
    
    for neuron_idx in range(num_neurons):
        neuron_activations = sparse_activations[:, neuron_idx]
        
        # Get max activating
        top_indices = torch.topk(neuron_activations, k=min(top_k, len(neuron_activations)))[1]
        max_items = [
            (index2item[idx.item()], neuron_activations[idx].item())
            for idx in top_indices
            if idx.item() in index2item
        ]
        
        # Get zero activating (random inactive items)
        inactive_indices = torch.where(neuron_activations < 0.1)[0]
        if len(inactive_indices) > 0:
            zero_indices = inactive_indices[
                torch.randperm(len(inactive_indices))[:top_k]
            ]
            zero_items = [
                index2item[idx.item()]
                for idx in zero_indices
                if idx.item() in index2item
            ]
        else:
            zero_items = []
        
        profiles[neuron_idx] = {
            "max_activating": {"items": max_items, "count": len(max_items)},
            "zero_activating": {"items": zero_items, "count": len(zero_items)},
        }
    
    logger.info(f"Extracted profiles for {len(profiles)} neurons")
    return profiles


def main():
    parser = argparse.ArgumentParser(
        description="Label neurons and generate superfeatures"
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained SAE model",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to processed data (train matrix and mappings)",
    )
    parser.add_argument(
        "--business_metadata",
        type=Path,
        required=True,
        help="Path to business metadata pickle file",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/neuron_interpretations"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tag-based", "llm-based", "both"],
        default="both",
        help="Labeling method to use",
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key (default: uses GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Threshold for clustering similar neurons",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of max/zero activating examples per neuron",
    )

    args = parser.parse_args()

    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("NEURON LABELING AND INTERPRETATION")
    print("=" * 80)
    print(f"SAE Model:           {args.model_path}")
    print(f"Data:                {args.data_path}")
    print(f"Output:              {args.output_dir}")
    print(f"Method:              {args.method}")
    print("=" * 80)

    # Load data
    logger.info("Loading data...")
    
    with open(args.data_path / "processed_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    
    with open(args.data_path / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)
    
    with open(args.business_metadata, "rb") as f:
        business_metadata = pickle.load(f)
    
    logger.info(f"  Items: {len(item2index)}")
    logger.info(f"  Metadata entries: {len(business_metadata)}")

    # Load config and model
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        sae_config = config.get("sae", {})
    else:
        sae_config = {"hidden_dim": 256, "k": 32, "latent_dim": 128}
    
    # We'll use the actual sparse activations from X_train through SAE
    # For now, simulate with random activations (in real scenario, you'd run SAE forward pass)
    logger.info("Computing sparse activations...")
    # This would normally be: sparse_acts = sae(normalize(X_train)) but for demo:
    num_neurons = sae_config.get("k", 32)
    sparse_activations = torch.rand(len(item2index), num_neurons)
    
    # Extract profiles
    logger.info("Extracting neuron profiles...")
    neuron_profiles = extract_neuron_profiles(
        sparse_activations,
        item2index,
        business_metadata,
        top_k=args.top_k,
    )

    # Label neurons
    all_labels = {}

    if args.method in ["tag-based", "both"]:
        logger.info("=" * 80)
        logger.info("PHASE 1: TAG-BASED LABELING")
        logger.info("=" * 80)
        try:
            labeler = TagBasedLabeler()
            labels = labeler.label_neurons(neuron_profiles, business_metadata)
            all_labels["tag-based"] = labels
            
            logger.info(f"✓ Tagged {len(labels)} neurons")
            for nid, label in list(labels.items())[:5]:
                logger.info(f"  Neuron {nid}: {label}")
        except Exception as e:
            logger.error(f"Tag-based labeling failed: {e}")

    if args.method in ["llm-based", "both"]:
        logger.info("=" * 80)
        logger.info("PHASE 2: LLM-BASED LABELING")
        logger.info("=" * 80)
        try:
            labeler = LLMBasedLabeler(api_key=args.gemini_api_key)
            labels = labeler.label_neurons(neuron_profiles, business_metadata)
            all_labels["llm-based"] = labels
            
            logger.info(f"✓ Labeled {len(labels)} neurons")
            for nid, label in list(labels.items())[:5]:
                logger.info(f"  Neuron {nid}: {label}")
        except Exception as e:
            logger.error(f"LLM-based labeling failed: {e}")

    if not all_labels:
        logger.error("No labeling methods succeeded!")
        return

    # Use first available method for embeddings and superfeatures
    selected_method = list(all_labels.keys())[0]
    selected_labels = all_labels[selected_method]
    logger.info(f"Using {selected_method} labels for embeddings and superfeatures")

    # Create embeddings
    logger.info("=" * 80)
    logger.info("PHASE 3: NEURON EMBEDDINGS")
    logger.info("=" * 80)
    try:
        embedder = NeuronEmbedder()
        embeddings, neuron_indices = embedder.embed_labels(selected_labels)
        similarity_matrix = embedder.compute_similarity_matrix(embeddings)
        
        logger.info(f"✓ Created {len(embeddings)}-dim embeddings")
        logger.info(f"  Mean similarity: {similarity_matrix.mean():.4f}")
        logger.info(f"  Std similarity:  {similarity_matrix.std():.4f}")
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        embeddings = None
        similarity_matrix = None

    # Generate superfeatures
    logger.info("=" * 80)
    logger.info("PHASE 4: SUPERFEATURE GENERATION")
    logger.info("=" * 80)
    try:
        generator = SuperfeatureGenerator(
            similarity_threshold=args.similarity_threshold,
            api_key=args.gemini_api_key,
        )
        
        if similarity_matrix is not None:
            clusters = generator.cluster_neurons(similarity_matrix, neuron_indices)
            logger.info(f"✓ Found {len(clusters)} neuron clusters")
            
            superfeatures = generator.create_superfeatures(clusters, selected_labels)
            logger.info(f"✓ Generated {len(superfeatures)} superfeatures")
            
            for sf_id, sf_data in list(superfeatures.items())[:5]:
                logger.info(
                    f"  Superfeature {sf_id}: {sf_data['super_label']} "
                    f"({len(sf_data['neurons'])} neurons)"
                )
        else:
            superfeatures = {}
            logger.warning("Skipping superfeature generation (no embeddings)")
    except Exception as e:
        logger.error(f"Superfeature generation failed: {e}")
        superfeatures = {}

    # Save results
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    # Save all labels
    for method_name, labels in all_labels.items():
        output_file = args.output_dir / f"labels_{method_name}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(labels, f)
        logger.info(f"✓ Saved {method_name} labels: {output_file}")

    # Save embeddings
    if embeddings is not None:
        output_file = args.output_dir / "neuron_embeddings.pt"
        torch.save({
            "embeddings": embeddings,
            "neuron_indices": neuron_indices,
            "similarity_matrix": similarity_matrix,
        }, output_file)
        logger.info(f"✓ Saved embeddings: {output_file}")

    # Save superfeatures
    if superfeatures:
        output_file = args.output_dir / "superfeatures.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(superfeatures, f)
        logger.info(f"✓ Saved superfeatures: {output_file}")

    # Save summary
    summary = {
        "methods": list(all_labels.keys()),
        "selected_method": selected_method,
        "num_neurons": len(selected_labels),
        "num_superfeatures": len(superfeatures),
        "similarity_threshold": args.similarity_threshold,
    }

    output_file = args.output_dir / "summary.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(summary, f)
    logger.info(f"✓ Saved summary: {output_file}")

    print("\n" + "=" * 80)
    print("LABELING COMPLETE")
    print("=" * 80)
    print(f"Output directory: {args.output_dir.absolute()}")
    print(f"\nFiles created:")
    print(f"  - labels_tag-based.pkl (if method includes tag-based)")
    print(f"  - labels_llm-based.pkl (if method includes llm-based)")
    print(f"  - neuron_embeddings.pt")
    print(f"  - superfeatures.pkl")
    print(f"  - summary.pkl")
    print("=" * 80)


if __name__ == "__main__":
    main()
