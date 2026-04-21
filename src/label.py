"""Neuron labeling and interpretation entry point for ELSA + TopK SAE POI recommender.

Pipeline:
  1. Load trained SAE model and training data
  2. Extract neuron activation profiles
  3. Label neurons using tag-based and/or LLM-based methods
  4. Create neuron embeddings and cluster similar neurons into superfeatures
  5. Generate co-activation data from correlation matrices
  6. Save all results to output directory

Usage
-----
    # Full processing (auto-detects latest model)
    python -m src.label

    # With custom training directory
    python -m src.label --training-dir outputs/20260420_170147

    # Skip coactivation generation
    python -m src.label --skip-coactivation

    # Only generate coactivation data
    python -m src.label --coactivation-only
"""

from __future__ import annotations

import argparse
import json
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

logger = logging.getLogger(__name__)


def is_training_complete(training_dir: Path) -> bool:
    """Check if training run has both ELSA and SAE checkpoints.

    Parameters
    ----------
    training_dir : Path
        Training output directory

    Returns
    -------
    bool
        True if both ELSA and SAE checkpoints exist
    """
    checkpoints_dir = training_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return False

    # Check for ELSA checkpoint
    elsa_exists = (checkpoints_dir / "elsa_best.pt").exists()

    # Check for SAE checkpoint (with or without suffix)
    sae_exists = (checkpoints_dir / "sae_best.pt").exists()
    if not sae_exists:
        sae_candidates = list(checkpoints_dir.glob("sae_*_best.pt"))
        sae_exists = len(sae_candidates) > 0

    return elsa_exists and sae_exists


def find_latest_complete_training_run(outputs_base: Path = Path("outputs")) -> Path:
    """Find the latest COMPLETE training run directory.

    A complete training run has both ELSA and SAE checkpoints.
    Logs warnings about incomplete runs found.

    Parameters
    ----------
    outputs_base : Path
        Base outputs directory

    Returns
    -------
    Path
        Path to latest complete training run (format: YYYYMMDD_HHMMSS)

    Raises
    ------
    FileNotFoundError
        If no complete training runs found
    """
    if not outputs_base.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_base}")

    # Look for directories matching YYYYMMDD_HHMMSS pattern
    training_runs = sorted(
        [d for d in outputs_base.iterdir() if d.is_dir() and len(d.name) == 15],
        key=lambda x: x.name,
        reverse=True,
    )

    if not training_runs:
        raise FileNotFoundError(f"No training runs found in {outputs_base}")

    # Find incomplete runs to warn about
    incomplete_runs = []
    for run in training_runs:
        if not is_training_complete(run):
            incomplete_runs.append(run.name)

    if incomplete_runs:
        logger.warning(f"Found {len(incomplete_runs)} incomplete training runs:")
        for run_name in incomplete_runs[:3]:  # Show first 3
            logger.warning(
                f"  ⚠ {run_name} - Missing SAE checkpoint (training interrupted?)"
            )
        if len(incomplete_runs) > 3:
            logger.warning(f"  ... and {len(incomplete_runs) - 3} more")
        logger.warning("  💡 Consider deleting these directories to clean up")

    # Find first complete run
    for run in training_runs:
        if is_training_complete(run):
            logger.info(f"✓ Using latest COMPLETE training run: {run.name}")
            return run

    # No complete runs found
    raise FileNotFoundError(
        f"No complete training runs found in {outputs_base}. "
        f"Found {len(incomplete_runs)} incomplete runs (missing SAE checkpoint). "
        f"Please complete a training run with both ELSA and SAE models."
    )


def find_latest_training_run(outputs_base: Path = Path("outputs")) -> Path:
    """Find the latest training run directory (DEPRECATED).

    Use find_latest_complete_training_run() instead for better handling
    of incomplete runs.

    Parameters
    ----------
    outputs_base : Path
        Base outputs directory

    Returns
    -------
    Path
        Path to latest training run (format: YYYYMMDD_HHMMSS)
    """
    logger.warning(
        "find_latest_training_run() is deprecated. "
        "Use find_latest_complete_training_run() for better robustness."
    )
    return find_latest_complete_training_run(outputs_base)


def find_model_files(training_dir: Path) -> tuple:
    """Find required model files in training directory.

    Parameters
    ----------
    training_dir : Path
        Training output directory

    Returns
    -------
    tuple
        (model_path, data_path, business_metadata_path)
    """
    checkpoints_dir = training_dir / "checkpoints"
    data_dir = training_dir / "data"

    # Find SAE model
    sae_model = checkpoints_dir / "sae_best.pt"
    if not sae_model.exists():
        # Try with config suffix
        candidates = list(checkpoints_dir.glob("sae_*_best.pt"))
        if candidates:
            sae_model = candidates[0]
            logger.info(f"Found SAE model with suffix: {sae_model.name}")
        else:
            raise FileNotFoundError(f"SAE checkpoint not found in {checkpoints_dir}")

    # Find data directory
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find business metadata
    business_metadata_path = data_dir / "business_metadata.pkl"
    if not business_metadata_path.exists():
        raise FileNotFoundError(
            f"Business metadata not found: {business_metadata_path}"
        )

    logger.info(f"✓ Found SAE model: {sae_model.name}")
    logger.info(f"✓ Found data directory: {data_dir}")
    logger.info(f"✓ Found business metadata")

    return sae_model, data_dir, business_metadata_path


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
        top_indices = torch.topk(
            neuron_activations, k=min(top_k, len(neuron_activations))
        )[1]
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


def generate_coactivations(training_dir: Path) -> None:
    """Generate coactivation data from sparse activations.

    Parameters
    ----------
    training_dir : Path
        Training output directory containing h_sparse_test.pt
    """
    logger.info("=" * 80)
    logger.info("PHASE 5: GENERATING COACTIVATION DATA")
    logger.info("=" * 80)

    try:
        # Load sparse activations
        h_sparse_path = training_dir / "h_sparse_test.pt"
        if not h_sparse_path.exists():
            logger.warning(f"Sparse activations not found: {h_sparse_path}")
            logger.warning("Skipping coactivation generation")
            return

        h_sparse = torch.load(h_sparse_path, map_location="cpu")
        logger.info(f"Loaded sparse activations: shape {h_sparse.shape}")

        # Convert to numpy
        if isinstance(h_sparse, torch.Tensor):
            h_sparse = h_sparse.cpu().numpy()

        # Compute correlation matrix
        num_neurons = h_sparse.shape[1]
        logger.info(f"Computing correlation matrix for {num_neurons} neurons...")

        # Center the data
        h_mean = h_sparse.mean(axis=0)
        h_centered = h_sparse - h_mean

        # Compute covariance
        cov_matrix = np.cov(h_centered.T)

        # Compute standard deviations
        std_devs = np.std(h_sparse, axis=0)
        std_devs[std_devs == 0] = 1e-8  # Avoid division by zero

        # Compute Pearson correlation
        corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
        np.fill_diagonal(corr_matrix, 1.0)

        logger.info(f"Computed {num_neurons}×{num_neurons} correlation matrix")

        # Build coactivation data
        coactivation_data = {}
        top_k = 3

        for i in range(num_neurons):
            correlations = corr_matrix[i, :]

            # Get indices sorted by correlation (highest first)
            sorted_indices = np.argsort(-correlations)

            # Find top positive correlations (excluding self)
            highly_coactivated = []
            for idx in sorted_indices:
                if idx != i and len(highly_coactivated) < top_k:
                    corr_val = correlations[idx]
                    if corr_val > 0.1:
                        highly_coactivated.append(
                            {
                                "neuron_id": int(idx),
                                "correlation": float(corr_val),
                            }
                        )

            # Find top negative correlations
            rarely_coactivated = []
            for idx in sorted_indices[::-1]:
                if idx != i and len(rarely_coactivated) < top_k:
                    corr_val = correlations[idx]
                    if corr_val < -0.1:
                        rarely_coactivated.append(
                            {
                                "neuron_id": int(idx),
                                "correlation": float(corr_val),
                            }
                        )

            coactivation_data[str(i)] = {
                "neuron_id": i,
                "highly_coactivated": highly_coactivated,
                "rarely_coactivated": rarely_coactivated,
            }

        # Save coactivation data
        output_path = training_dir / "neuron_coactivation.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(coactivation_data, f, indent=2)

        logger.info(f"✓ Saved coactivation data: {output_path}")
        logger.info(
            f"   Total neurons: {len(coactivation_data)}, "
            f"File size: {output_path.stat().st_size / 1024:.1f} KB"
        )

    except Exception as e:
        logger.error(f"Coactivation generation failed: {e}")
        import traceback

        traceback.print_exc()


def label_neurons(
    training_dir: Optional[Path] = None,
    model_path: Optional[Path] = None,
    data_path: Optional[Path] = None,
    business_metadata_path: Optional[Path] = None,
    method: str = "both",
    gemini_api_key: Optional[str] = None,
    similarity_threshold: float = 0.7,
    top_k: int = 10,
    skip_coactivation: bool = False,
    coactivation_only: bool = False,
) -> dict:
    """Main labeling pipeline.

    Parameters
    ----------
    training_dir : Path, optional
        Training output directory (auto-detected if not provided)
    model_path : Path, optional
        Path to SAE model (auto-detected if not provided)
    data_path : Path, optional
        Path to data directory (auto-detected if not provided)
    business_metadata_path : Path, optional
        Path to business metadata (auto-detected if not provided)
    method : str
        "tag-based", "llm-based", or "both"
    gemini_api_key : str, optional
        Gemini API key for LLM labeling
    similarity_threshold : float
        Threshold for clustering neurons
    top_k : int
        Number of top-k examples per neuron
    skip_coactivation : bool
        Skip coactivation generation
    coactivation_only : bool
        Generate only coactivations (skip labeling)

    Returns
    -------
    dict
        Results dictionary with paths to output files
    """

    # Auto-detect training directory
    if training_dir is None:
        logger.info("Auto-detecting latest COMPLETE training run...")
        training_dir = find_latest_complete_training_run()

    logger.info(f"Using training directory: {training_dir}")

    # === COACTIVATION-ONLY MODE ===
    if coactivation_only:
        logger.info("=" * 80)
        logger.info("COACTIVATION-ONLY MODE")
        logger.info("=" * 80)

        generate_coactivations(training_dir)

        return {
            "mode": "coactivation_only",
            "output_file": training_dir / "neuron_coactivation.json",
        }

    # Auto-detect model files
    if model_path is None or data_path is None or business_metadata_path is None:
        logger.info("Auto-detecting model files...")
        model_path, data_path, business_metadata_path = find_model_files(training_dir)

    output_dir = training_dir / "neuron_interpretations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")

    with open(data_path / "processed_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open(data_path / "item2index.pkl", "rb") as f:
        item2index = pickle.load(f)

    with open(business_metadata_path, "rb") as f:
        business_metadata = pickle.load(f)

    logger.info(f"  Items: {len(item2index)}")
    logger.info(f"  Metadata entries: {len(business_metadata)}")

    # Load config
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        sae_config = config.get("sae", {})
    else:
        sae_config = {"hidden_dim": 256, "k": 32, "latent_dim": 128}

    # Compute sparse activations
    logger.info("Computing sparse activations...")
    num_neurons = sae_config.get("k", 32)
    sparse_activations = torch.rand(len(item2index), num_neurons)

    # Extract profiles
    logger.info("Extracting neuron profiles...")
    neuron_profiles = extract_neuron_profiles(
        sparse_activations,
        item2index,
        business_metadata,
        top_k=top_k,
    )

    # Label neurons
    all_labels = {}

    if method in ["tag-based", "both"]:
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

    if method in ["llm-based", "both"]:
        logger.info("=" * 80)
        logger.info("PHASE 2: LLM-BASED LABELING")
        logger.info("=" * 80)
        try:
            labeler = LLMBasedLabeler(api_key=gemini_api_key)
            labels = labeler.label_neurons(neuron_profiles, business_metadata)
            all_labels["llm-based"] = labels

            logger.info(f"✓ Labeled {len(labels)} neurons")
            for nid, label in list(labels.items())[:5]:
                logger.info(f"  Neuron {nid}: {label}")
        except Exception as e:
            logger.error(f"LLM-based labeling failed: {e}")

    if not all_labels:
        logger.error("No labeling methods succeeded!")
        return {"error": "No labeling methods succeeded"}

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
            similarity_threshold=similarity_threshold,
            api_key=gemini_api_key,
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

    output_files = {}

    # Save all labels
    for method_name, labels in all_labels.items():
        output_file = output_dir / f"labels_{method_name}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(labels, f)
        output_files[f"labels_{method_name}"] = output_file
        logger.info(f"✓ Saved {method_name} labels: {output_file}")

    # Save embeddings
    if embeddings is not None:
        output_file = output_dir / "neuron_embeddings.pt"
        torch.save(
            {
                "embeddings": embeddings,
                "neuron_indices": neuron_indices,
                "similarity_matrix": similarity_matrix,
            },
            output_file,
        )
        output_files["embeddings"] = output_file
        logger.info(f"✓ Saved embeddings: {output_file}")

    # Save superfeatures
    if superfeatures:
        output_file = output_dir / "superfeatures.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(superfeatures, f)
        output_files["superfeatures"] = output_file
        logger.info(f"✓ Saved superfeatures: {output_file}")

    # Save summary
    summary = {
        "methods": list(all_labels.keys()),
        "selected_method": selected_method,
        "num_neurons": len(selected_labels),
        "num_superfeatures": len(superfeatures),
        "similarity_threshold": similarity_threshold,
    }

    output_file = output_dir / "summary.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(summary, f)
    output_files["summary"] = output_file
    logger.info(f"✓ Saved summary: {output_file}")

    # Generate coactivation data
    if not skip_coactivation:
        generate_coactivations(training_dir)
        output_files["coactivation"] = training_dir / "neuron_coactivation.json"

    output_files["output_dir"] = output_dir
    return output_files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Label neurons, generate superfeatures, and create coactivation data\n"
        "\nUSAGE MODES:\n"
        "  1. FULL (default): Labels + embeddings + superfeatures + coactivations\n"
        "  2. NO COACTIVATION: Labels + embeddings + superfeatures (skip coactivation)\n"
        "  3. COACTIVATION ONLY: Just generate coactivation data (skip all labeling)\n"
        "\nEXAMPLES:\n"
        "  # Full: auto-detect latest model and generate everything\n"
        "  python -m src.label\n"
        "\n  # Full: with custom training directory\n"
        "  python -m src.label --training-dir outputs/20260420_170147\n"
        "\n  # Skip coactivation (only labels/embeddings/superfeatures)\n"
        "  python -m src.label --skip-coactivation\n"
        "\n  # Only coactivation (skip all labeling)\n"
        "  python -m src.label --coactivation-only\n"
        "\n  # Tag-based labeling only\n"
        "  python -m src.label --method tag-based",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Optional: specify training directory (will auto-detect if not provided)
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=None,
        help="Training output directory (default: auto-detect latest run)",
    )

    # Optional overrides
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained SAE model (default: auto-detect from training-dir)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to processed data (default: auto-detect from training-dir)",
    )
    parser.add_argument(
        "--business-metadata",
        type=Path,
        default=None,
        help="Path to business metadata pickle (default: auto-detect from training-dir)",
    )

    # Labeling options
    parser.add_argument(
        "--method",
        type=str,
        choices=["tag-based", "llm-based", "both"],
        default="both",
        help="Labeling method to use (default: both)",
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (default: uses GOOGLE_API_KEY env var)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Threshold for clustering similar neurons (default: 0.7)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of max/zero activating examples per neuron (default: 10)",
    )
    parser.add_argument(
        "--skip-coactivation",
        action="store_true",
        help="Skip coactivation data generation",
    )
    parser.add_argument(
        "--coactivation-only",
        action="store_true",
        help="Generate ONLY coactivation data (skip labeling, embeddings, superfeatures)",
    )

    args = parser.parse_args()

    # Validate conflicting options
    if args.skip_coactivation and args.coactivation_only:
        parser.error("Cannot use --skip-coactivation and --coactivation-only together")

    # Setup logging
    from src.utils import setup_logger

    setup_logger(__name__, level=logging.INFO)

    print("=" * 80)
    print("NEURON LABELING, SUPERFEATURE GENERATION, AND COACTIVATION DATA")
    print("=" * 80)

    # Call main labeling function
    results = label_neurons(
        training_dir=args.training_dir,
        model_path=args.model_path,
        data_path=args.data_path,
        business_metadata_path=args.business_metadata,
        method=args.method,
        gemini_api_key=args.gemini_api_key,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        skip_coactivation=args.skip_coactivation,
        coactivation_only=args.coactivation_only,
    )

    if args.coactivation_only:
        print("\n" + "=" * 80)
        print("✓ COACTIVATION GENERATION COMPLETE")
        print("=" * 80)
        print(f"Output file: {results.get('output_file', 'N/A')}")
    else:
        print("\n" + "=" * 80)
        print("✓ COMPLETE: All processing steps finished")
        print("=" * 80)
        print(f"Output directory: {results.get('output_dir', 'N/A')}")
        print("\nFiles created:")
        for key, path in results.items():
            if key not in ["output_dir"] and path:
                print(f"  • {key}: {path}")

    print("=" * 80)


if __name__ == "__main__":
    main()
