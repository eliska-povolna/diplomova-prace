"""Neuron labeling and interpretation entry point for ELSA + TopK SAE POI recommender.

Pipeline:
  1. Load trained SAE model and training data
  2. Extract neuron activation profiles
    3. Label neurons using tag-based, matrix-based, and/or LLM-based methods
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

    # Run all non-LLM labeling options
    python -m src.label --method non-llm
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
from src.interpret.matrix_based_labeling import matrix_based_neuron_labeling
from src.run_registry import write_latest_run_pointer

logger = logging.getLogger(__name__)


def _load_business_metadata_fallback(data_dir: Path) -> dict:
    """Rebuild business metadata when the saved pickle is missing.

    The training pipeline only writes business_metadata.pkl when the metadata
    object is available in memory. Older runs or alternative training paths may
    not have that file, but they still usually have the DB and mapping files.
    """
    item_map_path = data_dir / "item_map_after_kcore.pkl"
    reviews_path = data_dir / "reviews_df.pkl"

    if not item_map_path.exists() and not reviews_path.exists():
        raise FileNotFoundError(
            f"Could not rebuild business metadata: missing {item_map_path.name} and {reviews_path.name}"
        )

    item_ids = None
    if item_map_path.exists():
        with open(item_map_path, "rb") as f:
            item_map = pickle.load(f)
        item_ids = list(item_map.keys())
        logger.info(
            f"Rebuilding business metadata from {item_map_path.name} ({len(item_ids)} items)"
        )
    else:
        with open(reviews_path, "rb") as f:
            reviews_df = pickle.load(f)
        item_ids = sorted(set(reviews_df["business_id"].dropna().astype(str)))
        logger.info(
            f"Rebuilding business metadata from {reviews_path.name} ({len(item_ids)} businesses)"
        )

    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        raise FileNotFoundError(
            "configs/default.yaml not found, cannot resolve database path for metadata fallback"
        )

    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    data_config = config.get("data", {})
    db_path = Path(data_config.get("db_path", "yelp.duckdb"))
    if not db_path.is_absolute():
        db_path = (Path(__file__).resolve().parent.parent / db_path).resolve()

    from src.data.yelp_loader import load_businesses

    businesses_df = load_businesses(
        db_path=db_path,
        state_filter=data_config.get("state_filter"),
        min_review_count=0,
    )

    businesses_df = businesses_df[
        businesses_df["business_id"].astype(str).isin(item_ids)
    ]

    business_metadata = {}
    for _, row in businesses_df.iterrows():
        categories = []
        if "categories" in row and row["categories"] is not None:
            categories = [
                c.strip() for c in str(row["categories"]).split(",") if c.strip()
            ]

        business_metadata[str(row["business_id"])] = {
            "name": row.get("name", "Unknown"),
            "city": row.get("city", "Unknown"),
            "state": row.get("state", "Unknown"),
            "categories": categories,
            "stars": row.get("stars", None),
            "review_count": row.get("review_count", None),
        }

    if not business_metadata:
        raise RuntimeError(
            "Failed to reconstruct business metadata from DuckDB business table"
        )

    logger.info(
        f"✓ Reconstructed business metadata for {len(business_metadata)} businesses from database"
    )
    return business_metadata


def _load_experiment_manifest(experiment_dir: Path) -> dict:
    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Experiment manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if not isinstance(manifest, dict):
        raise ValueError(f"Invalid experiment manifest: {manifest_path}")

    return manifest


def _find_latest_experiment_dir() -> Optional[Path]:
    pointer_path = Path("outputs") / "LATEST_EXPERIMENT.txt"
    if pointer_path.exists():
        try:
            candidate = Path(pointer_path.read_text(encoding="utf-8").strip())
            if candidate.exists() and (candidate / "manifest.json").exists():
                return candidate
        except Exception as e:
            logger.warning(f"Failed to read LATEST_EXPERIMENT.txt: {e}")

    experiments_base = Path("outputs") / "experiments"
    if not experiments_base.exists():
        return None

    experiment_dirs = [
        d
        for d in experiments_base.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ]
    if not experiment_dirs:
        return None

    return sorted(experiment_dirs, key=lambda d: d.name)[-1]


def _label_experiment_runs(
    experiment_dir: Path,
    *,
    method: str,
    gemini_api_key: Optional[str],
    similarity_threshold: float,
    top_k: int,
    skip_coactivation: bool,
    coactivation_only: bool,
) -> dict:
    manifest = _load_experiment_manifest(experiment_dir)
    runs = manifest.get("runs", [])
    if not runs:
        raise ValueError(f"No runs found in experiment manifest: {experiment_dir}")

    logger.info("EXPERIMENT BATCH LABELING MODE")
    logger.info("=" * 80)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Run count: {len(runs)}")
    logger.info("=" * 80)

    batch_results = []
    for idx, run in enumerate(runs, start=1):
        run_dir = Path(run.get("run_dir", ""))
        if not run_dir.exists():
            logger.warning(f"Skipping missing run directory: {run_dir}")
            continue

        logger.info(f"[{idx}/{len(runs)}] Labeling run: {run_dir.name}")
        result = label_neurons(
            training_dir=run_dir,
            method=method,
            gemini_api_key=gemini_api_key,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            skip_coactivation=skip_coactivation,
            coactivation_only=coactivation_only,
        )

        batch_results.append(
            {
                "run_name": run.get("run_name"),
                "run_dir": str(run_dir),
                "result": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in result.items()
                },
            }
        )

    label_manifest = {
        "experiment_id": manifest.get("experiment_id"),
        "experiment_dir": str(experiment_dir),
        "created": manifest.get("created"),
        "source_config": manifest.get("source_config"),
        "base_config": manifest.get("base_config"),
        "method": method,
        "skip_coactivation": skip_coactivation,
        "coactivation_only": coactivation_only,
        "runs": batch_results,
    }

    label_manifest_path = experiment_dir / "labeling_manifest.json"
    with label_manifest_path.open("w", encoding="utf-8") as f:
        json.dump(label_manifest, f, indent=2)

    logger.info(f"✓ Saved batch labeling manifest: {label_manifest_path}")
    return {
        "mode": "experiment_batch",
        "experiment_dir": str(experiment_dir),
        "labeling_manifest": label_manifest_path,
        "runs_labeled": len(batch_results),
    }


def _normalize_categories(categories) -> list[str]:
    if not categories:
        return []

    if isinstance(categories, str):
        categories = [part.strip() for part in categories.split(",")]

    if not isinstance(categories, (list, tuple, set)):
        return []

    normalized = []
    for category in categories:
        category_text = str(category).strip()
        if category_text:
            normalized.append(category_text)
    return normalized


def _build_neuron_category_metadata(
    neuron_profiles: dict,
    business_metadata: dict,
    top_k: int = 10,
) -> dict:
    """Build per-neuron category metadata for the interpretability UI."""

    metadata = {}

    for neuron_idx, profile in neuron_profiles.items():
        max_items = profile.get("max_activating", {}).get("items", [])[:top_k]
        category_weights: dict[str, list[float]] = {}
        top_items = []
        activation_values = []

        for business_id, activation in max_items:
            business_key = str(business_id)
            business_info = business_metadata.get(business_key, {})
            categories = _normalize_categories(business_info.get("categories", []))

            activation_value = float(activation)
            activation_values.append(activation_value)

            top_items.append(
                {
                    "business_id": business_key,
                    "item_id": business_key,
                    "name": str(business_info.get("name", "Unknown")),
                    "activation": activation_value,
                    "categories": categories,
                    "city": str(business_info.get("city", "Unknown")),
                    "state": str(business_info.get("state", "Unknown")),
                    "stars": (
                        float(business_info["stars"])
                        if business_info.get("stars") is not None
                        else None
                    ),
                    "review_count": (
                        int(business_info["review_count"])
                        if business_info.get("review_count") is not None
                        else None
                    ),
                }
            )

            for category in categories:
                category_weights.setdefault(category, []).append(activation_value)

        metadata[str(neuron_idx)] = {
            "neuron_id": int(neuron_idx),
            "top_items": top_items,
            "category_weights": category_weights,
            "top_categories": sorted(
                category_weights.keys(),
                key=lambda category: len(category_weights[category]),
                reverse=True,
            ),
            "max_activation": max(activation_values) if activation_values else 0.0,
            "mean_activation": (
                sum(activation_values) / len(activation_values)
                if activation_values
                else 0.0
            ),
            "num_examples": len(top_items),
        }

    return metadata


def _build_precomputed_wordcloud_payloads(category_metadata: dict) -> dict[str, dict]:
    """Build simple wordcloud payloads for the optional UI cache."""
    payloads = {}

    for neuron_id, metadata in category_metadata.items():
        frequencies: dict[str, int] = {}

        for category, activations in metadata.get("category_weights", {}).items():
            if not activations:
                continue

            weight = max(
                1,
                int(len(activations) * (sum(activations) / len(activations)) * 100),
            )
            frequencies[category] = weight

        if not frequencies:
            continue

        words = []
        for word, count in frequencies.items():
            words.extend([word] * count)

        payloads[str(neuron_id)] = {
            "text": " ".join(words),
            "freq": list(frequencies.values()),
            "words": list(frequencies.keys()),
            "word_count": len(frequencies),
        }

    return payloads


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
        logger.warning(
            f"Business metadata not found at {business_metadata_path}; will rebuild from database if needed"
        )

    logger.info(f"✓ Found SAE model: {sae_model.name}")
    logger.info(f"✓ Found data directory: {data_dir}")
    if business_metadata_path.exists():
        logger.info("✓ Found business metadata")

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

    elsa_config = config.get("elsa", {})
    sae_config = config.get("sae", {})

    latent_dim = config.get("latent_dim", elsa_config.get("latent_dim", 128))
    hidden_dim = config.get(
        "hidden_dim",
        sae_config.get("hidden_dim") or (sae_config.get("width_ratio", 2) * latent_dim),
    )
    k = config.get("k", sae_config.get("k", 32))

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


def _load_neuron_labels_for_coactivation(training_dir: Path) -> dict[int, str]:
    """Load the selected neuron labels used to annotate coactivation outputs."""
    labels_path = training_dir / "neuron_interpretations" / "neuron_labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Missing neuron labels artifact required for coactivation generation: {labels_path}"
        )

    with labels_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    labels = payload.get("neuron_labels")
    if not isinstance(labels, dict) or not labels:
        raise RuntimeError(
            f"Invalid neuron labels artifact for coactivation generation: {labels_path}"
        )

    return {int(neuron_id): str(label) for neuron_id, label in labels.items()}


def _require_neuron_label(neuron_labels: dict[int, str], neuron_id: int) -> str:
    label = neuron_labels.get(int(neuron_id))
    if not label:
        raise KeyError(f"Missing label for neuron {neuron_id} in coactivation data")
    return label


def generate_coactivations(training_dir: Path, neuron_labels: dict[int, str]) -> None:
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
                                "label": _require_neuron_label(neuron_labels, int(idx)),
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
                                "label": _require_neuron_label(neuron_labels, int(idx)),
                                "correlation": float(corr_val),
                            }
                        )

            coactivation_data[str(i)] = {
                "neuron_id": i,
                "label": _require_neuron_label(neuron_labels, i),
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

        neuron_labels = _load_neuron_labels_for_coactivation(training_dir)
        generate_coactivations(training_dir, neuron_labels)

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

    item2index_path = data_path / "item2index.pkl"
    if item2index_path.exists():
        with open(item2index_path, "rb") as f:
            item2index = pickle.load(f)
        logger.info(f"  Loaded item2index from {item2index_path.name}")
    else:
        fallback_item_map = data_path / "item_map_after_kcore.pkl"
        if not fallback_item_map.exists():
            raise FileNotFoundError(
                f"Neither {item2index_path.name} nor {fallback_item_map.name} was found in {data_path}"
            )
        with open(fallback_item_map, "rb") as f:
            item2index = pickle.load(f)
        logger.info(
            f"  Loaded item2index from fallback mapping {fallback_item_map.name}"
        )

    if business_metadata_path is not None and business_metadata_path.exists():
        with open(business_metadata_path, "rb") as f:
            business_metadata = pickle.load(f)
    else:
        business_metadata = _load_business_metadata_fallback(data_path)

    logger.info(f"  Items: {len(item2index)}")
    logger.info(f"  Metadata entries: {len(business_metadata)}")

    # Load config
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)
        sae_config = config.get("sae", {})
        elsa_config = config.get("elsa", {})
    else:
        sae_config = {"width_ratio": 2, "k": 32}
        elsa_config = {"latent_dim": 128}

    # Compute sparse activations
    logger.info("Computing sparse activations...")
    num_neurons = sae_config.get("hidden_dim") or (
        sae_config.get("width_ratio", 2) * elsa_config.get("latent_dim", 128)
    )
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

    if method in ["tag-based", "non-llm", "both"]:
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

    if method in ["matrix-based", "non-llm"]:
        logger.info("=" * 80)
        logger.info("PHASE 2: MATRIX-BASED LABELING")
        logger.info("=" * 80)
        try:
            index_to_business_id = {
                index: business_id for business_id, index in item2index.items()
            }
            labels, analysis_results = matrix_based_neuron_labeling(
                business_metadata=business_metadata,
                item_index_to_business_id=index_to_business_id,
                neuron_profiles=neuron_profiles,
                sparse_activations=sparse_activations,
            )
            all_labels["matrix-based"] = labels

            logger.info(f"✓ Labeled {len(labels)} neurons")
            logger.info(f"  Tags extracted: {analysis_results.get('num_tags', 'N/A')}")
            for nid, label in list(labels.items())[:5]:
                logger.info(f"  Neuron {nid}: {label}")
        except Exception as e:
            logger.error(f"Matrix-based labeling failed: {e}")

    if method in ["llm-based", "both"]:
        logger.info("=" * 80)
        logger.info("PHASE 3: LLM-BASED LABELING")
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

    labels_json_path = output_dir / "neuron_labels.json"
    with labels_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_method": selected_method,
                "neuron_labels": selected_labels,
                "methods": all_labels,
            },
            f,
            indent=2,
        )
    output_files["labels_json"] = labels_json_path
    logger.info(f"✓ Saved labels JSON: {labels_json_path}")

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

    # Save interpretability metadata for the Streamlit UI
    logger.info("Saving interpretability metadata...")
    category_metadata = _build_neuron_category_metadata(
        neuron_profiles=neuron_profiles,
        business_metadata=business_metadata,
        top_k=top_k,
    )

    category_metadata_path = training_dir / "neuron_category_metadata.json"
    with category_metadata_path.open("w", encoding="utf-8") as f:
        json.dump(category_metadata, f, indent=2)
    output_files["category_metadata"] = category_metadata_path
    logger.info(f"✓ Saved category metadata: {category_metadata_path}")

    precomputed_cache_dir = training_dir / "precomputed_ui_cache" / "neuron_wordclouds"
    precomputed_cache_dir.mkdir(parents=True, exist_ok=True)

    wordcloud_payloads = _build_precomputed_wordcloud_payloads(category_metadata)
    for neuron_id, payload in wordcloud_payloads.items():
        output_file = precomputed_cache_dir / f"neuron_{neuron_id}.json"
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    wordclouds_path = precomputed_cache_dir / "wordclouds.json"
    with wordclouds_path.open("w", encoding="utf-8") as f:
        json.dump(wordcloud_payloads, f, indent=2)

    output_files["wordcloud_cache"] = wordclouds_path
    logger.info(
        f"✓ Saved precomputed wordcloud cache: {wordclouds_path} ({len(wordcloud_payloads)} neurons)"
    )

    # Generate coactivation data
    if not skip_coactivation:
        generate_coactivations(training_dir, selected_labels)
        output_files["coactivation"] = training_dir / "neuron_coactivation.json"

    output_files["output_dir"] = output_dir
    write_latest_run_pointer(training_dir)
    return output_files


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Label neurons, generate superfeatures, and create coactivation data\n"
        "\nUSAGE MODES:\n"
        "  1. FULL (default): Labels + embeddings + superfeatures + coactivations\n"
        "  2. NO COACTIVATION: Labels + embeddings + superfeatures (skip coactivation)\n"
        "  3. COACTIVATION ONLY: Just generate coactivation data (skip all labeling)\n"
        "  4. NON-LLM: Tag-based + matrix-based labels without Gemini\n"
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
        "  python -m src.label --method tag-based\n"
        "\n  # Matrix-based labeling only\n"
        "  python -m src.label --method matrix-based\n"
        "\n  # All non-LLM labeling methods\n"
        "  python -m src.label --method non-llm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Optional: specify training directory (will auto-detect if not provided)
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=None,
        help="Training output directory (default: auto-detect latest run)",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Experiment directory containing manifest.json (labels every run in the experiment)",
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
        choices=["tag-based", "matrix-based", "llm-based", "non-llm", "both"],
        default="both",
        help="Labeling method to use (default: both; non-llm runs tag-based + matrix-based)",
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
    if args.training_dir and args.experiment_dir:
        parser.error("Cannot use --training-dir and --experiment-dir together")

    # Setup logging
    from src.utils import setup_logger

    setup_logger(__name__, level=logging.INFO)

    print("=" * 80)
    print("NEURON LABELING, SUPERFEATURE GENERATION, AND COACTIVATION DATA")
    print("=" * 80)

    # Call main labeling function
    if args.experiment_dir:
        results = _label_experiment_runs(
            args.experiment_dir,
            method=args.method,
            gemini_api_key=args.gemini_api_key,
            similarity_threshold=args.similarity_threshold,
            top_k=args.top_k,
            skip_coactivation=args.skip_coactivation,
            coactivation_only=args.coactivation_only,
        )
    elif args.training_dir is None:
        latest_experiment_dir = _find_latest_experiment_dir()
        if latest_experiment_dir:
            logger.info(
                f"Detected latest experiment directory, labeling all runs: {latest_experiment_dir}"
            )
            results = _label_experiment_runs(
                latest_experiment_dir,
                method=args.method,
                gemini_api_key=args.gemini_api_key,
                similarity_threshold=args.similarity_threshold,
                top_k=args.top_k,
                skip_coactivation=args.skip_coactivation,
                coactivation_only=args.coactivation_only,
            )
        else:
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
    else:
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

    if results.get("mode") == "experiment_batch":
        print("\n" + "=" * 80)
        print("✓ EXPERIMENT BATCH LABELING COMPLETE")
        print("=" * 80)
        print(f"Experiment directory: {results.get('experiment_dir', 'N/A')}")
        print(f"Runs labeled: {results.get('runs_labeled', 0)}")
        print(f"Batch manifest: {results.get('labeling_manifest', 'N/A')}")
    elif args.coactivation_only:
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
