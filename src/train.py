"""Training entry point for ELSA + TopK SAE POI recommender.

Pipeline:
    1. Load Yelp review/business data from DuckDB
  2. Build CSR matrix from interactions
  3. Train ELSA model on CSR matrix
  4. Encode users with ELSA (frozen), get latent vectors
  5. Train TopK SAE on latent vectors
  6. Save models and metrics to output directory

Usage
-----
    python src/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.data.shared_preprocessing_cache import (
    build_preprocessing_cache_key,
    get_shared_preprocessing_cache_dir,
    prepare_shared_preprocessing_cache,
    shared_preprocessing_manifest_path,
)
from src.models.collaborative_filtering import ELSA, NMSELoss
from src.models.sparse_autoencoder import TopKSAE
from src.run_registry import RunRegistry, create_run_id, write_latest_run_pointer
from src.ui.services.secrets_helper import get_cloud_storage_bucket
from src.utils import CheckpointManager, Config, load_config, setup_logger
from src.utils.evaluation import (
    build_holdout_split_sparse,
    compare_model_performance,
    compute_coverage,
    compute_entropy,
    evaluate_recommendations_batched,
    benchmark_inference,
    print_evaluation_report,
)

logger = logging.getLogger(__name__)


def _load_yaml_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a dictionary: {path}")
    return data


def _deep_merge_dict(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_sweep_dict(mapping: dict) -> list[dict]:
    """Expand a dict with list-valued leaves into a cartesian product of variants."""

    def _expand_node(node):
        if isinstance(node, dict):
            variants = [{}]
            for key, value in node.items():
                child_variants = _expand_node(value)
                next_variants = []
                for current in variants:
                    for child in child_variants:
                        combined = copy.deepcopy(current)
                        combined[key] = copy.deepcopy(child)
                        next_variants.append(combined)
                variants = next_variants
            return variants

        if isinstance(node, list):
            return [copy.deepcopy(item) for item in node]

        return [copy.deepcopy(node)]

    return _expand_node(mapping)


def _write_experiment_pointer(experiment_dir: Path) -> Path:
    pointer_path = Path.cwd() / "outputs" / "LATEST_EXPERIMENT.txt"
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(str(experiment_dir), encoding="utf-8")
    logger.info(
        f"Latest experiment pointer updated: {pointer_path} -> {experiment_dir}"
    )
    return pointer_path


def _save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _upload_experiment_manifest_to_cloud(
    experiment_dir: Path, experiment_id: str
) -> bool:
    """Upload experiment sweep metadata so the UI can discover it from GCS."""
    try:
        gcs_bucket_name = get_cloud_storage_bucket()
        if not gcs_bucket_name:
            logger.info("GCS_BUCKET_NAME not set, skipping experiment manifest upload")
            return True

        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
        manifest_path = experiment_dir / "manifest.json"
        if manifest_path.exists():
            cloud_storage.upload_json(
                manifest_path,
                f"experiments/{experiment_id}/manifest.json",
                metadata={"timestamp": experiment_id, "type": "experiment_manifest"},
            )

        latest_pointer = Path.cwd() / "outputs" / "LATEST_EXPERIMENT.txt"
        if latest_pointer.exists():
            blob = cloud_storage.bucket.blob("experiments/LATEST_EXPERIMENT.txt")
            blob.upload_from_filename(str(latest_pointer), content_type="text/plain")

        logger.info(f"✅ Uploaded experiment manifest to GCS: {experiment_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to upload experiment manifest to GCS: {e}")
        return False


def _run_experiment_sweep(args: argparse.Namespace) -> None:
    """Run multiple training jobs from an experiments YAML file."""

    project_root = Path(__file__).resolve().parent.parent
    experiment_config = _load_yaml_dict(args.config)
    if "experiments" not in experiment_config:
        raise ValueError(
            f"{args.config} does not contain an 'experiments' list; use normal training mode instead."
        )

    base_config = _load_yaml_dict(args.base_config)
    experiments = experiment_config.get("experiments", [])
    if not experiments:
        logger.warning("No experiments defined in experiments file")
        return

    output_base = Path(base_config.get("output", {}).get("base_dir", "outputs"))
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_base / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_experiment_pointer(experiment_dir)

    manifest: dict = {
        "experiment_id": experiment_id,
        "created": datetime.now().isoformat(),
        "source_config": str(Path(args.config).resolve()),
        "base_config": str(Path(args.base_config).resolve()),
        "runs": [],
    }

    logger.info("=" * 60)
    logger.info("EXPERIMENT SWEEP MODE")
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Experiments config: {args.config}")
    logger.info(f"Base config: {args.base_config}")
    logger.info("=" * 60)

    total_runs = 0
    prepared_cache_keys: set[str] = set()
    for experiment in experiments:
        experiment_name = experiment.get("name", "unnamed_experiment")
        experiment_desc = experiment.get("description", "")
        sweep_source = {
            key: value
            for key, value in experiment.items()
            if key not in {"name", "description"}
        }
        variants = _expand_sweep_dict(sweep_source)

        logger.info(
            f"Experiment '{experiment_name}' -> {len(variants)} run(s): {experiment_desc}"
        )

        for variant_idx, variant in enumerate(variants, start=1):
            total_runs += 1
            resolved_config = _deep_merge_dict(base_config, variant)
            resolved_config["experiment"] = {
                "experiment_id": experiment_id,
                "experiment_name": experiment_name,
                "experiment_description": experiment_desc,
                "variant_index": variant_idx,
                "variant_count": len(variants),
            }

            run_name = f"{experiment_name}_run{variant_idx:03d}"
            run_config_path = experiment_dir / f"{run_name}.yaml"
            _save_yaml(run_config_path, resolved_config)

            logger.info("-" * 60)
            logger.info(
                f"[{total_runs}] Running {run_name} (variant {variant_idx}/{len(variants)})"
            )
            logger.info("Resolved config saved to: %s", run_config_path)

            command = [
                sys.executable,
                "-m",
                "src.train",
                "--config",
                str(run_config_path),
            ]
            cache_key = build_preprocessing_cache_key(resolved_config)
            if args.skip_preprocessing or cache_key in prepared_cache_keys:
                command.append("--skip-preprocessing")
            if args.skip_elsa:
                command.append("--skip-elsa")
            if args.skip_sae:
                command.append("--skip-sae")
            if args.elsa_checkpoint:
                command.extend(["--elsa-checkpoint", args.elsa_checkpoint])
            if args.sae_checkpoint:
                command.extend(["--sae-checkpoint", args.sae_checkpoint])

            result = subprocess.run(command, cwd=project_root)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Experiment run failed: {run_name} (exit code {result.returncode})"
                )
            prepared_cache_keys.add(cache_key)

            latest_run_path = Path.cwd() / "outputs" / "LATEST_RUN.txt"
            if latest_run_path.exists():
                run_dir = Path(latest_run_path.read_text(encoding="utf-8").strip())
            else:
                run_dir = Path("outputs")

            summary_path = run_dir / "summary.json"
            summary = None
            if summary_path.exists():
                try:
                    with summary_path.open("r", encoding="utf-8") as f:
                        summary = json.load(f)
                except Exception as exc:
                    logger.warning(f"Could not load summary for {run_name}: {exc}")

            manifest["runs"].append(
                {
                    "run_name": run_name,
                    "experiment_name": experiment_name,
                    "variant_index": variant_idx,
                    "run_dir": str(run_dir),
                    "config_path": str(run_config_path),
                    "summary_path": str(summary_path),
                    "summary": summary,
                    "parameters": variant,
                }
            )

    manifest_path = experiment_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved experiment manifest: {manifest_path}")
    _upload_experiment_manifest_to_cloud(experiment_dir, experiment_id)
    logger.info(f"Total experiment runs completed: {total_runs}")


def _build_business_metadata_from_db(
    db_path: str | Path,
    item_map_after_kcore: dict,
    state_filter: str | None = None,
) -> dict:
    """Build business metadata for the items that survived k-core filtering."""
    businesses = load_businesses(
        db_path=db_path,
        state_filter=state_filter,
        min_review_count=0,
    )

    item_ids = {str(business_id) for business_id in item_map_after_kcore.keys()}
    businesses = businesses[businesses["business_id"].astype(str).isin(item_ids)]

    business_metadata = {}
    for _, row in businesses.iterrows():
        categories = []
        if "categories" in row and row["categories"] is not None:
            categories = [
                category.strip()
                for category in str(row["categories"]).split(",")
                if category.strip()
            ]

        business_metadata[str(row["business_id"])] = {
            "name": row.get("name", "Unknown"),
            "city": row.get("city", "Unknown"),
            "state": row.get("state", "Unknown"),
            "categories": categories,
            "stars": row.get("stars", None),
            "review_count": row.get("review_count", None),
        }

    return business_metadata


def _score_elsa_batch(elsa_model: ELSA, batch: np.ndarray, device: str) -> np.ndarray:
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = elsa_model.decode(elsa_model.encode(batch_tensor)).cpu().numpy()
    scores = scores - batch
    scores[batch > 0] = -np.inf
    return scores


def _score_sae_batch(
    elsa_model: ELSA,
    sae_model: TopKSAE,
    batch: np.ndarray,
    device: str,
) -> np.ndarray:
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = (
            elsa_model.decode(
                sae_model.decode(sae_model.encode(elsa_model.encode(batch_tensor)))
            )
            .cpu()
            .numpy()
        )
    scores = scores - batch
    scores[batch > 0] = -np.inf
    return scores


def _compute_auxiliary_metrics(
    elsa_model: ELSA,
    sae_model: TopKSAE,
    X_input_csr,
    *,
    device: str,
    batch_size: int,
    top_k: int,
) -> tuple[float, float, dict[str, float]]:
    n_items = X_input_csr.shape[1]
    item_counts = np.zeros(n_items, dtype=np.int64)
    total_recommendations = 0
    latency_sample_scores: list[np.ndarray] = []
    collected_latency_users = 0
    target_latency_users = min(100, X_input_csr.shape[0])

    for start in range(0, X_input_csr.shape[0], batch_size):
        end = min(start + batch_size, X_input_csr.shape[0])
        batch = X_input_csr[start:end].toarray().astype(np.float32)
        scores = _score_sae_batch(elsa_model, sae_model, batch, device)

        top_items = np.argsort(-scores, axis=1)[:, :top_k]
        for row in top_items:
            item_counts[row] += 1
        total_recommendations += top_items.size

        if collected_latency_users < target_latency_users:
            take = min(scores.shape[0], target_latency_users - collected_latency_users)
            latency_sample_scores.append(scores[:take])
            collected_latency_users += take

    coverage = float(np.count_nonzero(item_counts) / n_items) if n_items > 0 else 0.0

    if total_recommendations > 0 and n_items > 1:
        probs = item_counts[item_counts > 0] / total_recommendations
        entropy = float(-np.sum(probs * np.log(probs)) / np.log(n_items))
    else:
        entropy = 0.0

    latency_metrics = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    if latency_sample_scores:
        latency_metrics = benchmark_inference(
            np.vstack(latency_sample_scores),
            n_samples=min(100, collected_latency_users),
        )

    return coverage, entropy, latency_metrics


def upload_results_to_cloud(output_dir: Path, timestamp: str) -> bool:
    """
    Upload training results to GCS if configured.

    Args:
        output_dir: Local output directory with results
        timestamp: Training timestamp (YYYYMMDD_HHMMSS)

    Returns:
        True if uploaded or not configured, False if upload failed
    """
    try:
        gcs_bucket_name = get_cloud_storage_bucket()
        if not gcs_bucket_name:
            logger.info("GCS_BUCKET_NAME not set, skipping cloud upload")
            return True

        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
        logger.info(f"Uploading results to GCS bucket: {gcs_bucket_name}")

        gcs_prefix = f"models/{timestamp}"

        def _upload_file(
            local_file: Path, gcs_path: str, content_type: Optional[str] = None
        ) -> None:
            if not local_file.exists():
                return
            blob = cloud_storage.bucket.blob(gcs_path)
            if content_type:
                blob.upload_from_filename(str(local_file), content_type=content_type)
            else:
                blob.upload_from_filename(str(local_file))
            logger.info(
                f"✅ Uploaded {local_file.name} → gs://{gcs_bucket_name}/{gcs_path}"
            )

        # Upload summary.json
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            cloud_storage.upload_json(
                summary_path,
                f"{gcs_prefix}/summary.json",
                metadata={"timestamp": timestamp, "type": "training_summary"},
            )

        # Upload resolved config for experiment comparison / ablations
        config_path = output_dir / "resolved_config.yaml"
        if config_path.exists():
            blob = cloud_storage.bucket.blob(f"{gcs_prefix}/resolved_config.yaml")
            blob.upload_from_filename(str(config_path), content_type="text/yaml")

        # Upload training_results.json
        results_path = output_dir / "training_results.json"
        if results_path.exists():
            cloud_storage.upload_json(
                results_path,
                f"{gcs_prefix}/training_results.json",
                metadata={"timestamp": timestamp, "type": "training_results"},
            )

        # Upload ranking metrics report (text)
        report_path = output_dir / "ranking_metrics_report.txt"
        _upload_file(
            report_path, f"{gcs_prefix}/ranking_metrics_report.txt", "text/plain"
        )

        # Upload model checkpoints so the Streamlit Cloud UI can load inference models
        checkpoints_dir = output_dir / "checkpoints"
        if checkpoints_dir.exists():
            for checkpoint_file in checkpoints_dir.glob("*.pt"):
                _upload_file(
                    checkpoint_file,
                    f"{gcs_prefix}/checkpoints/{checkpoint_file.name}",
                    "application/octet-stream",
                )

        # Upload interpretability artifacts used by the UI
        interpretations_dir = output_dir / "neuron_interpretations"
        if interpretations_dir.exists():
            for artifact_file in interpretations_dir.glob("*"):
                if artifact_file.is_file():
                    content_type = (
                        "application/json"
                        if artifact_file.suffix.lower() == ".json"
                        else "application/octet-stream"
                    )
                    _upload_file(
                        artifact_file,
                        f"{gcs_prefix}/neuron_interpretations/{artifact_file.name}",
                        content_type,
                    )

        _upload_file(
            output_dir / "neuron_category_metadata.json",
            f"{gcs_prefix}/neuron_category_metadata.json",
            "application/json",
        )
        _upload_file(
            output_dir / "neuron_coactivation.json",
            f"{gcs_prefix}/neuron_coactivation.json",
            "application/json",
        )

        precomputed_cache_dir = (
            output_dir / "precomputed_ui_cache" / "neuron_wordclouds"
        )
        if precomputed_cache_dir.exists():
            for cache_file in precomputed_cache_dir.glob("*"):
                if cache_file.is_file():
                    content_type = (
                        "application/json"
                        if cache_file.suffix.lower() == ".json"
                        else "application/octet-stream"
                    )
                    _upload_file(
                        cache_file,
                        f"{gcs_prefix}/precomputed_ui_cache/neuron_wordclouds/{cache_file.name}",
                        content_type,
                    )

        # Small pipeline metadata useful for cloud diagnostics
        for metadata_file in [
            output_dir / "business_metadata.pkl",
            output_dir / "item_map_after_kcore.pkl",
            output_dir / "reviews_df.pkl",
            output_dir / "data" / "train_user_ids.json",
            output_dir / "data" / "test_user_ids.json",
            output_dir / "data" / "test_users_top50.json",
        ]:
            _upload_file(
                metadata_file,
                f"{gcs_prefix}/data/{metadata_file.name}"
                if metadata_file.parent.name == "data"
                else f"{gcs_prefix}/{metadata_file.name}",
                "application/json"
                if metadata_file.suffix.lower() == ".json"
                else "application/octet-stream",
            )

        logger.info(
            f"✅ Training results uploaded to gs://{gcs_bucket_name}/{gcs_prefix}/"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to upload results to cloud: {e}")
        logger.warning(
            "Continuing with local-only results (app won't access them on cloud)"
        )
        return False


def precompute_user_csr_matrices(
    reviews_df,
    item_map_after_kcore,
    output_dir: Path,
    upload_to_cloud: bool = True,
    top_n_users: int = 50,
    allowed_user_ids: Optional[list[str]] = None,
):
    """
    Precompute CSR matrices for top-N users (for Streamlit app).

    Builds a 1×n_items sparse matrix for each user representing their interaction history.
    Only computes for the top N users by interaction count (default: 50 for Streamlit demo).
    This enables fast lookup in the Streamlit app without querying the database each time.

    Args:
        reviews_df: DataFrame with user_id and business_id columns
        item_map_after_kcore: business_id -> model_index mapping (filtered)
        output_dir: Training output directory for saving results
        upload_to_cloud: Whether to upload to Cloud Storage
        top_n_users: Number of top users to precompute (default: 50)
        allowed_user_ids: Optional allowlist of users to consider. If provided,
            ranking is done only within this subset.

    Returns:
        Dict of {user_id: csr_matrix}
    """
    logger.info("=" * 60)
    logger.info("PRECOMPUTING USER CSR MATRICES")
    logger.info("=" * 60)

    precomp_dir = output_dir / "precomputed"
    precomp_dir.mkdir(parents=True, exist_ok=True)

    local_path = precomp_dir / "user_csr_matrices.pkl"

    n_items = len(item_map_after_kcore)

    candidate_reviews = reviews_df
    if allowed_user_ids is not None:
        allowed_user_ids = list(dict.fromkeys(allowed_user_ids))
        candidate_reviews = reviews_df[reviews_df["user_id"].isin(allowed_user_ids)]
        logger.info(
            f"Restricting user CSR precompute to {len(allowed_user_ids)} allowed users"
        )

    # Find top N users by interaction count
    user_interaction_counts = (
        candidate_reviews.groupby("user_id").size().sort_values(ascending=False)
    )
    top_users = user_interaction_counts.head(top_n_users).index.tolist()

    logger.info(
        f"Total users in dataset: {len(user_interaction_counts)}, using top {len(top_users)} by interaction count"
    )
    logger.info(
        f"Interaction distribution - min: {user_interaction_counts[top_users].min()}, max: {user_interaction_counts[top_users].max()}"
    )

    # Filter reviews to only top users
    reviews_filtered = candidate_reviews[candidate_reviews["user_id"].isin(top_users)]
    all_users = reviews_filtered["user_id"].unique()

    logger.info(f"Building CSR matrices for {len(all_users)} users, {n_items} items...")

    user_matrices = {}
    failed_users = []
    matrices_built = 0

    for user_idx, user_id in enumerate(all_users, 1):
        try:
            # Get all interactions for this user (from filtered reviews)
            user_reviews = reviews_filtered[reviews_filtered["user_id"] == user_id]
            business_ids = user_reviews["business_id"].values

            # Map business IDs to model indices (skip if not in filtered set)
            poi_indices = [
                item_map_after_kcore[bid]
                for bid in business_ids
                if bid in item_map_after_kcore
            ]

            if not poi_indices:
                failed_users.append((user_id, "no_valid_interactions"))
                continue

            # Build CSR matrix: 1 row (user), n_items columns
            row = np.zeros(len(poi_indices), dtype=int)
            col = np.array(poi_indices, dtype=int)
            data_vals = np.ones(len(poi_indices), dtype=np.float32)

            user_csr = csr_matrix((data_vals, (row, col)), shape=(1, n_items))
            user_matrices[user_id] = user_csr
            matrices_built += 1

            if user_idx % 500 == 0 or user_idx == len(all_users):
                logger.info(
                    f"[{user_idx}/{len(all_users)}] Built {matrices_built} matrices..."
                )

        except Exception as e:
            logger.debug(f"Failed to build matrix for user {user_id}: {e}")
            failed_users.append((user_id, str(e)))

    logger.info(f"✅ Successfully built {matrices_built} user CSR matrices")

    if failed_users:
        logger.warning(f"⚠️ Failed for {len(failed_users)} users")

    # Save locally
    logger.info(f"💾 Saving {matrices_built} matrices to {local_path}...")
    try:
        with open(local_path, "wb") as f:
            pickle.dump(user_matrices, f)
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Saved locally: {local_path} ({file_size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"❌ Failed to save locally: {e}")
        return {}

    # Upload to cloud
    if upload_to_cloud:
        try:
            gcs_bucket_name = get_cloud_storage_bucket()
            if not gcs_bucket_name:
                logger.info(
                    "GCS not configured, skipping cloud upload of user matrices"
                )
                return user_matrices

            from src.ui.services.cloud_storage_helper import CloudStorageHelper

            cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
            gcs_paths = [f"models/{output_dir.name}/precomputed/user_csr_matrices.pkl"]
            if allowed_user_ids is None:
                gcs_paths.append("metadata/user_csr_matrices.pkl")

            for gcs_path in gcs_paths:
                blob = cloud_storage.bucket.blob(gcs_path)
                blob.upload_from_filename(str(local_path))
                logger.info(f"âś… Uploaded to gs://{gcs_bucket_name}/{gcs_path}")
        except Exception as e:
            logger.warning(f"⚠️ Cloud upload failed (app will use local file): {e}")

    return user_matrices


def persist_test_user_artifacts(
    output_dir: Path,
    reviews_df,
    train_user_ids: list[str],
    test_user_ids: list[str],
    *,
    top_k: int = 50,
) -> dict[str, Path]:
    """Persist run-scoped train/test user artifacts for the UI."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train_ids_path = data_dir / "train_user_ids.json"
    test_ids_path = data_dir / "test_user_ids.json"
    top_users_path = data_dir / "test_users_top50.json"

    interaction_counts = (
        reviews_df[reviews_df["user_id"].isin(test_user_ids)]
        .groupby("user_id")
        .size()
        .sort_values(ascending=False)
    )

    top_users = [
        {"id": str(user_id), "interactions": int(count)}
        for user_id, count in interaction_counts.head(top_k).items()
    ]

    train_ids_path.write_text(json.dumps([str(uid) for uid in train_user_ids], indent=2))
    test_ids_path.write_text(json.dumps([str(uid) for uid in test_user_ids], indent=2))
    top_users_path.write_text(json.dumps(top_users, indent=2))

    logger.info("Saved train/test user artifacts to %s", data_dir)
    logger.info("Saved %d ranked test users for UI", len(top_users))

    return {
        "train_user_ids": train_ids_path,
        "test_user_ids": test_ids_path,
        "test_users_top50": top_users_path,
    }


class SparseDataset(Dataset):
    """Dataset wrapper for sparse CSR matrices that converts rows on-the-fly."""

    def __init__(self, csr_matrix):
        self.data = csr_matrix

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx].toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Examples
    --------
    # Full pipeline (default)
    python -m src.train --config configs/default.yaml

    # Only train ELSA (reuse existing preprocessing)
    python -m src.train --config configs/default.yaml --skip-sae

    # Only train SAE (reuse existing ELSA + preprocessing)
    python -m src.train --config configs/default.yaml --skip-elsa

    # Skip preprocessing, require an existing shared preprocessing cache
    python -m src.train --config configs/default.yaml --skip-preprocessing

    # Experiments: try different SAE hyperparams without retraining ELSA
    python -m src.train --config configs/default.yaml --skip-elsa --skip-preprocessing

    """
    parser = argparse.ArgumentParser(
        description="Train ELSA + TopK SAE POI recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Component selection:\n"
        "  --skip-preprocessing  Require existing shared preprocessing cache; do not rebuild raw preprocessing\n"
        "  --skip-elsa           Skip ELSA training (reuse best checkpoint)\n"
        "  --skip-sae            Skip SAE training (reuse best checkpoint)\n"
        "\nFor grid search experiments, use --skip-elsa --skip-preprocessing to\n"
        "only train SAE with different hyperparameters while reusing ELSA outputs.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml). "
        "If the file contains a top-level 'experiments' list, train.py switches to sweep mode.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/default.yaml",
        help="Base config used when --config points to an experiments YAML file.",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Require an existing shared preprocessing cache for this data configuration. "
        "If the cache is missing or invalid, training fails instead of rebuilding preprocessing.",
    )
    parser.add_argument(
        "--skip-elsa",
        action="store_true",
        help="Skip ELSA training. Loads best ELSA checkpoint from previous run. "
        "Useful for SAE hyperparameter tuning without retraining ELSA.",
    )
    parser.add_argument(
        "--skip-sae",
        action="store_true",
        help="Skip SAE training. Useful for testing data pipeline or ELSA only.",
    )
    parser.add_argument(
        "--elsa-checkpoint",
        default=None,
        help="Path to specific ELSA checkpoint to load (overrides 'latest' search). "
        "Format: path/to/outputs/YYYYMMDD_HHMMSS/checkpoints/elsa_best.pt",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default=None,
        help="Path to specific SAE checkpoint to load (for SAE-only experiments).",
    )
    return parser.parse_args()


class MetricsCollector:
    """Collect and report metrics during training."""

    def __init__(self) -> None:
        self.data: dict[str, list] = {}

    def record(self, epoch: int, **kwargs: float) -> None:
        """Record metrics for an epoch."""
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def to_dict(self) -> dict[str, list]:
        """Get metrics as dictionary."""
        return self.data

    def get_summary(self) -> str:
        """Get formatted summary of latest metrics."""
        if not self.data:
            return "No metrics recorded"

        lines = []
        for key, values in self.data.items():
            if values:
                latest = values[-1]
                lines.append(f"  {key}: {latest:.6f}")
        return "\n".join(lines)


def cosine_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine reconstruction loss (1 - cosine_similarity)."""
    recon_norm = torch.nn.functional.normalize(recon, dim=-1)
    target_norm = torch.nn.functional.normalize(target, dim=-1)
    return (
        1.0
        - torch.nn.functional.cosine_similarity(recon_norm, target_norm, dim=-1).mean()
    )


def compute_model_sizes(elsa_model: ELSA, sae_model: TopKSAE, output_dir: Path) -> dict:
    """Compute and save model file sizes in bytes and MB."""
    elsa_temp = output_dir / "elsa_temp_size.pt"
    sae_temp = output_dir / "sae_temp_size.pt"

    try:
        # Save model state dicts to temp files
        torch.save(elsa_model.state_dict(), elsa_temp)
        torch.save(sae_model.state_dict(), sae_temp)

        # Get file sizes
        elsa_size_bytes = elsa_temp.stat().st_size
        sae_size_bytes = sae_temp.stat().st_size

        sizes = {
            "elsa_bytes": int(elsa_size_bytes),
            "elsa_mb": float(elsa_size_bytes / 1e6),
            "sae_bytes": int(sae_size_bytes),
            "sae_mb": float(sae_size_bytes / 1e6),
            "total_bytes": int(elsa_size_bytes + sae_size_bytes),
            "total_mb": float((elsa_size_bytes + sae_size_bytes) / 1e6),
        }

        logger.info(
            f"Model sizes: ELSA={sizes['elsa_mb']:.2f}MB, "
            f"SAE={sizes['sae_mb']:.2f}MB, Total={sizes['total_mb']:.2f}MB"
        )

        return sizes
    finally:
        # Clean up temp files
        elsa_temp.unlink(missing_ok=True)
        sae_temp.unlink(missing_ok=True)


def compute_sparsity_stats(
    sae_model: TopKSAE, Z_data: torch.Tensor, device: str, batch_size: int = 256
) -> dict:
    """Compute sparsity statistics (active neurons, sparsity ratio)."""
    sae_model.eval()
    all_active_counts = []

    loader = DataLoader(TensorDataset(Z_data), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (z_batch,) in loader:
            z_batch = z_batch.to(device)
            _, h_sparse, _ = sae_model(z_batch)

            # Count active neurons per sample
            active_per_sample = (h_sparse != 0).sum(dim=1).float()
            all_active_counts.append(active_per_sample)

    all_active = torch.cat(all_active_counts)
    avg_active = all_active.mean().item()
    max_active = all_active.max().item()

    # Total neurons in SAE hidden layer (dictionary size)
    total_neurons = sae_model.hidden_dim

    return {
        "avg_active_neurons": float(avg_active),
        "max_active_neurons": float(max_active),
        "total_neurons": int(total_neurons),
        "sparsity_ratio": float(1.0 - (avg_active / total_neurons)),
    }


def train_elsa(
    config: Config,
    X_train,
    X_val,
    n_items: int,
    checkpoint_mgr: CheckpointManager,
) -> tuple[ELSA, float, dict]:
    """Train ELSA model.

    Parameters
    ----------
    config : Config
        Training configuration.
    X_train : torch.Tensor or Dataset
        Training interaction matrix (dense tensor or Dataset).
    X_val : torch.Tensor or Dataset
        Validation interaction matrix.
    n_items : int
        Number of items in the interaction matrix.
    checkpoint_mgr : CheckpointManager
        Checkpoint manager for saving.

    Returns
    -------
    tuple[ELSA, float, dict]
        Trained ELSA model, best validation loss, and training statistics
    """
    logger.info("=" * 60)
    logger.info("TRAINING ELSA")
    logger.info("=" * 60)

    elsa_cfg = config["elsa"]

    device = elsa_cfg["device"]
    model = ELSA(n_items, latent_dim=elsa_cfg["latent_dim"]).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(elsa_cfg["learning_rate"]),
        betas=(0.9, 0.99),
        weight_decay=float(elsa_cfg.get("weight_decay", 0.0)),
    )
    criterion = NMSELoss()

    train_loader = torch.utils.data.DataLoader(
        X_train, batch_size=elsa_cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        X_val, batch_size=elsa_cfg["batch_size"], shuffle=False
    )

    metrics = MetricsCollector()
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_reason = None
    epoch_started = 0

    logger.info(
        f"Config: latent_dim={elsa_cfg['latent_dim']}, "
        f"lr={elsa_cfg['learning_rate']}, epochs={elsa_cfg['num_epochs']}"
    )

    training_start = time.time()

    for epoch in range(elsa_cfg["num_epochs"]):
        # Training
        model.train()
        train_loss = 0.0

        for x_batch in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(X_train)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch.to(device)
                recon = model(x_batch)
                loss = criterion(recon, x_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(X_val)

        metrics.record(epoch, train_loss=train_loss, val_loss=val_loss)

        logger.info(
            f"Epoch {epoch+1:3d}/{elsa_cfg['num_epochs']} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            epoch_started = epoch
            # Save with dataset metadata
            metadata = {
                "n_items": model.A.shape[0],
                "latent_dim": model.latent_dim,
            }
            checkpoint_mgr.save(
                model,
                epoch=epoch,
                metrics=metrics.to_dict(),
                name="elsa_best",
                metadata=metadata,
            )
        else:
            patience_counter += 1
            if patience_counter >= elsa_cfg["patience"]:
                early_stop_reason = f"patience threshold ({elsa_cfg['patience']} epochs without improvement)"
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    training_time = time.time() - training_start

    # Load best model
    model.load_state_dict(
        torch.load(checkpoint_mgr.checkpoint_dir / "elsa_best.pt", map_location=device)[
            "model_state_dict"
        ]
    )
    checkpoint_mgr.save_metrics(metrics.to_dict(), split="elsa_train")

    logger.info(f"ELSA training complete. Best val_loss={best_val_loss:.6f}")

    return (
        model,
        best_val_loss,
        {
            "best_epoch": int(epoch_started),
            "final_epoch": int(epoch),
            "training_time_sec": float(training_time),
            "early_stop_reason": early_stop_reason or "completed all epochs",
        },
    )


def train_sae(
    config: Config,
    elsa_model: ELSA,
    Z_train: torch.Tensor,
    Z_val: torch.Tensor,
    checkpoint_mgr: CheckpointManager,
) -> tuple[TopKSAE, float, dict]:
    """Train TopK SAE model.

    Parameters
    ----------
    config : Config
        Training configuration.
    elsa_model : ELSA
        Trained ELSA model (will be frozen).
    Z_train : torch.Tensor
        Training latent vectors from ELSA.
    Z_val : torch.Tensor
        Validation latent vectors from ELSA.
    checkpoint_mgr : CheckpointManager
        Checkpoint manager for saving.

    Returns
    -------
    tuple[TopKSAE, float, dict]
        Trained SAE model, best validation loss, and training statistics.
    """
    logger.info("=" * 60)
    logger.info("TRAINING TOPK SAE")
    logger.info("=" * 60)

    sae_cfg = config["sae"]
    elsa_cfg = config["elsa"]

    device = sae_cfg["device"]
    hidden_dim = sae_cfg["width_ratio"] * elsa_cfg["latent_dim"]

    sae = TopKSAE(
        input_dim=elsa_cfg["latent_dim"],
        hidden_dim=hidden_dim,
        k=sae_cfg["k"],
        l1_coef=float(sae_cfg["l1_coef"]),
    ).to(device)

    optimizer = optim.Adam(
        sae.parameters(),
        lr=float(sae_cfg["learning_rate"]),
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    train_loader = torch.utils.data.DataLoader(
        Z_train, batch_size=sae_cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        Z_val, batch_size=sae_cfg["batch_size"], shuffle=False
    )

    metrics = MetricsCollector()
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_reason = None
    epoch_started = 0

    logger.info(
        f"Config: width_ratio={sae_cfg['width_ratio']}, "
        f"hidden_dim={hidden_dim}, k={sae_cfg['k']}, "
        f"l1_coef={sae_cfg['l1_coef']}, epochs={sae_cfg['num_epochs']}"
    )

    training_start = time.time()

    for epoch in range(sae_cfg["num_epochs"]):
        # Training
        sae.train()
        train_recon_loss = 0.0
        train_l1_loss = 0.0
        train_active_neurons = []

        for z_batch in train_loader:
            z_batch = z_batch.to(device)
            optimizer.zero_grad()

            recon, h_sparse, _ = sae(z_batch)

            rec_loss = cosine_recon_loss(recon, z_batch)
            l1_loss = h_sparse.abs().mean()
            loss = rec_loss + float(sae_cfg["l1_coef"]) * l1_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

            train_recon_loss += rec_loss.item() * z_batch.size(0)
            train_l1_loss += l1_loss.item() * z_batch.size(0)

            # Track sparsity
            active_per_sample = (h_sparse != 0).sum(dim=1).float()
            train_active_neurons.append(active_per_sample)

        train_recon_loss /= len(Z_train)
        train_l1_loss /= len(Z_train)
        avg_train_active = torch.cat(train_active_neurons).mean().item()

        # Validation
        sae.eval()
        val_recon_loss = 0.0
        cosine_sims = []
        val_active_neurons = []

        with torch.no_grad():
            for z_batch in val_loader:
                z_batch = z_batch.to(device)

                recon, h_sparse, _ = sae(z_batch)

                rec_loss = cosine_recon_loss(recon, z_batch)
                val_recon_loss += rec_loss.item() * z_batch.size(0)

                # Cosine similarity
                recon_norm = torch.nn.functional.normalize(recon, dim=-1)
                z_norm = torch.nn.functional.normalize(z_batch, dim=-1)
                cos_sim = torch.nn.functional.cosine_similarity(
                    recon_norm, z_norm, dim=-1
                ).mean()
                cosine_sims.append(cos_sim.item())

                # Track sparsity
                active_per_sample = (h_sparse != 0).sum(dim=1).float()
                val_active_neurons.append(active_per_sample)

        val_recon_loss /= len(Z_val)
        avg_cosine_sim = np.mean(cosine_sims)
        avg_val_active = torch.cat(val_active_neurons).mean().item()

        metrics.record(
            epoch,
            train_recon=train_recon_loss,
            train_l1=train_l1_loss,
            train_active=avg_train_active,
            val_recon=val_recon_loss,
            val_active=avg_val_active,
            cosine_sim=avg_cosine_sim,
        )

        logger.info(
            f"Epoch {epoch+1:3d}/{sae_cfg['num_epochs']} | "
            f"train_recon={train_recon_loss:.6f} train_l1={train_l1_loss:.6f} | "
            f"val_recon={val_recon_loss:.6f} cosine_sim={avg_cosine_sim:.4f} | "
            f"active={avg_val_active:.1f}"
        )

        # Early stopping
        if (best_val_loss - val_recon_loss) > float(sae_cfg.get("min_delta", 0.0)):
            best_val_loss = val_recon_loss
            patience_counter = 0
            epoch_started = epoch
            # Save with hyperparameter metadata
            metadata = {
                "k": sae_cfg["k"],
                "width_ratio": sae_cfg["width_ratio"],
                "latent_dim": elsa_model.latent_dim,
            }
            checkpoint_mgr.save(
                sae,
                epoch=epoch,
                metrics=metrics.to_dict(),
                name=f"sae_r{sae_cfg['width_ratio']}_k{sae_cfg['k']}_best",
                metadata=metadata,
            )
        else:
            patience_counter += 1
            if patience_counter >= sae_cfg["patience"]:
                early_stop_reason = f"patience threshold ({sae_cfg['patience']} epochs without improvement)"
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    training_time = time.time() - training_start

    # Load best model
    model_path = (
        checkpoint_mgr.checkpoint_dir
        / f"sae_r{sae_cfg['width_ratio']}_k{sae_cfg['k']}_best.pt"
    )
    sae.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
    )
    checkpoint_mgr.save_metrics(metrics.to_dict(), split="sae_train")

    logger.info(f"SAE training complete. Best val_recon={best_val_loss:.6f}")

    return (
        sae,
        best_val_loss,
        {
            "best_epoch": int(epoch_started),
            "final_epoch": int(epoch),
            "training_time_sec": float(training_time),
            "early_stop_reason": early_stop_reason or "completed all epochs",
        },
    )


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    raw_config = _load_yaml_dict(args.config)
    if "experiments" in raw_config:
        _run_experiment_sweep(args)
        return

    # Load config
    config = load_config(args.config)

    # Create output directory with timestamp
    output_cfg = config["output"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_cfg["base_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize run registry
    registry = RunRegistry()
    registry.register_run(
        run_id,
        "train",
        config={"skip_preprocessing": args.skip_preprocessing},
        status="pending",
    )

    # Set up logging
    setup_logger(
        __name__,
        log_dir=output_dir,
        level=getattr(logging, output_cfg["log_level"]),
    )
    logger.info(f"Output directory: {output_dir}")

    # Save resolved config for downstream comparison / UI selectors
    resolved_config_path = output_dir / "resolved_config.yaml"
    _save_yaml(resolved_config_path, config.to_dict())
    logger.info(f"Saved resolved config to: {resolved_config_path}")

    # Checkpoint manager
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # Device
    device = config["elsa"]["device"]
    logger.info(f"Using device: {device}")

    try:
        # Load data
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)

        db_path = config["data"]["db_path"]
        config_dict = config.to_dict()
        shared_cache_dir = get_shared_preprocessing_cache_dir(config_dict)
        preprocessing_cache_key = build_preprocessing_cache_key(config_dict)

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            raise FileNotFoundError(f"DuckDB database not found: {db_path_obj}")

        if args.skip_preprocessing:
            logger.info(
                "--skip-preprocessing: loading shared preprocessing cache from %s",
                shared_cache_dir,
            )

        preprocessing_payload, preprocessing_source, _ = prepare_shared_preprocessing_cache(
            config_dict, require_existing=args.skip_preprocessing
        )
        preprocessing_manifest = preprocessing_payload["manifest"]
        reviews = preprocessing_payload["reviews"]
        X_csr = preprocessing_payload["final_dataset"].csr
        item_map_after_kcore = preprocessing_payload["item_map_after_kcore"]
        final_user_ids = preprocessing_payload["final_user_ids"]
        universal_user_map = preprocessing_payload["universal_user_map"]
        universal_business_map = preprocessing_payload["universal_business_map"]
        item_count_before_kcore = preprocessing_manifest["counts"]["raw_n_items"]

        logger.info(
            "Using shared preprocessing dataset: %d users x %d items, %d interactions",
            X_csr.shape[0],
            X_csr.shape[1],
            X_csr.nnz,
        )

        # Save universal mappings for downstream use (e.g., labeling notebook)
        mappings_dir = output_dir / "mappings"
        mappings_dir.mkdir(parents=True, exist_ok=True)

        with open(mappings_dir / "user2index_universal.pkl", "wb") as f:
            pickle.dump(universal_user_map, f)
        with open(mappings_dir / "business2index_universal.pkl", "wb") as f:
            pickle.dump(universal_business_map, f)

        logger.info(f"Universal mappings saved to {mappings_dir}")

        # Save FILTERED item mapping (after k-core)
        with open(mappings_dir / "item2index.pkl", "wb") as f:
            pickle.dump(item_map_after_kcore, f)

        logger.info(
            f"Saved item2index_filtered (model space): {len(item_map_after_kcore)} items"
        )

        if len(final_user_ids) != X_csr.shape[0]:
            raise RuntimeError(
                "Shared preprocessing cache has inconsistent user ordering: "
                f"{len(final_user_ids)} user IDs vs CSR rows {X_csr.shape[0]}"
            )

        # Train/test split
        n_users = X_csr.shape[0]
        user_indices = np.arange(n_users)
        train_test_ratio = config["data"]["train_test_split"]
        val_ratio = config["data"]["val_split"]

        train_users, test_users = train_test_split(
            user_indices,
            test_size=1 - train_test_ratio,
            random_state=config["data"]["seed"],
        )

        n_test = len(test_users)
        n_train = len(train_users)
        logger.info(
            f"Train/test split: {n_train} users ({train_test_ratio*100:.0f}%) → train, "
            f"{n_test} users ({(1-train_test_ratio)*100:.0f}%) → test"
        )

        X_train_csr = X_csr[train_users]
        X_test_csr = X_csr[test_users]
        train_user_ids = [str(final_user_ids[idx]) for idx in train_users]
        test_user_ids = [str(final_user_ids[idx]) for idx in test_users]

        # Create datasets that handle sparse matrices efficiently
        X_train_dataset = SparseDataset(X_train_csr)
        X_test_dataset = SparseDataset(X_test_csr)

        # Train/val split on indices (not data)
        train_indices = np.arange(X_train_csr.shape[0])
        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=val_ratio,
            random_state=config["data"]["seed"],
        )

        n_val = len(val_idx)
        n_train_split = len(train_idx)
        logger.info(
            f"Train/val split (from training data): {n_train_split} users ({(1-val_ratio)*100:.0f}%) → train, "
            f"{n_val} users ({val_ratio*100:.0f}%) → validation"
        )

        # Create subset datasets
        from torch.utils.data import Subset

        X_train_split = Subset(X_train_dataset, train_idx)
        X_val_split = Subset(X_train_dataset, val_idx)

        # Train ELSA
        elsa_model, elsa_best_loss, elsa_stats = train_elsa(
            config, X_train_split, X_val_split, X_train_csr.shape[1], checkpoint_mgr
        )

        # Encode all users with ELSA (frozen) using chunked encoding for large matrices
        logger.info("Encoding users with ELSA...")
        elsa_model.eval()
        with torch.no_grad():
            # Use chunked encoding for sparse matrices to avoid memory overflow
            Z_train = elsa_model.encode_csr_chunked(
                X_train_csr, chunk_size=4096, device=device
            )
            Z_val = elsa_model.encode_csr_chunked(
                X_train_csr[val_idx], chunk_size=4096, device=device
            )
            Z_test = elsa_model.encode_csr_chunked(
                X_test_csr, chunk_size=4096, device=device
            )

        logger.info(
            f"Encoded: z_train={Z_train.shape}, z_val={Z_val.shape}, z_test={Z_test.shape}"
        )

        # Train SAE
        sae_model, sae_best_loss, sae_stats = train_sae(
            config, elsa_model, Z_train, Z_val, checkpoint_mgr
        )

        # Final evaluation on test set
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 60)

        sae_model.eval()
        elsa_model.eval()
        with torch.no_grad():
            z_test_recon = sae_model.enc(Z_test)
            h_test = sae_model.encode(Z_test)
            z_recon = sae_model.decode(h_test)

            test_recon_loss = cosine_recon_loss(z_recon, Z_test).item()
            z_recon_norm = torch.nn.functional.normalize(z_recon, dim=-1)
            z_test_norm = torch.nn.functional.normalize(Z_test, dim=-1)
            test_cosine_sim = (
                torch.nn.functional.cosine_similarity(z_recon_norm, z_test_norm, dim=-1)
                .mean()
                .item()
            )

            # Sparsity analysis
            active_neurons = (h_test != 0).sum(dim=1).float().mean().item()

        sparse_activations_path = output_dir / "h_sparse_test.pt"
        torch.save(h_test.cpu(), sparse_activations_path)
        logger.info(f"Saved sparse test activations to {sparse_activations_path}")

        # Compute ranking metrics (NDCG, Recall, MRR, etc.)
        logger.info("Computing ranking metrics...")
        evaluation_start = time.time()

        # ⭐ DIAGNOSTIC LOGGING (Phase 1)
        logger.info("\n📊 EVALUATION DIAGNOSTICS:")
        logger.info(
            f"Test set size: {X_test_csr.shape[0]} users × {X_test_csr.shape[1]} items"
        )
        logger.info(
            f"Test set sparsity: {(1.0 - X_test_csr.nnz / (X_test_csr.shape[0] * X_test_csr.shape[1])) * 100:.2f}%"
        )
        logger.info(f"Train users indices (first 5): {train_users[:5]}")
        logger.info(f"Test users indices (first 5): {test_users[:5]}")
        overlap_check = len(set(train_users) & set(test_users))
        logger.info(f"Train/test overlap: {overlap_check} users (should be 0)")
        if overlap_check > 0:
            logger.warning(
                f"⚠️  POTENTIAL DATA LEAKAGE: {overlap_check} users in both train and test!"
            )

        with torch.no_grad():
            # Build a sparse holdout ranking split: masked input vs held-out target items.
            holdout_ratio = config.get("evaluation", {}).get("holdout_ratio", 0.2)
            min_interactions = config.get("evaluation", {}).get("min_interactions", 5)
            X_eval_input_csr, X_eval_target_csr = build_holdout_split_sparse(
                X_test_csr,
                holdout_ratio=holdout_ratio,
                min_interactions=min_interactions,
                seed=config["data"]["seed"],
            )

            logger.info(
                "Evaluating ranking on held-out test items (per-user item holdout, masked input)"
            )

            # === ELSA ALONE EVALUATION ===
            logger.info("\nEvaluating ELSA-only on held-out test items...")
            ranking_metrics_elsa, n_eval_users = evaluate_recommendations_batched(
                X_eval_input_csr,
                X_eval_target_csr,
                lambda batch: _score_elsa_batch(elsa_model, batch, device),
                ks=[5, 10, 20],
                batch_size=256,
            )

            # === SAE+ELSA EVALUATION ===
            logger.info("\nEvaluating SAE+ELSA on held-out test items...")
            ranking_metrics_sae, _ = evaluate_recommendations_batched(
                X_eval_input_csr,
                X_eval_target_csr,
                lambda batch: _score_sae_batch(elsa_model, sae_model, batch, device),
                ks=[5, 10, 20],
                batch_size=256,
            )

        coverage, entropy, latency_metrics = _compute_auxiliary_metrics(
            elsa_model,
            sae_model,
            X_eval_input_csr,
            device=device,
            batch_size=256,
            top_k=20,
        )

        # Add these to SAE+ELSA ranking metrics
        ranking_metrics_sae["coverage"] = coverage
        ranking_metrics_sae["entropy"] = entropy
        ranking_metrics_sae["latency"] = latency_metrics

        # ⭐ Print comparison table
        comparison_report = compare_model_performance(
            ranking_metrics_elsa, ranking_metrics_sae
        )
        logger.info("\n" + comparison_report)

        # Print detailed reports
        logger.info(
            "\nELSA-only metrics:\n" + print_evaluation_report(ranking_metrics_elsa)
        )
        logger.info(
            "\nSAE+ELSA metrics:\n" + print_evaluation_report(ranking_metrics_sae)
        )

        # Use SAE+ELSA metrics as primary ranking_metrics for saving
        ranking_metrics = ranking_metrics_sae

        evaluation_time = time.time() - evaluation_start
        logger.info(f"Ranking evaluation completed in {evaluation_time:.2f}s")

        logger.info(f"Test reconstruction loss: {test_recon_loss:.6f}")
        logger.info(f"Test cosine similarity: {test_cosine_sim:.4f}")
        logger.info(
            f"Average active neurons: {active_neurons:.1f}/{config['sae']['width_ratio'] * config['elsa']['latent_dim']}"
        )

        # Compute model sizes
        model_sizes = compute_model_sizes(elsa_model, sae_model, output_dir)

        # Compute sparsity stats on test set
        test_sparsity = compute_sparsity_stats(
            sae_model, Z_test, device, batch_size=256
        )

        test_user_artifacts = persist_test_user_artifacts(
            output_dir,
            reviews,
            train_user_ids,
            test_user_ids,
            top_k=50,
        )

        # Save comprehensive summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "data": {
                "n_users": int(n_users),
                "n_items": int(X_csr.shape[1]),
                "n_items_before_kcore": int(item_count_before_kcore),
                "n_interactions": int(X_csr.nnz),
                "n_train_users": int(len(train_users)),
                "n_test_users": int(len(test_users)),
                "sparsity_percent": float(
                    100.0 * (1.0 - X_csr.nnz / (X_csr.shape[0] * X_csr.shape[1]))
                ),
            },
            "elsa": {
                "best_val_loss": float(elsa_best_loss),
                "best_epoch": elsa_stats["best_epoch"],
                "final_epoch": elsa_stats["final_epoch"],
                "training_time_sec": elsa_stats["training_time_sec"],
                "early_stop_reason": elsa_stats["early_stop_reason"],
            },
            "sae": {
                "best_val_loss": float(sae_best_loss),
                "best_epoch": sae_stats["best_epoch"],
                "final_epoch": sae_stats["final_epoch"],
                "training_time_sec": sae_stats["training_time_sec"],
                "early_stop_reason": sae_stats["early_stop_reason"],
                "test_recon_loss": float(test_recon_loss),
                "test_cosine_sim": float(test_cosine_sim),
                "test_avg_active_neurons": float(active_neurons),
                "test_avg_active_neurons_detailed": test_sparsity["avg_active_neurons"],
                "test_max_active_neurons": test_sparsity["max_active_neurons"],
                "test_total_neurons": test_sparsity["total_neurons"],
                "test_sparsity_ratio": test_sparsity["sparsity_ratio"],
            },
            "ranking_metrics_elsa": ranking_metrics_elsa,  # ELSA-only on holdout
            "ranking_metrics_sae": ranking_metrics_sae,  # SAE+ELSA on holdout
            "ranking_metrics": ranking_metrics,  # Primary (same as ranking_metrics_sae)
            "model_sizes": model_sizes,
            "training": {
                "total_time_sec": elsa_stats["training_time_sec"]
                + sae_stats["training_time_sec"],
                "elsa_time_sec": elsa_stats["training_time_sec"],
                "sae_time_sec": sae_stats["training_time_sec"],
                "evaluation_time_sec": evaluation_time,
            },
            "output_dir": str(output_dir),
            "preprocessing": {
                "cache_key": preprocessing_cache_key,
                "cache_dir": str(shared_cache_dir),
                "source": preprocessing_source,
                "manifest_path": str(shared_preprocessing_manifest_path(shared_cache_dir)),
                "manifest": preprocessing_manifest,
            },
            "artifacts": {
                key: str(path) for key, path in test_user_artifacts.items()
            },
            "evaluation_protocol": {
                "split": "per-user holdout",
                "holdout_ratio": holdout_ratio,
                "min_interactions": min_interactions,
                "metric_input": "masked user history",
                "metric_target": "held-out interactions",
            },
        }

        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        # � SAVE DATA FOR NEURON LABELING REPRODUCIBILITY
        logger.info("\nSaving data files for neuron labeling...")
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Save the processed training CSR used by the model pipeline
            with open(data_dir / "processed_train.pkl", "wb") as f:
                pickle.dump(X_train_csr, f)
            logger.info(
                f"✅ Saved processed_train to {data_dir / 'processed_train.pkl'}"
            )

            # Save filtered reviews DataFrame
            with open(data_dir / "reviews_df.pkl", "wb") as f:
                pickle.dump(reviews, f)
            logger.info(
                f"✅ Saved reviews_df ({len(reviews)} rows) to {data_dir / 'reviews_df.pkl'}"
            )

            # Save item mapping (k-core filtered)
            with open(data_dir / "item_map_after_kcore.pkl", "wb") as f:
                pickle.dump(item_map_after_kcore, f)
            logger.info(
                f"✅ Saved item_map_after_kcore ({len(item_map_after_kcore)} items) to {data_dir / 'item_map_after_kcore.pkl'}"
            )

            # Save business metadata for neuron labeling and UI interpretation
            if "business_metadata" not in locals():
                business_metadata = _build_business_metadata_from_db(
                    db_path=db_path,
                    item_map_after_kcore=item_map_after_kcore,
                    state_filter=config["data"].get("state_filter"),
                )

            with open(data_dir / "business_metadata.pkl", "wb") as f:
                pickle.dump(business_metadata, f)
            logger.info(
                f"✅ Saved business_metadata to {data_dir / 'business_metadata.pkl'}"
            )
        except Exception as e:
            logger.error(f"Failed to save data for neuron labeling: {e}")
            logger.warning("Continuing without saved data (neuron labeling may fail)")

        # 🔄 PRECOMPUTE USER CSR MATRICES (post-training step)
        # This enables fast user history lookup in the app without querying DB each time
        logger.info("\nPRECOMPUTATION PHASE:")
        try:
            precompute_user_csr_matrices(
                reviews,
                item_map_after_kcore,
                output_dir,
                upload_to_cloud=True,
                top_n_users=50,
                allowed_user_ids=test_user_ids,
            )
        except Exception as e:
            logger.error(f"User matrix precomputation failed: {e}")
            logger.warning(
                "Continuing without precomputed matrices (app will work but slower)"
            )

        # Upload results to cloud if configured
        timestamp = output_dir.name  # YYYYMMDD_HHMMSS format
        upload_results_to_cloud(output_dir, timestamp)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Total training time: {summary['training']['total_time_sec']:.1f}s"
        )
        logger.info(f"  ELSA: {summary['training']['elsa_time_sec']:.1f}s")
        logger.info(f"  SAE: {summary['training']['sae_time_sec']:.1f}s")
        logger.info(f"  Evaluation: {summary['training']['evaluation_time_sec']:.1f}s")
        logger.info(
            f"Model sizes: Total {model_sizes['total_mb']:.2f}MB "
            f"(ELSA {model_sizes['elsa_mb']:.2f}MB, SAE {model_sizes['sae_mb']:.2f}MB)"
        )
        logger.info(f"\nRanking Metrics Summary:")
        for metric_name in ["ndcg", "recall", "precision", "mrr"]:
            if metric_name in ranking_metrics:
                values = ranking_metrics[metric_name]
                logger.info(
                    f"  {metric_name.upper()}: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in sorted(values.items()))
                )
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Summary saved to: {summary_path}")

        # Register run as completed
        final_metrics = {
            "elsa_epochs": config["elsa"]["num_epochs"],
            "sae_epochs": config["sae"]["num_epochs"],
            "final_output_dir": str(output_dir),
        }
        registry.update_run_status(run_id, "train", "completed", final_metrics)
        write_latest_run_pointer(output_dir)
        logger.info(f"✓ Run {run_id} registered as completed")

    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        # Register run as failed
        registry.update_run_status(run_id, "train", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()

