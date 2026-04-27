"""Evaluation entry point for SAE-CF POI recommender.

Loads trained models and evaluates on test/validation sets.
Computes Recall@K, NDCG@K, HR@K, and other metrics.

Usage
-----
    python -m src.evaluate

    # Evaluate a specific run
    python -m src.evaluate --checkpoint outputs/20240316_120000/checkpoints --split test
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from src.data.shared_preprocessing_cache import (
    prepare_shared_preprocessing_cache,
    shared_preprocessing_manifest_path,
)
from src.models.collaborative_filtering import ELSA
from src.models.sparse_autoencoder import TopKSAE
from src.run_registry import RunRegistry
from src.utils import CheckpointManager, set_global_reproducibility, setup_logger
from src.utils.evaluation import (
    benchmark_inference,
    build_holdout_split_sparse,
    compare_metric_dicts,
    compare_model_performance,
    compute_holdout_diagnostics,
    compute_score_diagnostics,
    evaluate_recommendations_batched,
    load_holdout_split_artifacts,
    print_evaluation_report,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SAE-CF POI recommender")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to checkpoint directory (contains summary.json). If omitted, the latest trained run is used.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="K values for metrics. Defaults to the saved evaluation protocol for the run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--sync-summary",
        action="store_true",
        help="Overwrite the run summary with refreshed evaluation metrics after validation.",
    )
    parser.add_argument(
        "--sync-experiment-manifest",
        action="store_true",
        help="Refresh the local experiment manifest (and cloud copy if configured) after syncing the run summary.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow reconstructed fallback evaluation when exact replay artifacts are missing. "
        "Without this flag, evaluation fails loudly for non-replayable older runs.",
    )
    return parser.parse_args()


def resolve_checkpoint_dir(checkpoint_arg: str | None) -> Path:
    """Resolve the checkpoint directory, defaulting to the latest trained run."""
    if checkpoint_arg:
        checkpoint_dir = Path(checkpoint_arg)
        if (
            checkpoint_dir.name != "checkpoints"
            and (checkpoint_dir / "checkpoints").exists()
        ):
            checkpoint_dir = checkpoint_dir / "checkpoints"
        return checkpoint_dir

    registry = RunRegistry()
    latest_train_runs = registry.get_runs_by_stage("train")
    if not latest_train_runs:
        raise FileNotFoundError(
            "No completed training run found. Run training first or pass --checkpoint."
        )

    output_dir = Path.cwd() / "outputs" / latest_train_runs[0]
    checkpoint_dir = output_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found for latest training run: {checkpoint_dir}"
        )

    logger.info(f"Auto-detected latest trained run: {latest_train_runs[0]}")
    return checkpoint_dir


def _load_user_id_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    return [str(item) for item in data]


def _load_saved_protocol(
    output_dir: Path, summary: dict, k_values_arg
) -> tuple[dict, list[int]]:
    data_dir = output_dir / "data"
    protocol_path = data_dir / "evaluation_protocol.json"
    protocol = {}
    if protocol_path.exists():
        try:
            with protocol_path.open("r", encoding="utf-8") as f:
                protocol = json.load(f) or {}
        except Exception as e:
            logger.warning(
                "Failed to load saved evaluation protocol %s: %s", protocol_path, e
            )

    if not protocol:
        protocol = dict(summary.get("evaluation_protocol") or {})

    ks = list(k_values_arg) if k_values_arg else list(protocol.get("k_values") or [])
    if not ks:
        metric_dict = (
            summary.get("ranking_metrics_sae") or summary.get("ranking_metrics") or {}
        )
        ndcg_metrics = (
            metric_dict.get("ndcg", {}) if isinstance(metric_dict, dict) else {}
        )
        inferred_ks = []
        for key in ndcg_metrics.keys():
            if isinstance(key, str) and key.startswith("@"):
                try:
                    inferred_ks.append(int(key[1:]))
                except ValueError:
                    continue
        if inferred_ks:
            ks = sorted(inferred_ks)
    if not ks:
        ks = list(
            summary.get("config", {})
            .get("evaluation", {})
            .get("k_values", [5, 10, 20, 50])
        )

    protocol.setdefault("split", "per-user holdout")
    protocol.setdefault("holdout_ratio", 0.2)
    protocol.setdefault("min_interactions", 5)
    protocol["k_values"] = ks
    return protocol, ks


def _resolve_split_from_artifacts(
    output_dir: Path,
    final_user_ids: list[str],
    split: str,
    seed: int,
    *,
    train_test_ratio: float,
    val_ratio: float,
):
    data_dir = output_dir / "data"
    train_user_ids = _load_user_id_list(data_dir / "train_user_ids.json")
    test_user_ids = _load_user_id_list(data_dir / "test_user_ids.json")
    val_user_ids = _load_user_id_list(data_dir / "val_user_ids.json")

    if split == "val" and not val_user_ids and train_user_ids:
        train_indices = np.arange(len(train_user_ids))
        _, val_idx = train_test_split(
            train_indices,
            test_size=val_ratio,
            random_state=seed,
        )
        val_user_ids = [train_user_ids[idx] for idx in val_idx]

    wanted_user_ids = test_user_ids if split == "test" else val_user_ids
    source = "persisted_split_artifacts"

    if not wanted_user_ids:
        source = "recomputed_split_fallback"
        n_users = len(final_user_ids)
        user_indices = np.arange(n_users)
        train_users, test_users = train_test_split(
            user_indices,
            test_size=1 - train_test_ratio,
            random_state=seed,
        )
        if split == "test":
            return np.array(test_users), source
        train_idx = np.arange(len(train_users))
        _, val_idx = train_test_split(
            train_idx,
            test_size=val_ratio,
            random_state=seed,
        )
        return np.array(train_users)[val_idx], source

    user_to_index = {str(user_id): idx for idx, user_id in enumerate(final_user_ids)}
    split_indices = [
        user_to_index[user_id]
        for user_id in wanted_user_ids
        if user_id in user_to_index
    ]
    missing = len(wanted_user_ids) - len(split_indices)
    if missing:
        logger.warning(
            "Could not map %d persisted user IDs onto the cached dataset; evaluation may be incomplete.",
            missing,
        )
    return np.array(split_indices, dtype=int), source


def _flatten_metric_values(metrics: dict, prefix: str = "") -> dict[str, float]:
    flattened = {}
    if not isinstance(metrics, dict):
        return flattened
    for key, value in metrics.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_metric_values(value, full_key))
        elif isinstance(value, (int, float, np.floating)):
            flattened[full_key] = float(value)
    return flattened


def _log_metric_drift(stored: dict, refreshed: dict, label: str) -> None:
    stored_flat = _flatten_metric_values(stored)
    refreshed_flat = _flatten_metric_values(refreshed)
    if not stored_flat:
        logger.info("No stored %s metrics found in summary for comparison.", label)
        return

    diffs = []
    for key, refreshed_value in refreshed_flat.items():
        if key not in stored_flat:
            continue
        diff = abs(stored_flat[key] - refreshed_value)
        diffs.append((key, diff, stored_flat[key], refreshed_value))

    significant = [row for row in diffs if row[1] > 1e-6]
    if not significant:
        logger.info("Refreshed %s metrics match the stored training summary.", label)
        return

    significant.sort(key=lambda row: row[1], reverse=True)
    top_key, top_diff, old_value, new_value = significant[0]
    logger.warning(
        "Detected drift in %s metrics vs stored summary. Largest difference: %s (stored=%.6f, refreshed=%.6f, |Δ|=%.6f)",
        label,
        top_key,
        old_value,
        new_value,
        top_diff,
    )


def _sync_summary(
    summary_path: Path,
    summary: dict,
    metrics_elsa: dict,
    metrics_sae: dict,
    protocol: dict,
    results_path: Path,
    split: str,
    *,
    replay_mode: str,
    drift_elsa: dict,
    drift_sae: dict,
) -> None:
    if split != "test":
        logger.info(
            "Skipping summary sync for %s split; training summary remains canonical for test metrics.",
            split,
        )
        return

    summary["ranking_metrics_elsa"] = metrics_elsa
    summary["ranking_metrics_sae"] = metrics_sae
    summary["ranking_metrics"] = metrics_sae
    summary["evaluation_protocol"] = protocol
    summary["posthoc_evaluation"] = {
        "results_path": str(results_path),
        "synced_at": datetime.now().isoformat(),
        "source": "src.evaluate",
        "mode": replay_mode,
        "drift": {
            "elsa_only": drift_elsa,
            "sae_plus_elsa": drift_sae,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Synced refreshed test metrics into %s", summary_path)


def _sync_experiment_manifest(output_dir: Path, summary: dict) -> None:
    experiment_cfg = summary.get("config", {}).get("experiment") or {}
    experiment_id = experiment_cfg.get("experiment_id")
    if not experiment_id:
        logger.info(
            "Run is not part of an experiment sweep; skipping experiment manifest sync."
        )
        return

    manifest_path = (
        Path("outputs") / "experiments" / str(experiment_id) / "manifest.json"
    )
    if not manifest_path.exists():
        logger.warning("Experiment manifest not found for sync: %s", manifest_path)
        return

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    run_dir_str = str(output_dir)
    updated = False
    for run in manifest.get("runs", []):
        if run.get("run_dir") == run_dir_str:
            run["summary"] = summary
            run["summary_path"] = str(output_dir / "summary.json")
            updated = True
            break

    if not updated:
        logger.warning(
            "Run %s not found in experiment manifest %s", run_dir_str, manifest_path
        )
        return

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Updated experiment manifest: %s", manifest_path)

    try:
        from src.ui.services.secrets_helper import get_cloud_storage_bucket

        gcs_bucket_name = get_cloud_storage_bucket()
        if gcs_bucket_name:
            from src.ui.services.cloud_storage_helper import CloudStorageHelper

            cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
            cloud_storage.upload_json(
                manifest_path,
                f"experiments/{experiment_id}/manifest.json",
                metadata={
                    "timestamp": str(experiment_id),
                    "type": "experiment_manifest",
                },
            )
            logger.info(
                "Uploaded refreshed experiment manifest to GCS for %s", experiment_id
            )
    except Exception as e:
        logger.warning("Failed to upload refreshed experiment manifest to GCS: %s", e)


def _score_elsa_batch(elsa: ELSA, batch: np.ndarray, device: str) -> np.ndarray:
    """Score a dense user batch with ELSA and convert to residual scores."""
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = elsa.decode(elsa.encode(batch_tensor)).cpu().numpy()
    scores = scores - batch
    scores[batch > 0] = -np.inf
    return scores


def _score_sae_batch(
    elsa: ELSA, sae: TopKSAE, batch: np.ndarray, device: str
) -> np.ndarray:
    """Score a dense user batch with the exact training-time SAE+ELSA path."""
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = (
            elsa.decode(sae.decode(sae.encode(elsa.encode(batch_tensor)))).cpu().numpy()
        )
    scores = scores - batch
    scores[batch > 0] = -np.inf
    return scores


def _compute_auxiliary_metrics(
    elsa: ELSA,
    sae: TopKSAE,
    X_input_csr,
    X_target_csr,
    *,
    device: str,
    batch_size: int,
    top_k: int,
) -> tuple[float, float, dict[str, float]]:
    """Compute coverage, entropy, and a latency benchmark without densifying the full matrix."""
    n_items = X_target_csr.shape[1]
    item_counts = np.zeros(n_items, dtype=np.int64)
    total_recommendations = 0
    latency_sample_scores: list[np.ndarray] = []
    collected_latency_users = 0
    target_latency_users = min(100, X_input_csr.shape[0])

    for start in range(0, X_input_csr.shape[0], batch_size):
        end = min(start + batch_size, X_input_csr.shape[0])
        batch = X_input_csr[start:end].toarray().astype(np.float32)
        scores = _score_sae_batch(elsa, sae, batch, device)

        top_items = np.argsort(-scores, axis=1)[:, :top_k]
        for row in top_items:
            item_counts[row] += 1
        total_recommendations += top_items.size

        if collected_latency_users < target_latency_users:
            take = min(scores.shape[0], target_latency_users - collected_latency_users)
            latency_sample_scores.append(scores[:take])
            collected_latency_users += take

    coverage = float(np.count_nonzero(item_counts) / n_items) if n_items > 0 else 0.0

    if total_recommendations > 0:
        probs = item_counts[item_counts > 0] / total_recommendations
        entropy = (
            float(-np.sum(probs * np.log(probs)) / np.log(n_items))
            if n_items > 1
            else 0.0
        )
    else:
        entropy = 0.0

    latency_metrics = {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    if latency_sample_scores:
        latency_metrics = benchmark_inference(
            np.vstack(latency_sample_scores),
            n_samples=min(100, collected_latency_users),
        )

    return coverage, entropy, latency_metrics


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    # Load checkpoint and config
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Find summary.json in parent directory
    output_dir = checkpoint_dir.parent
    summary_path = output_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r") as f:
        summary = json.load(f)

    config = summary["config"]

    seed = int(config["data"]["seed"])
    reproducibility = set_global_reproducibility(seed)

    # Set up logging
    setup_logger(__name__, log_dir=output_dir, level=logging.INFO)
    logger.info(f"Evaluating checkpoint from {output_dir}")
    logger.info(
        "Reproducibility configured for evaluation: seed=%d, cudnn_deterministic=%s, deterministic_algorithms=%s",
        reproducibility["seed"],
        reproducibility["cudnn_deterministic"],
        reproducibility["deterministic_algorithms_enabled"],
    )

    # Extract run_id from output directory and initialize registry
    run_id = output_dir.name
    registry = RunRegistry()
    registry.register_run(run_id, "evaluate", config={}, status="pending")

    device = config["evaluation"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "CUDA was requested by the saved config, but this PyTorch build does not have CUDA. Falling back to CPU."
        )
        device = "cpu"
    logger.info(f"Using device: {device}")

    try:
        # Load data
        logger.info("=" * 60)
        logger.info(f"LOADING {args.split.upper()} DATA")
        logger.info("=" * 60)
        (
            preprocessing_payload,
            preprocessing_source,
            shared_cache_dir,
        ) = prepare_shared_preprocessing_cache(config, require_existing=False)
        X_csr = preprocessing_payload["final_dataset"].csr
        final_user_ids = [
            str(user_id) for user_id in preprocessing_payload["final_user_ids"]
        ]
        logger.info(
            "Using shared preprocessing cache (%s): %s",
            preprocessing_source,
            shared_cache_dir,
        )
        logger.info("Built CSR: %d users x %d items", X_csr.shape[0], X_csr.shape[1])

        protocol, k_values = _load_saved_protocol(output_dir, summary, args.k_values)
        split_indices, split_source = _resolve_split_from_artifacts(
            output_dir,
            final_user_ids,
            args.split,
            seed,
            train_test_ratio=float(config["data"]["train_test_split"]),
            val_ratio=float(config["data"]["val_split"]),
        )
        X_split_csr = X_csr[split_indices]

        logger.info(
            "Evaluation set: %d users x %d items (%s)",
            X_split_csr.shape[0],
            X_split_csr.shape[1],
            split_source,
        )

        # Load models
        logger.info("Loading models...")

        checkpoint_mgr = CheckpointManager(checkpoint_dir)

        # ELSA
        elsa = ELSA(
            n_items=X_csr.shape[1],
            latent_dim=config["elsa"]["latent_dim"],
        ).to(device)

        elsa_info = checkpoint_mgr.load(
            elsa, checkpoint_name="elsa_best", device=device
        )
        logger.info(f"Loaded ELSA model from epoch {elsa_info['epoch']}")

        # SAE
        sae_name = f"sae_r{config['sae']['width_ratio']}_k{config['sae']['k']}_best"
        sae = TopKSAE(
            input_dim=config["elsa"]["latent_dim"],
            hidden_dim=config["sae"]["width_ratio"] * config["elsa"]["latent_dim"],
            k=config["sae"]["k"],
            l1_coef=float(config["sae"]["l1_coef"]),
        ).to(device)

        sae_info = checkpoint_mgr.load(sae, checkpoint_name=sae_name, device=device)
        logger.info(f"Loaded SAE model from epoch {sae_info['epoch']}")

        # Evaluate
        logger.info("=" * 60)
        logger.info("COMPUTING HOLDOUT METRICS")
        logger.info("=" * 60)

        holdout_ratio = float(protocol.get("holdout_ratio", 0.2))
        min_interactions = int(protocol.get("min_interactions", 5))
        replay_mode = "reconstructed_fallback"
        holdout_metadata: dict = {}
        exact_holdout = load_holdout_split_artifacts(
            output_dir / "data", split=args.split
        )
        if exact_holdout is not None:
            X_eval_input_csr, X_eval_target_csr, holdout_metadata = exact_holdout
            replay_mode = "exact_replay"
            logger.info(
                "Using persisted %s holdout replay artifacts from %s",
                args.split,
                output_dir / "data",
            )
        else:
            if not args.allow_fallback:
                raise RuntimeError(
                    f"Exact {args.split} holdout replay artifacts are missing for run {output_dir.name}. "
                    "This run predates deterministic replay support, so `src.evaluate` cannot verify it faithfully. "
                    "Use the canonical metrics in summary.json, retrain the run with the patched pipeline, "
                    "or rerun with --allow-fallback if you explicitly want non-canonical reconstructed diagnostics."
                )
            X_eval_input_csr, X_eval_target_csr = build_holdout_split_sparse(
                X_split_csr,
                holdout_ratio=holdout_ratio,
                min_interactions=min_interactions,
                seed=config["data"]["seed"],
            )
            logger.warning(
                "Exact %s holdout replay artifacts are missing; falling back to reconstructed holdout.",
                args.split,
            )

        holdout_diagnostics = compute_holdout_diagnostics(
            X_eval_input_csr,
            X_eval_target_csr,
            min_interactions=min_interactions,
        )
        logger.info(
            "Holdout diagnostics: %d split users, %d evaluated users, avg held-out items %.2f, skipped %.2f%%",
            holdout_diagnostics["n_split_users"],
            holdout_diagnostics["n_eval_users"],
            holdout_diagnostics["avg_heldout_items_per_user"],
            holdout_diagnostics["pct_skipped_users"] * 100.0,
        )
        if holdout_diagnostics["n_eval_users"] == 0:
            raise RuntimeError(
                "Evaluation produced zero effective users. Check holdout artifacts, min_interactions, and saved test split."
            )
        if holdout_diagnostics["n_eval_users"] < 10:
            raise RuntimeError(
                f"Evaluation only has {holdout_diagnostics['n_eval_users']} effective users; refusing to report unstable metrics."
            )

        logger.info(
            "Evaluating on per-user holdout split: masked input vs held-out target items"
        )

        # ELSA-only baseline
        logger.info("Evaluating ELSA-only baseline...")
        metrics_elsa, n_eval_users = evaluate_recommendations_batched(
            X_eval_input_csr,
            X_eval_target_csr,
            lambda batch: _score_elsa_batch(elsa, batch, device),
            ks=k_values,
            batch_size=args.batch_size,
        )

        # SAE+ELSA model
        logger.info("Evaluating SAE+ELSA model...")
        metrics_sae, _ = evaluate_recommendations_batched(
            X_eval_input_csr,
            X_eval_target_csr,
            lambda batch: _score_sae_batch(elsa, sae, batch, device),
            ks=k_values,
            batch_size=args.batch_size,
        )

        elsa_score_diag = compute_score_diagnostics(
            lambda batch: _score_elsa_batch(elsa, batch, device),
            X_eval_input_csr,
            X_eval_target_csr,
            top_k=max(k_values),
        )
        sae_score_diag = compute_score_diagnostics(
            lambda batch: _score_sae_batch(elsa, sae, batch, device),
            X_eval_input_csr,
            X_eval_target_csr,
            top_k=max(k_values),
        )
        logger.info(
            "ELSA score diagnostics: finite_fraction=%.4f mean=%.4f std=%.4f range=[%.4f, %.4f]",
            elsa_score_diag["finite_fraction"],
            elsa_score_diag["score_mean"],
            elsa_score_diag["score_std"],
            elsa_score_diag["score_min"],
            elsa_score_diag["score_max"],
        )
        logger.info(
            "SAE+ELSA score diagnostics: finite_fraction=%.4f mean=%.4f std=%.4f range=[%.4f, %.4f]",
            sae_score_diag["finite_fraction"],
            sae_score_diag["score_mean"],
            sae_score_diag["score_std"],
            sae_score_diag["score_min"],
            sae_score_diag["score_max"],
        )

        # Compare models with explicit labels
        comparison_report = compare_model_performance(metrics_elsa, metrics_sae)

        # Log results
        logger.info("=" * 60)
        logger.info(f"RESULTS ON {args.split.upper()} SET (MASKED HOLDOUT)")
        logger.info("=" * 60)

        logger.info("\nELSA-only metrics:\n" + print_evaluation_report(metrics_elsa))
        logger.info("\nSAE+ELSA metrics:\n" + print_evaluation_report(metrics_sae))
        logger.info("\n" + comparison_report)

        # Additional diversity and latency metrics for the SAE+ELSA model
        coverage, entropy, latency_metrics = _compute_auxiliary_metrics(
            elsa,
            sae,
            X_eval_input_csr,
            X_eval_target_csr,
            device=device,
            batch_size=args.batch_size,
            top_k=max(k_values),
        )
        metrics_sae["coverage"] = coverage
        metrics_sae["entropy"] = entropy
        metrics_sae["latency"] = latency_metrics

        drift_elsa = compare_metric_dicts(
            summary.get("ranking_metrics_elsa", {}),
            metrics_elsa,
        )
        drift_sae = compare_metric_dicts(
            summary.get("ranking_metrics_sae", {}),
            metrics_sae,
        )
        if drift_elsa["matches"]:
            logger.info(
                "Refreshed ELSA-only metrics match the stored training summary."
            )
        else:
            largest = drift_elsa["largest_diff"]
            logger.warning(
                "Detected drift in ELSA-only metrics vs stored summary. Largest difference: %s (stored=%.6f, refreshed=%.6f, |Δ|=%.6f)",
                largest["metric"],
                largest["stored"],
                largest["refreshed"],
                largest["abs_diff"],
            )
        if drift_sae["matches"]:
            logger.info("Refreshed SAE+ELSA metrics match the stored training summary.")
        else:
            largest = drift_sae["largest_diff"]
            logger.warning(
                "Detected drift in SAE+ELSA metrics vs stored summary. Largest difference: %s (stored=%.6f, refreshed=%.6f, |Δ|=%.6f)",
                largest["metric"],
                largest["stored"],
                largest["refreshed"],
                largest["abs_diff"],
            )

        # Save results
        results = {
            "split": args.split,
            "mode": replay_mode,
            "evaluation_protocol": {
                **protocol,
                "metric_input": "masked user history",
                "metric_target": "held-out interactions",
            },
            "metrics_by_model": {
                "elsa_only": metrics_elsa,
                "sae_plus_elsa": metrics_sae,
            },
            "metrics": metrics_sae,
            "config": config,
            "preprocessing": {
                "source": preprocessing_source,
                "cache_dir": str(shared_cache_dir),
                "manifest_path": str(
                    shared_preprocessing_manifest_path(shared_cache_dir)
                ),
            },
            "split_artifacts": {
                "source": split_source,
                "replay_mode": replay_mode,
                "data_dir": str(output_dir / "data"),
                "n_split_users": int(X_split_csr.shape[0]),
            },
            "holdout": {
                "metadata": holdout_metadata,
                "diagnostics": holdout_diagnostics,
            },
            "score_diagnostics": {
                "elsa_only": elsa_score_diag,
                "sae_plus_elsa": sae_score_diag,
            },
            "drift": {
                "elsa_only": drift_elsa,
                "sae_plus_elsa": drift_sae,
            },
            "reproducibility": {
                **reproducibility,
                "device": device,
                "holdout_split_seed": seed,
            },
        }

        results_path = output_dir / f"evaluation_{args.split}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

        if args.sync_summary and replay_mode != "exact_replay":
            logger.warning(
                "Skipping summary sync because evaluation ran in %s mode. Exact replay is required before overwriting canonical metrics.",
                replay_mode,
            )
        elif args.sync_summary:
            _sync_summary(
                summary_path,
                summary,
                metrics_elsa,
                metrics_sae,
                results["evaluation_protocol"],
                results_path,
                args.split,
                replay_mode=replay_mode,
                drift_elsa=drift_elsa,
                drift_sae=drift_sae,
            )
            if args.sync_experiment_manifest:
                _sync_experiment_manifest(output_dir, summary)

        # Register evaluation as completed
        eval_metadata = {
            "split": args.split,
            "n_users_evaluated": n_eval_users,
            "model": "SAE+ELSA",
            "split_source": split_source,
            "replay_mode": replay_mode,
            **{
                f"elsa_only_{key.replace('@', '_at_').replace('.', '_')}": value
                for key, value in _flatten_metric_values(metrics_elsa).items()
            },
            **{
                f"sae_plus_elsa_{key.replace('@', '_at_').replace('.', '_')}": value
                for key, value in _flatten_metric_values(metrics_sae).items()
            },
        }
        registry.update_run_status(run_id, "evaluate", "completed", eval_metadata)
        logger.info(f"✓ Run {run_id} registered as evaluated")

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        registry.update_run_status(run_id, "evaluate", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
