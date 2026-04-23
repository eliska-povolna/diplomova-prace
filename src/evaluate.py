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
from pathlib import Path

import numpy as np
import torch

from src.data.shared_preprocessing_cache import (
    prepare_shared_preprocessing_cache,
    shared_preprocessing_manifest_path,
)
from src.models.collaborative_filtering import ELSA
from src.models.sae_cf_model import ELSASAEModel
from src.models.sparse_autoencoder import TopKSAE
from src.run_registry import RunRegistry
from src.utils import CheckpointManager, setup_logger
from src.utils.evaluation import (
    benchmark_inference,
    build_holdout_split_sparse,
    compare_model_performance,
    compute_coverage,
    evaluate_recommendations_batched,
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
        default=[10, 20, 50],
        help="K values for metrics",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation",
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


def _score_elsa_batch(elsa: ELSA, batch: np.ndarray, device: str) -> np.ndarray:
    """Score a dense user batch with ELSA and convert to residual scores."""
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = elsa(batch_tensor).cpu().numpy()
    scores = scores - batch
    scores[batch > 0] = -np.inf
    return scores


def _score_sae_batch(model: ELSASAEModel, batch: np.ndarray, device: str) -> np.ndarray:
    """Score a dense user batch with the combined SAE+ELSA model."""
    batch_tensor = torch.tensor(batch, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = model.recommend(batch_tensor).cpu().numpy()
    scores = scores - batch
    scores[batch > 0] = -np.inf
    return scores


def _compute_auxiliary_metrics(
    model: ELSASAEModel,
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
        scores = _score_sae_batch(model, batch, device)

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

    # Set up logging
    setup_logger(__name__, log_dir=output_dir, level=logging.INFO)
    logger.info(f"Evaluating checkpoint from {output_dir}")

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
        preprocessing_payload, preprocessing_source, shared_cache_dir = (
            prepare_shared_preprocessing_cache(config, require_existing=False)
        )
        reviews = preprocessing_payload["reviews"]
        X_csr = preprocessing_payload["final_dataset"].csr
        logger.info(
            "Using shared preprocessing cache (%s): %s",
            preprocessing_source,
            shared_cache_dir,
        )
        logger.info("Built CSR: %d users x %d items", X_csr.shape[0], X_csr.shape[1])

        # Get appropriate split
        from sklearn.model_selection import train_test_split

        n_users = X_csr.shape[0]
        user_indices = np.arange(n_users)
        train_users, test_users = train_test_split(
            user_indices,
            test_size=1 - config["data"]["train_test_split"],
            random_state=config["data"]["seed"],
        )

        if args.split == "test":
            X_split_csr = X_csr[test_users]
        else:  # val
            X_train_csr = X_csr[train_users]
            train_indices = np.arange(X_train_csr.shape[0])
            train_idx, val_idx = train_test_split(
                train_indices,
                test_size=config["data"]["val_split"],
                random_state=config["data"]["seed"],
            )
            X_split_csr = X_train_csr[val_idx]

        logger.info(
            f"Evaluation set: {X_split_csr.shape[0]} users × {X_split_csr.shape[1]} items"
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

        # Combined model
        model = ELSASAEModel(
            n_items=X_csr.shape[1],
            latent_dim=config["elsa"]["latent_dim"],
            sae_hidden_dim=config["sae"]["width_ratio"] * config["elsa"]["latent_dim"],
            k=config["sae"]["k"],
            l1_coef=float(config["sae"]["l1_coef"]),
        ).to(device)

        # Copy loaded weights
        model.elsa.load_state_dict(elsa.state_dict())
        model.sae.load_state_dict(sae.state_dict())
        model.eval()

        # Evaluate
        logger.info("=" * 60)
        logger.info("COMPUTING HOLDOUT METRICS")
        logger.info("=" * 60)

        holdout_ratio = config.get("evaluation", {}).get("holdout_ratio", 0.2)
        min_interactions = config.get("evaluation", {}).get("min_interactions", 5)

        X_eval_input_csr, X_eval_target_csr = build_holdout_split_sparse(
            X_split_csr,
            holdout_ratio=holdout_ratio,
            min_interactions=min_interactions,
            seed=config["data"]["seed"],
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
            ks=args.k_values,
            batch_size=args.batch_size,
        )

        # SAE+ELSA model
        logger.info("Evaluating SAE+ELSA model...")
        metrics_sae, _ = evaluate_recommendations_batched(
            X_eval_input_csr,
            X_eval_target_csr,
            lambda batch: _score_sae_batch(model, batch, device),
            ks=args.k_values,
            batch_size=args.batch_size,
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
            model,
            X_eval_input_csr,
            X_eval_target_csr,
            device=device,
            batch_size=args.batch_size,
            top_k=max(args.k_values),
        )
        metrics_sae["coverage"] = coverage
        metrics_sae["entropy"] = entropy
        metrics_sae["latency"] = latency_metrics

        # Save results
        results = {
            "split": args.split,
            "evaluation_protocol": {
                "split": "per-user holdout",
                "holdout_ratio": holdout_ratio,
                "min_interactions": min_interactions,
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
        }

        results_path = output_dir / f"evaluation_{args.split}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

        # Register evaluation as completed
        eval_metadata = {
            "split": args.split,
            "n_users_evaluated": n_eval_users,
            "model": "SAE+ELSA",
            **{
                f"elsa_only_{str(k).replace('@', '_at_')}": float(v)
                for k, v in metrics_elsa.items()
                if isinstance(v, (int, float, np.floating))
            },
            **{
                f"sae_plus_elsa_{str(k).replace('@', '_at_')}": float(v)
                for k, v in metrics_sae.items()
                if isinstance(v, (int, float, np.floating))
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

