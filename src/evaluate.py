"""Evaluation entry point for SAE-CF POI recommender.

Loads trained models and evaluates on test/validation sets.
Computes Recall@K, NDCG@K, HR@K, and other metrics.

Usage
-----
    python src/evaluate.py --checkpoint outputs/20240316_120000/checkpoints --split test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from src.data.preprocessing import build_csr
from src.data.yelp_loader import load_reviews
from src.models.collaborative_filtering import ELSA, ndcg_at_k, recall_at_k
from src.models.sae_cf_model import ELSASAEModel
from src.models.sparse_autoencoder import TopKSAE
from src.utils import CheckpointManager, setup_logger

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SAE-CF POI recommender")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to checkpoint directory (contains summary.json)",
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


def compute_metrics(
    scores: np.ndarray,
    gt_mask: np.ndarray,
    k_values: list[int],
) -> dict[str, float]:
    """Compute Recall@K, NDCG@K, HR@K for ranking metrics.

    Parameters
    ----------
    scores : np.ndarray
        Predicted scores of shape (n_users, n_items).
    gt_mask : np.ndarray
        Ground truth binary matrix of shape (n_users, n_items).
    k_values : list[int]
        List of k values to compute metrics for.

    Returns
    -------
    dict[str, float]
        Dictionary of computed metrics.
    """
    metrics = {}

    for k in k_values:
        # Get top-k items for each user
        top_indices = np.argsort(-scores, axis=1)[:, :k]

        # Hit rate: users with at least one relevant item in top-k
        hits = gt_mask[np.arange(gt_mask.shape[0])[:, None], top_indices].sum(axis=1)
        hr = (hits > 0).mean()
        metrics[f"HR@{k}"] = float(hr)

        # Recall@k
        recalls = []
        for i in range(scores.shape[0]):
            y_true = gt_mask[i]
            y_pred = top_indices[i]
            if y_true.sum() > 0:
                rec = recall_at_k(y_true, y_pred, k=k)
                recalls.append(rec)

        if recalls:
            metrics[f"Recall@{k}"] = float(np.nanmean(recalls))
        else:
            metrics[f"Recall@{k}"] = 0.0

        # NDCG@k
        ndcgs = []
        for i in range(scores.shape[0]):
            y_true = gt_mask[i]
            y_pred = top_indices[i]
            ndcg = ndcg_at_k(y_true, y_pred, k=k)
            ndcgs.append(ndcg)

        metrics[f"NDCG@{k}"] = float(np.nanmean(ndcgs))

    return metrics


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()

    # Load checkpoint and config
    checkpoint_dir = Path(args.checkpoint)
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

    device = config["evaluation"]["device"]
    logger.info(f"Using device: {device}")

    try:
        # Load data
        logger.info("=" * 60)
        logger.info(f"LOADING {args.split.upper()} DATA")
        logger.info("=" * 60)

        parquet_dir = Path(config["data"]["parquet_dir"])

        if not parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

        reviews = load_reviews(
            parquet_dir,
            db_path=config["data"]["db_path"],
            pos_threshold=config["data"]["pos_threshold"],
        )
        logger.info(f"Loaded {len(reviews)} reviews")

        # Build CSR matrix
        logger.info("Building CSR matrix...")
        dataset = build_csr(reviews)
        X_csr = dataset.csr
        logger.info(f"Built CSR: {X_csr.shape[0]} users × {X_csr.shape[1]} items")

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

        X_split = torch.tensor(X_split_csr.toarray(), dtype=torch.float32)
        gt_mask = X_split_csr.toarray().astype(bool)

        logger.info(
            f"Evaluation set: {X_split.shape[0]} users × {X_split.shape[1]} items"
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
            l1_coef=config["sae"]["l1_coef"],
        ).to(device)

        sae_info = checkpoint_mgr.load(sae, checkpoint_name=sae_name, device=device)
        logger.info(f"Loaded SAE model from epoch {sae_info['epoch']}")

        # Combined model
        model = ELSASAEModel(
            n_items=X_csr.shape[1],
            latent_dim=config["elsa"]["latent_dim"],
            sae_hidden_dim=config["sae"]["width_ratio"] * config["elsa"]["latent_dim"],
            k=config["sae"]["k"],
            l1_coef=config["sae"]["l1_coef"],
        ).to(device)

        # Copy loaded weights
        model.elsa.load_state_dict(elsa.state_dict())
        model.sae.load_state_dict(sae.state_dict())
        model.eval()

        # Evaluate
        logger.info("=" * 60)
        logger.info("COMPUTING METRICS")
        logger.info("=" * 60)

        all_scores = []

        with torch.no_grad():
            for i in range(0, X_split.shape[0], args.batch_size):
                x_batch = X_split[i : i + args.batch_size].to(device)
                scores = model.recommend(x_batch)
                all_scores.append(scores.cpu().numpy())

        all_scores = np.vstack(all_scores)
        logger.info(f"Computed scores for {all_scores.shape[0]} users")

        # Compute metrics
        metrics = compute_metrics(all_scores, gt_mask, args.k_values)

        # Log results
        logger.info("=" * 60)
        logger.info(f"RESULTS ON {args.split.upper()} SET")
        logger.info("=" * 60)

        for key, value in sorted(metrics.items()):
            logger.info(f"{key}: {value:.4f}")

        # Save results
        results = {
            "split": args.split,
            "metrics": metrics,
            "config": config,
        }

        results_path = output_dir / f"evaluation_{args.split}.json"
        with results_path.open("w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
