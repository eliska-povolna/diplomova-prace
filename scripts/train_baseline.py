"""Train baseline models (ALS & EASE) on preprocessed Yelp data with grid search.

This script trains simple but representable baseline recommender systems
for comparison with ELSA+SAE in the thesis evaluation.

Usage:
    python scripts/train_baseline.py --config scripts/baseline_gridsearch.yaml
    python scripts/train_baseline.py --config scripts/baseline_gridsearch.yaml --model als
    python scripts/train_baseline.py --config scripts/baseline_gridsearch.yaml --model ease
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy.sparse import load_npz, csr_matrix
from tqdm import tqdm

# Suppress implicit-package warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import implicit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ALSBaseline(nn.Module):
    """Alternating Least Squares baseline using implicit library.

    Fast matrix factorization using Weighted ALS algorithm.
    Good for implicit feedback data (like recommendation systems).
    """

    def __init__(
        self,
        n_items: int,
        factors: int = 64,
        regularization: float = 0.1,
        iterations: int = 15,
        use_gpu: bool = False,
    ):
        super().__init__()
        self.n_items = n_items
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.use_gpu = use_gpu
        self.model = None
        self.name = "ALS"

    def fit(self, X_csr: csr_matrix) -> dict[str, float]:
        """Fit ALS model to interaction matrix.

        Parameters
        ----------
        X_csr:
            Sparse CSR matrix of shape (n_users, n_items) with interaction counts.

        Returns
        -------
        dict[str, float]
            Training metadata (factors, reg, iterations).
        """
        logger.info(
            f"Fitting ALS: factors={self.factors}, reg={self.regularization}, "
            f"iter={self.iterations}, gpu={self.use_gpu}"
        )

        # Convert to COO format for implicit (implicit expects item-user matrix)
        X_coo = X_csr.tocoo()

        # Train model (implicit expects item-user matrix; pass transpose)
        self.model = implicit.als.AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            use_gpu=self.use_gpu,
            num_threads=1,
            random_state=42,
        )
        self.model.fit(X_csr.T)

        metadata = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }
        logger.info(f"✓ ALS training complete")
        return metadata

    def predict(self, X_csr: csr_matrix, n_recommendations: int = 100, mask_csr: csr_matrix = None) -> np.ndarray:
        """Generate recommendations using trained model.

        Parameters
        ----------
        X_csr:
            Sparse CSR matrix of shape (n_users, n_items) with user indices for prediction.
        n_recommendations:
            Unused (for compatibility). ALS.predict calculates all scores.
        mask_csr:
            Optional sparse CSR matrix to mask out (e.g., training interactions).
            If None, no masking is applied.

        Returns
        -------
        np.ndarray
            Predicted scores of shape (n_users, n_items).
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_users, n_items = X_csr.shape
        predicted_scores = np.zeros((n_users, n_items), dtype=np.float32)

        # Compute scores as: user_factors @ item_factors.T
        # Process in batches to avoid memory issues
        batch_size = 1024
        for start_idx in tqdm(range(0, n_users, batch_size), desc="Predicting (ALS)"):
            end_idx = min(start_idx + batch_size, n_users)
            # implicit may store factors with users/items swapped depending on how fit() was called.
            uf = self.model.user_factors
            itf = self.model.item_factors
            # Determine correct orientation
            if uf.shape[0] == n_users and itf.shape[0] == n_items:
                user_factors = uf[start_idx:end_idx]
                predicted_scores[start_idx:end_idx] = user_factors @ itf.T
            elif uf.shape[0] == n_items and itf.shape[0] == n_users:
                # swapped: item_factors are actually user factors and vice versa
                # compute scores by (item_factors_as_users)[start:end] @ (user_factors_as_items).T
                user_like = itf[start_idx:end_idx]
                predicted_scores[start_idx:end_idx] = user_like @ uf.T
            else:
                # fallback: try direct multiplication and hope shapes align
                user_factors = uf[start_idx:end_idx]
                predicted_scores[start_idx:end_idx] = user_factors @ itf.T

        # Optionally mask out interactions (e.g., training items when evaluating on same user)
        if mask_csr is not None:
            predicted_scores[mask_csr.nonzero()] = 0

        return predicted_scores

    def score_users(self, n_users: int, start_idx: int, end_idx: int) -> np.ndarray:
        """Score a batch of users without materializing the full user-item matrix."""
        n_items = self.n_items
        scores = np.zeros((end_idx - start_idx, n_items), dtype=np.float32)
        uf = self.model.user_factors
        itf = self.model.item_factors
        if uf.shape[0] == n_users and itf.shape[0] == n_items:
            scores[:] = uf[start_idx:end_idx] @ itf.T
        elif uf.shape[0] == n_items and itf.shape[0] == n_users:
            scores[:] = itf[start_idx:end_idx] @ uf.T
        else:
            scores[:] = uf[start_idx:end_idx] @ itf.T
        return scores

    def state_dict(self) -> dict[str, Any]:
        """Return model state for checkpoint saving."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return {
            "user_factors": self.model.user_factors,
            "item_factors": self.model.item_factors,
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load model state from checkpoint."""
        self.model = implicit.als.AlternatingLeastSquares(
            factors=state["factors"],
            regularization=state["regularization"],
            iterations=state["iterations"],
            use_gpu=self.use_gpu,
            num_threads=1,
        )
        self.model.user_factors = state["user_factors"]
        self.model.item_factors = state["item_factors"]


class BPRBaseline:
    """Bayesian Personalized Ranking (BPR) baseline using implicit library.

    Optimized for ranking metrics (NDCG, Recall) rather than reconstruction.
    Memory-efficient and much faster than EASE on large datasets.
    """

    def __init__(self, n_items: int, factors: int = 64, regularization: float = 0.01, iterations: int = 100):
        super().__init__()
        self.n_items = n_items
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.model = None
        self.name = "BPR"

    def fit(self, X_csr: csr_matrix) -> dict[str, float]:
        """Fit BPR model to interaction matrix.

        Parameters
        ----------
        X_csr:
            Sparse CSR matrix of shape (n_users, n_items) with implicit feedback (1/0).

        Returns
        -------
        dict[str, float]
            Training metadata (factors, regularization, iterations).
        """
        logger.info(
            f"Fitting BPR: factors={self.factors}, reg={self.regularization}, "
            f"iter={self.iterations}"
        )

        # BPR requires the matrix in COO format with explicit counts
        X_coo = X_csr.tocoo()

        # Train model using BPR (implicit expects item-user matrix; pass transpose)
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=42,
            num_threads=1,
            use_gpu=False,
        )
        self.model.fit(X_csr.T)

        metadata = {
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }
        logger.info(f"✓ BPR training complete")
        return metadata

    def predict(self, X_csr: csr_matrix, mask_csr: csr_matrix = None) -> np.ndarray:
        """Generate recommendations using trained BPR model.

        Parameters
        ----------
        X_csr:
            Sparse CSR matrix of shape (n_users, n_items) with user indices for prediction.
        mask_csr:
            Optional sparse CSR matrix to mask out (e.g., training interactions).
            If None, no masking is applied.

        Returns
        -------
        np.ndarray
            Predicted scores of shape (n_users, n_items).
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        n_users, n_items = X_csr.shape
        predicted_scores = np.zeros((n_users, n_items), dtype=np.float32)

        # BPR scoring: scores = user_factors @ item_factors.T
        # Compute in batches to avoid memory issues
        batch_size = 1024
        for start_idx in tqdm(range(0, n_users, batch_size), desc="Predicting (BPR)"):
            end_idx = min(start_idx + batch_size, n_users)
            uf = self.model.user_factors
            itf = self.model.item_factors
            if uf.shape[0] == n_users and itf.shape[0] == n_items:
                user_factors = uf[start_idx:end_idx]
                predicted_scores[start_idx:end_idx] = user_factors @ itf.T
            elif uf.shape[0] == n_items and itf.shape[0] == n_users:
                user_like = itf[start_idx:end_idx]
                predicted_scores[start_idx:end_idx] = user_like @ uf.T
            else:
                user_factors = uf[start_idx:end_idx]
                predicted_scores[start_idx:end_idx] = user_factors @ itf.T

        # Optionally mask out interactions (e.g., training items when evaluating on same user)
        if mask_csr is not None:
            predicted_scores[mask_csr.nonzero()] = 0

        return predicted_scores

    def score_users(self, n_users: int, start_idx: int, end_idx: int) -> np.ndarray:
        """Score a batch of users without materializing the full user-item matrix."""
        scores = np.zeros((end_idx - start_idx, self.n_items), dtype=np.float32)
        uf = self.model.user_factors
        itf = self.model.item_factors
        if uf.shape[0] == n_users and itf.shape[0] == self.n_items:
            scores[:] = uf[start_idx:end_idx] @ itf.T
        elif uf.shape[0] == self.n_items and itf.shape[0] == n_users:
            scores[:] = itf[start_idx:end_idx] @ uf.T
        else:
            scores[:] = uf[start_idx:end_idx] @ itf.T
        return scores


def compute_ranking_metrics_batched(
    model: Any,
    test_csr: csr_matrix,
    train_csr: csr_matrix | None = None,
    batch_size: int = 256,
    k_values: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute ranking metrics without building a full dense score matrix."""
    if k_values is None:
        k_values = [5, 10, 20, 50]

    metrics = {
        "recall": {},
        "ndcg": {},
        "precision": {},
        "mrr": {},
        "map": {},
    }

    n_users, n_items = test_csr.shape

    for k in k_values:
        recall_sum = 0.0
        ndcg_sum = 0.0
        precision_sum = 0.0
        mrr_sum = 0.0
        map_sum = 0.0
        valid_users = 0

        for start_idx in tqdm(range(0, n_users, batch_size), desc=f"Computing @{k}", disable=True):
            end_idx = min(start_idx + batch_size, n_users)
            if hasattr(model, "score_users"):
                batch_scores = model.score_users(n_users, start_idx, end_idx)
            else:
                raise ValueError("Model does not implement score_users().")

            if train_csr is not None:
                train_batch = train_csr[start_idx:end_idx]
                for local_user_idx in range(end_idx - start_idx):
                    train_items = train_batch[local_user_idx].indices
                    if len(train_items) > 0:
                        batch_scores[local_user_idx, train_items] = -np.inf

            topk = np.argpartition(batch_scores, -k, axis=1)[:, -k:]
            topk_scores = np.take_along_axis(batch_scores, topk, axis=1)
            order = np.argsort(-topk_scores, axis=1)
            topk = np.take_along_axis(topk, order, axis=1)

            for local_user_idx, top_k_indices in enumerate(topk):
                user_idx = start_idx + local_user_idx
                gt_indices = test_csr[user_idx].indices
                gt_items = set(gt_indices)

                if len(gt_items) == 0:
                    continue

                valid_users += 1
                hits = len(set(top_k_indices) & gt_items)
                recall_sum += hits / len(gt_items)
                precision_sum += hits / k

                dcg = 0.0
                for rank, item_idx in enumerate(top_k_indices):
                    if item_idx in gt_items:
                        dcg += 1.0 / np.log2(rank + 2)
                ideal_rel = min(len(gt_items), k)
                idcg = sum((1.0 / np.log2(i + 2)) for i in range(ideal_rel))
                if idcg > 0:
                    ndcg_sum += dcg / idcg

                for rank, item_idx in enumerate(top_k_indices):
                    if item_idx in gt_items:
                        mrr_sum += 1.0 / (rank + 1)
                        break

                num_hits = 0
                ap = 0.0
                for rank, item_idx in enumerate(top_k_indices):
                    if item_idx in gt_items:
                        num_hits += 1
                        ap += num_hits / (rank + 1)
                denom = min(len(gt_items), k)
                if denom > 0 and num_hits > 0:
                    map_sum += ap / denom

        if valid_users > 0:
            metrics["recall"][f"@{k}"] = recall_sum / valid_users
            metrics["ndcg"][f"@{k}"] = ndcg_sum / valid_users
            metrics["precision"][f"@{k}"] = precision_sum / valid_users
            metrics["mrr"][f"@{k}"] = mrr_sum / valid_users
            metrics["map"][f"@{k}"] = map_sum / valid_users

    return metrics

    def state_dict(self) -> dict[str, Any]:
        """Return model state for checkpoint saving."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return {
            "user_factors": self.model.user_factors,
            "item_factors": self.model.item_factors,
            "factors": self.factors,
            "regularization": self.regularization,
            "iterations": self.iterations,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Load model state from checkpoint."""
        self.model = implicit.bpr.BayesianPersonalizedRanking(
            factors=state["factors"],
            regularization=state["regularization"],
            iterations=state["iterations"],
            use_gpu=False,
            num_threads=1,
        )
        self.model.user_factors = state["user_factors"]
        self.model.item_factors = state["item_factors"]


def compute_ranking_metrics(
    predicted_scores: np.ndarray,
    test_csr: csr_matrix,
    k_values: list[int] = None,
) -> dict[str, dict[str, float]]:
    """Compute ranking metrics: Recall, NDCG, Precision, MRR, MAP with sparse efficiency.

    Parameters
    ----------
    predicted_scores:
        Dense array of shape (n_users, n_items) with predicted scores.
    test_csr:
        Sparse CSR matrix of ground truth interactions.
    k_values:
        List of k values to compute metrics for (default: [5, 10, 20, 50]).

    Returns
    -------
    dict[str, dict[str, float]]
        Metrics in format: {"recall": {"@5": 0.123, "@10": 0.456, ...}, ...}
    """
    if k_values is None:
        k_values = [5, 10, 20, 50]

    metrics = {
        "recall": {},
        "ndcg": {},
        "precision": {},
        "mrr": {},
        "map": {},
    }

    n_users = test_csr.shape[0]

    for k in k_values:
        recall_sum = 0.0
        ndcg_sum = 0.0
        precision_sum = 0.0
        mrr_sum = 0.0
        map_sum = 0.0
        valid_users = 0

        for user_idx in tqdm(range(n_users), desc=f"Computing @{k}", disable=True):
            # Get top-k predictions
            top_k_indices = np.argsort(predicted_scores[user_idx])[-k:][::-1]

            # Ground truth for this user (use sparse indexing)
            gt_indices = test_csr[user_idx].nonzero()[1]  # Column indices of nonzero elements
            gt_items = set(gt_indices)

            if len(gt_items) == 0:
                continue

            valid_users += 1

            # Recall@k
            hits = len(set(top_k_indices) & gt_items)
            recall_sum += hits / len(gt_items)

            # Precision@k
            precision_sum += hits / k

            # NDCG@k (standard definition)
            dcg = 0.0
            for rank, item_idx in enumerate(top_k_indices):
                if item_idx in gt_items:
                    dcg += 1.0 / np.log2(rank + 2)
            ideal_rel = min(len(gt_items), k)
            idcg = sum((1.0 / np.log2(i + 2)) for i in range(ideal_rel))
            if idcg > 0:
                ndcg_sum += dcg / idcg

            # MRR@k
            for rank, item_idx in enumerate(top_k_indices):
                if item_idx in gt_items:
                    mrr_sum += 1.0 / (rank + 1)
                    break

            # MAP@k (average precision up to k)
            num_hits = 0
            ap = 0.0
            for rank, item_idx in enumerate(top_k_indices):
                if item_idx in gt_items:
                    num_hits += 1
                    ap += num_hits / (rank + 1)
            denom = min(len(gt_items), k)
            if denom > 0 and num_hits > 0:
                map_sum += ap / denom

        if valid_users > 0:
            metrics["recall"][f"@{k}"] = recall_sum / valid_users
            metrics["ndcg"][f"@{k}"] = ndcg_sum / valid_users
            metrics["precision"][f"@{k}"] = precision_sum / valid_users
            metrics["mrr"][f"@{k}"] = mrr_sum / valid_users
            metrics["map"][f"@{k}"] = map_sum / valid_users

    return metrics


def grid_search_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all grid search configurations from config.

    Parameters
    ----------
    config:
        Config dict with "grid" key containing parameter lists.

    Returns
    -------
    list[dict[str, Any]]
        List of all parameter combinations.
    """
    grid_params = config.get("grid", {})

    # Separate into lists and static values
    param_lists = {}
    static_params = {}

    for key, value in grid_params.items():
        if isinstance(value, list):
            param_lists[key] = value
        else:
            static_params[key] = value

    # Generate all combinations
    if not param_lists:
        return [static_params]

    keys = list(param_lists.keys())
    value_lists = [param_lists[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    configs_list = []
    for combo in combinations:
        config_dict = static_params.copy()
        for key, value in zip(keys, combo):
            config_dict[key] = value
        configs_list.append(config_dict)

    return configs_list


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline recommender models (ALS, EASE) with grid search"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="scripts/baseline_gridsearch.yaml",
        help="Path to grid search config",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["als", "ease", "both"],
        default="both",
        help="Which model to train",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/preprocessed_yelp",
        help="Directory with preprocessed CSR matrices",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/baseline_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ALS training",
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = project_root / "outputs" / f"baseline_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Project root: {project_root}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("Loading preprocessed data...")
    try:
        X_csr = load_npz(data_dir / "R_PA_compact.npz")  # or R_full.npz
        with open(data_dir / "user2index.pkl", "rb") as f:
            user2index = pickle.load(f)
        with open(data_dir / "item2index.pkl", "rb") as f:
            item2index = pickle.load(f)
    except FileNotFoundError as e:
        logger.error(f"Missing preprocessed data: {e}")
        return

    n_users, n_items = X_csr.shape
    logger.info(f"Data shape: {n_users} users × {n_items} items, {X_csr.nnz} interactions")

    # Train-test split: per-user holdout (keep same users in train and test matrices)
    def per_user_holdout(matrix: csr_matrix, test_frac: float = 0.2, seed: int = 42, min_test_items: int = 1):
        rng = np.random.default_rng(seed)
        n_users, n_items = matrix.shape
        train_mat = matrix.tolil(copy=True)
        test_mat = csr_matrix(matrix.shape, dtype=matrix.dtype).tolil()
        users_with_holdout = 0
        held_out_total = 0
        for u in range(n_users):
            items = matrix[u].nonzero()[1]
            if len(items) <= min_test_items:
                continue
            k = max(1, int(len(items) * test_frac))
            held = rng.choice(items, size=k, replace=False)
            for it in held:
                train_mat[u, it] = 0
                test_mat[u, it] = matrix[u, it]
            users_with_holdout += 1
            held_out_total += len(held)
        logger.info(f"Per-user holdout: {users_with_holdout} users with holdouts, {held_out_total} held-out interactions")
        return train_mat.tocsr(), test_mat.tocsr()

    X_train, X_test = per_user_holdout(X_csr, test_frac=0.2, seed=42)
    logger.info(f"Train/Test matrices: {X_train.shape[0]} users × {X_train.shape[1]} items")

    # Load grid search config
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        logger.info(f"Creating default config...")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = {
        "timestamp": datetime.now().isoformat(),
        "data": {"n_users": n_users, "n_items": n_items, "n_interactions": int(X_csr.nnz)},
        "train_test_split": {"train_users": int(X_train.shape[0]), "test_users": int(X_test.shape[0])},
        "runs": [],
    }

    # Train ALS
    if args.model in ["als", "both"]:
        logger.info("\n" + "=" * 80)
        logger.info("GRID SEARCH: ALS")
        logger.info("=" * 80)

        als_configs = grid_search_configs(config.get("als", {}))
        logger.info(f"Testing {len(als_configs)} ALS configurations...")

        best_als_metrics = None
        best_als_config = None
        best_als_score = -1

        for run_idx, als_params in enumerate(als_configs):
            logger.info(f"\n--- ALS Run {run_idx + 1}/{len(als_configs)} ---")
            logger.info(f"Params: {als_params}")

            try:
                model = ALSBaseline(
                    n_items=n_items,
                    factors=int(als_params.get("factors", 64)),
                    regularization=float(als_params.get("regularization", 0.1)),
                    iterations=int(als_params.get("iterations", 15)),
                    use_gpu=args.device == "cuda",
                )

                # Fit and evaluate in batches to avoid dense full-matrix allocation
                model.fit(X_train)
                metrics = compute_ranking_metrics_batched(
                    model,
                    test_csr=X_test,
                    train_csr=X_train,
                    batch_size=256,
                )

                # Track best
                ndcg20 = metrics["ndcg"].get("@20", 0)
                if ndcg20 > best_als_score:
                    best_als_score = ndcg20
                    best_als_metrics = metrics
                    best_als_config = als_params

                logger.info(f"  NDCG@20: {ndcg20:.4f}")

                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"als_run{run_idx:02d}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": als_params,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )

                results["runs"].append(
                    {
                        "model": "ALS",
                        "run_id": run_idx,
                        "config": als_params,
                        "metrics": metrics,
                        "checkpoint": str(checkpoint_path),
                        "best": False,
                    }
                )

            except Exception as e:
                logger.error(f"  Error in ALS run {run_idx}: {e}")

        if best_als_metrics:
            logger.info(f"\n✓ Best ALS: NDCG@20={best_als_score:.4f}")
            logger.info(f"  Config: {best_als_config}")

            # Save best checkpoint
            best_als_path = checkpoint_dir / "als_best.pt"
            torch.save(
                {
                    "model_state": None,  # Would need to retrain to get state
                    "config": best_als_config,
                    "metrics": best_als_metrics,
                },
                best_als_path,
            )

            results["runs"][-1]["best"] = True
            results["als_best"] = {
                "config": best_als_config,
                "metrics": best_als_metrics,
                "checkpoint": str(best_als_path),
            }

    # Train BPR
    if args.model in ["bpr", "both"]:
        logger.info("\n" + "=" * 80)
        logger.info("GRID SEARCH: BPR-MF")
        logger.info("=" * 80)

        bpr_configs = grid_search_configs(config.get("bpr", {}))
        logger.info(f"Testing {len(bpr_configs)} BPR configurations...")

        best_bpr_metrics = None
        best_bpr_config = None
        best_bpr_score = -1

        for run_idx, bpr_params in enumerate(bpr_configs):
            logger.info(f"\n--- BPR Run {run_idx + 1}/{len(bpr_configs)} ---")
            logger.info(f"Params: {bpr_params}")

            try:
                model = BPRBaseline(
                    n_items=n_items,
                    factors=int(bpr_params.get("factors", 64)),
                    regularization=float(bpr_params.get("regularization", 0.01)),
                    iterations=int(bpr_params.get("iterations", 100)),
                )

                # Fit and evaluate in batches to avoid dense full-matrix allocation
                model.fit(X_train)
                metrics = compute_ranking_metrics_batched(
                    model,
                    test_csr=X_test,
                    train_csr=X_train,
                    batch_size=256,
                )

                # Track best
                ndcg20 = metrics["ndcg"].get("@20", 0)
                if ndcg20 > best_bpr_score:
                    best_bpr_score = ndcg20
                    best_bpr_metrics = metrics
                    best_bpr_config = bpr_params

                logger.info(f"  NDCG@20: {ndcg20:.4f}")

                # Save checkpoint
                checkpoint_path = checkpoint_dir / f"bpr_run{run_idx:02d}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": bpr_params,
                        "metrics": metrics,
                    },
                    checkpoint_path,
                )

                results["runs"].append(
                    {
                        "model": "BPR",
                        "run_id": run_idx,
                        "config": bpr_params,
                        "metrics": metrics,
                        "checkpoint": str(checkpoint_path),
                        "best": False,
                    }
                )

            except Exception as e:
                logger.error(f"  Error in BPR run {run_idx}: {e}")

        if best_bpr_metrics:
            logger.info(f"\n✓ Best BPR: NDCG@20={best_bpr_score:.4f}")
            logger.info(f"  Config: {best_bpr_config}")

            # Save best checkpoint
            best_bpr_path = checkpoint_dir / "bpr_best.pt"
            torch.save(
                {
                    "model_state": None,
                    "config": best_bpr_config,
                    "metrics": best_bpr_metrics,
                },
                best_bpr_path,
            )

            results["runs"][-1]["best"] = True
            results["bpr_best"] = {
                "config": best_bpr_config,
                "metrics": best_bpr_metrics,
                "checkpoint": str(best_bpr_path),
            }

    # Save summary.json
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save experiment manifest.json for UI strict best-run mode
    experiment_id = output_dir.name
    manifest_path = output_dir.parent / "experiments" / experiment_id / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build manifest with als_best and bpr_best runs
    manifest_runs = []
    for run in results.get("runs", []):
        model_name = run.get("model", "unknown")
        run_dir = run.get("checkpoint", "").rsplit("\\", 1)[0] if run.get("checkpoint") else str(output_dir)
        
        # Convert metrics to ranking_metrics_sae format for UI compatibility
        metrics = run.get("metrics", {})
        ranking_metrics_sae = {
            "ndcg": metrics.get("ndcg", {}),
            "recall": metrics.get("recall", {}),
            "precision": metrics.get("precision", {}),
            "mrr": metrics.get("mrr", {}),
            "map": metrics.get("map", {}),
        }
        
        summary_with_metrics = {
            **results,
            "ranking_metrics_sae": ranking_metrics_sae,
        }
        
        manifest_runs.append({
            "run_name": f"{model_name.lower()}_run_{run.get('run_id', 0):02d}",
            "experiment_name": f"baseline_{model_name.lower()}",
            "variant_index": run.get("run_id", 0) + 1,
            "run_dir": str(run_dir),
            "config_path": str(output_dir / "config.yaml"),
            "summary_path": str(summary_path),
            "summary": summary_with_metrics,
        })
    
    manifest = {
        "experiment_id": experiment_id,
        "created": results.get("timestamp"),
        "source_config": "scripts/baseline_gridsearch.yaml",
        "base_config": "configs/default.yaml",
        "runs": manifest_runs,
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\n" + "=" * 80)
    logger.info(f"✓ Training complete!")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"  Manifest (UI): {manifest_path}")
    logger.info(f"=" * 80)


if __name__ == "__main__":
    main()
