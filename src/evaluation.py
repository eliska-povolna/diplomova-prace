"""Evaluation module for SAE-CF recommender.

Provides holdout-based evaluation similar to semestral_project:
- Holds out 20% of items for each user
- Computes Recall@K and NDCG@K
- Compares ELSA-only vs ELSA+SAE performance
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.models.collaborative_filtering import ELSA, ndcg_at_k, recall_at_k
from src.models.sparse_autoencoder import TopKSAE


def evaluate_user_rankings(
    X_test: csr_matrix,
    elsa_model: ELSA,
    sae_model: TopKSAE | None = None,
    k_values: list[int] = [20, 50],
    min_interactions: int = 5,
    holdout_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Evaluate models on test set using per-user item holdout.

    For each user with >= min_interactions items:
    1. Hold out 20% of items
    2. Generate recommendations from remaining items
    3. Compute Recall@K and NDCG@K on held-out items

    Parameters
    ----------
    X_test : csr_matrix
        Test interaction matrix (n_users × n_items)
    elsa_model : ELSA
        Trained ELSA model
    sae_model : TopKSAE, optional
        Trained SAE model. If None, only ELSA is evaluated.
    k_values : list[int]
        Values of K for Recall@K and NDCG@K (default [20, 50])
    min_interactions : int
        Minimum interactions per user to evaluate (default 5)
    holdout_ratio : float
        Fraction of items to hold out per user (default 0.2)
    seed : int
        Random seed for reproducibility (default 42)

    Returns
    -------
    dict
        Metrics with structure:
        {
            'elsa': {
                'recall_20': [...],  # Per-user recall values
                'recall_50': [...],
                'ndcg_20': [...],
                'ndcg_50': [...]
            },
            'sae': {...},  # Only if sae_model provided
            'n_users_eval': int,  # Number of users evaluated
            'means': {
                'elsa': {'recall_20': float, 'recall_50': float, ...},
                'sae': {...}
            }
        }
    """
    device = "cuda" if next(elsa_model.parameters()).is_cuda else "cpu"
    elsa_model.eval()
    if sae_model is not None:
        sae_model.eval()

    n_users = X_test.shape[0]

    # Storage for per-user metrics
    metrics_elsa = {f"recall_{k}": [] for k in k_values}
    metrics_elsa.update({f"ndcg_{k}": [] for k in k_values})

    metrics_sae = None
    if sae_model is not None:
        metrics_sae = {f"recall_{k}": [] for k in k_values}
        metrics_sae.update({f"ndcg_{k}": [] for k in k_values})

    # Per-user evaluation
    with torch.no_grad():
        for user_id in range(n_users):
            user_full = X_test.getrow(user_id).toarray().squeeze()

            # Skip users with insufficient interactions
            nonzero_items = np.where(user_full > 0)[0]
            if len(nonzero_items) < min_interactions:
                continue

            # Hold out 20% of items
            np.random.seed(seed + user_id)
            n_holdout = max(1, int(len(nonzero_items) * holdout_ratio))
            holdout_items = np.random.choice(
                nonzero_items, size=n_holdout, replace=False
            )

            # Create train/test split
            user_input = user_full.copy()
            user_target = np.zeros_like(user_full)

            user_input[holdout_items] = 0
            user_target[holdout_items] = user_full[holdout_items]

            user_tensor = torch.tensor(user_input, dtype=torch.float32).unsqueeze(0)
            user_tensor = user_tensor.to(device)

            # ELSA-only predictions (matching semestral_project exactly)
            # Rank by residual: reconstruction - input (not just reconstruction)
            recon_elsa = elsa_model(user_tensor)  # Full reconstruction
            scores_elsa = (
                (recon_elsa - user_tensor).squeeze().cpu().numpy()
            )  # Residual!

            # Mask already-seen items (don't rank them)
            scores_elsa[user_input > 0] = -np.inf
            top_indices_elsa = np.argsort(-scores_elsa)

            # Record ELSA metrics
            for k in k_values:
                metrics_elsa[f"recall_{k}"].append(
                    recall_at_k(user_target, top_indices_elsa, k)
                )
                metrics_elsa[f"ndcg_{k}"].append(
                    ndcg_at_k(user_target, top_indices_elsa, k)
                )

            # SAE predictions if model provided
            if sae_model is not None:
                # Get latent encoding (normalized internally in encode)
                z_user = elsa_model.encode(user_tensor)
                # Encode through SAE and reconstruct
                recon_z, _, _ = sae_model(z_user)
                # Reconstruct items from steered latent
                A_norm = torch.nn.functional.normalize(elsa_model.A, dim=-1)
                recon_items = recon_z @ A_norm.T
                # Score as residual (recon - input) like semestral_project
                scores_sae = (recon_items - user_tensor).squeeze().cpu().numpy()

                # Mask already-seen items
                scores_sae[user_input > 0] = -np.inf
                top_indices_sae = np.argsort(-scores_sae)

                # Record SAE metrics
                for k in k_values:
                    metrics_sae[f"recall_{k}"].append(
                        recall_at_k(user_target, top_indices_sae, k)
                    )
                    metrics_sae[f"ndcg_{k}"].append(
                        ndcg_at_k(user_target, top_indices_sae, k)
                    )

    # Compute means (handling NaN values)
    n_users_eval = len(metrics_elsa["recall_20"])
    means_elsa = {k: float(np.nanmean(v)) for k, v in metrics_elsa.items()}
    means_sae = (
        {k: float(np.nanmean(v)) for k, v in metrics_sae.items()} if metrics_sae else {}
    )

    return {
        "elsa": metrics_elsa,
        "sae": metrics_sae,
        "n_users_eval": n_users_eval,
        "means": {
            "elsa": means_elsa,
            "sae": means_sae,
        },
    }
