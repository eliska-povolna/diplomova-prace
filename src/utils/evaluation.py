"""Ranking metrics for recommendation system evaluation.

Computes standard metrics: NDCG, Recall, MRR, Hit Rate, MAP across all test users.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import sparse

logger = logging.getLogger(__name__)


def ndcg_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K for a single user.

    Parameters
    ----------
    y_true : np.ndarray
        Binary array of ground truth (1 = relevant, 0 = irrelevant)
    y_pred_argsorted : np.ndarray
        Indices sorted by prediction score (descending)
    k : int
        Cutoff for ranking

    Returns
    -------
    float
        NDCG@k value (0 to 1)
    """
    hits = y_true[y_pred_argsorted[:k]]
    if hits.sum() == 0:
        return 0.0

    # DCG: Discount by log2(position+1)
    gains = hits / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()

    # IDCG: Ideal ranking (all 1s first, then 0s)
    ideal = np.sort(y_true)[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, k + 2))).sum()

    return float(dcg / idcg) if idcg > 0 else 0.0


def recall_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """
    Recall @ K for a single user.

    Parameters
    ----------
    y_true : np.ndarray
        Binary array of ground truth
    y_pred_argsorted : np.ndarray
        Indices sorted by prediction score (descending)
    k : int
        Cutoff for ranking

    Returns
    -------
    float
        Recall@k value (0 to 1)
    """
    hits = y_true[y_pred_argsorted[:k]].sum()
    total = y_true.sum()
    return float(hits / total) if total > 0 else float("nan")


def precision_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """
    Precision @ K for a single user.

    Parameters
    ----------
    y_true : np.ndarray
        Binary array of ground truth
    y_pred_argsorted : np.ndarray
        Indices sorted by prediction score (descending)
    k : int
        Cutoff for ranking

    Returns
    -------
    float
        Precision@k value (0 to 1)
    """
    hits = y_true[y_pred_argsorted[:k]].sum()
    return float(hits / k) if k > 0 else 0.0


def mrr_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """
    Mean Reciprocal Rank @ K for a single user.

    Parameters
    ----------
    y_true : np.ndarray
        Binary array of ground truth
    y_pred_argsorted : np.ndarray
        Indices sorted by prediction score (descending)
    k : int
        Cutoff for ranking

    Returns
    -------
    float
        MRR@k value (0 to 1, or 0 if no hit in top-k)
    """
    hits = y_true[y_pred_argsorted[:k]]
    if hits.sum() == 0:
        return 0.0
    rank_of_first_hit = np.where(hits == 1)[0][0] + 1
    return float(1.0 / rank_of_first_hit)


def hit_rate_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """
    Hit Rate @ K (whether any relevant item is in top-k).

    Parameters
    ----------
    y_true : np.ndarray
        Binary array of ground truth
    y_pred_argsorted : np.ndarray
        Indices sorted by prediction score (descending)
    k : int
        Cutoff for ranking

    Returns
    -------
    float
        Hit rate (0 or 1)
    """
    hits = y_true[y_pred_argsorted[:k]].sum()
    return 1.0 if hits > 0 else 0.0


def map_at_k(y_true: np.ndarray, y_pred_argsorted: np.ndarray, k: int) -> float:
    """
    Mean Average Precision @ K for a single user.

    Parameters
    ----------
    y_true : np.ndarray
        Binary array of ground truth
    y_pred_argsorted : np.ndarray
        Indices sorted by prediction score (descending)
    k : int
        Cutoff for ranking

    Returns
    -------
    float
        MAP@k value (0 to 1)
    """
    y_true_topk = y_true[y_pred_argsorted[:k]]
    num_rel = y_true.sum()

    if num_rel == 0:
        return float("nan")

    score = 0.0
    num_hits = 0

    for i, rel in enumerate(y_true_topk):
        if rel == 1:
            num_hits += 1
            score += num_hits / (i + 1)

    return float(score / min(k, num_rel))


def compute_metrics_batch(
    y_true_batch: np.ndarray,
    y_pred_batch: np.ndarray,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, Dict[str, float]]:
    """
    Compute all ranking metrics for a batch of users.

    Parameters
    ----------
    y_true_batch : np.ndarray
        (n_users, n_items) binary matrix of ground truth interactions
    y_pred_batch : np.ndarray
        (n_users, n_items) prediction scores (typically from reconstruction)
    ks : List[int]
        Cutoff values to evaluate (default: [5, 10, 20])

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dict mapping metric_name -> {k -> average_value}
        E.g., {"ndcg": {"@5": 0.45, "@10": 0.40, "@20": 0.35}, ...}
    """
    n_users = y_true_batch.shape[0]
    metrics_k = {k: [] for k in ks}

    results = {
        "ndcg": {},
        "recall": {},
        "precision": {},
        "mrr": {},
        "hr": {},
        "map": {},
    }

    # Compute for each user
    for user_idx in range(n_users):
        y_true = y_true_batch[user_idx]
        y_pred = y_pred_batch[user_idx]

        # Skip users with no ground truth
        if y_true.sum() == 0:
            continue

        # Get ranking (descending order)
        y_pred_argsorted = np.argsort(-y_pred)

        for k in ks:
            ndcg = ndcg_at_k(y_true, y_pred_argsorted, k)
            recall = recall_at_k(y_true, y_pred_argsorted, k)
            prec = precision_at_k(y_true, y_pred_argsorted, k)
            mrr = mrr_at_k(y_true, y_pred_argsorted, k)
            hr = hit_rate_at_k(y_true, y_pred_argsorted, k)
            ap = map_at_k(y_true, y_pred_argsorted, k)

            # Store only non-NaN values
            if not np.isnan(ndcg):
                if k not in results["ndcg"]:
                    results["ndcg"][f"@{k}"] = []
                results["ndcg"][f"@{k}"].append(ndcg)

            if not np.isnan(recall):
                if k not in results["recall"]:
                    results["recall"][f"@{k}"] = []
                results["recall"][f"@{k}"].append(recall)

            if not np.isnan(prec):
                if k not in results["precision"]:
                    results["precision"][f"@{k}"] = []
                results["precision"][f"@{k}"].append(prec)

            if not np.isnan(mrr):
                if k not in results["mrr"]:
                    results["mrr"][f"@{k}"] = []
                results["mrr"][f"@{k}"].append(mrr)

            if not np.isnan(hr):
                if k not in results["hr"]:
                    results["hr"][f"@{k}"] = []
                results["hr"][f"@{k}"].append(hr)

            if not np.isnan(ap):
                if k not in results["map"]:
                    results["map"][f"@{k}"] = []
                results["map"][f"@{k}"].append(ap)

    # Average across users
    averaged_results = {
        "ndcg": {},
        "recall": {},
        "precision": {},
        "mrr": {},
        "hr": {},
        "map": {},
    }

    for metric in results:
        for k_str in results[metric]:
            values = results[metric][k_str]
            if values:
                averaged_results[metric][k_str] = float(np.mean(values))
            else:
                averaged_results[metric][k_str] = float("nan")

    return averaged_results


def evaluate_recommendations(
    X_test_csr,
    model_predictions: np.ndarray,
    ks: List[int] = [5, 10, 20],
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate recommendations against test set interactions.

    Parameters
    ----------
    X_test_csr : scipy.sparse.csr_matrix
        Test CSR matrix (sparse interactions)
    model_predictions : np.ndarray
        (n_test_users, n_items) prediction scores
    ks : List[int]
        Cutoff values to evaluate

    Returns
    -------
    Dict[str, Dict[str, float]]
        Ranking metrics for each K
    """
    # Convert sparse input to dense for metric computation if needed
    if hasattr(X_test_csr, "toarray"):
        X_test_dense = X_test_csr.toarray()
    else:
        X_test_dense = np.asarray(X_test_csr)

    return compute_metrics_batch(X_test_dense, model_predictions, ks)


def build_holdout_split(
    X_csr,
    *,
    holdout_ratio: float = 0.2,
    min_interactions: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a per-user holdout split for ranking evaluation.

    For each user, a fraction of non-zero items is moved from the input matrix
    into the target matrix. The model is scored on the masked input, and the
    held-out items form the ground truth for ranking metrics.
    """
    X_dense = X_csr.toarray().astype(np.float32)
    X_input = X_dense.copy()
    X_target = np.zeros_like(X_dense)

    for user_idx in range(X_dense.shape[0]):
        nonzero_items = np.where(X_dense[user_idx] > 0)[0]
        if len(nonzero_items) < min_interactions:
            continue

        rng = np.random.default_rng(seed + user_idx)
        n_holdout = max(1, int(len(nonzero_items) * holdout_ratio))
        holdout_items = rng.choice(nonzero_items, size=n_holdout, replace=False)

        X_input[user_idx, holdout_items] = 0.0
        X_target[user_idx, holdout_items] = X_dense[user_idx, holdout_items]

    return X_input, X_target


def build_holdout_split_sparse(
    X_csr,
    *,
    holdout_ratio: float = 0.2,
    min_interactions: int = 5,
    seed: int = 42,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Create a sparse per-user holdout split for large-scale ranking evaluation.

    This is the memory-safe variant of :func:`build_holdout_split`. It keeps the
    masked input and the held-out target as sparse matrices so evaluation can be
    performed batch by batch without densifying the full dataset.
    """
    X_input = X_csr.tolil(copy=True)
    X_target = sparse.lil_matrix(X_csr.shape, dtype=np.float32)

    for user_idx in range(X_csr.shape[0]):
        row_start = X_csr.indptr[user_idx]
        row_end = X_csr.indptr[user_idx + 1]
        nonzero_items = X_csr.indices[row_start:row_end]

        if len(nonzero_items) < min_interactions:
            continue

        rng = np.random.default_rng(seed + user_idx)
        n_holdout = max(1, int(len(nonzero_items) * holdout_ratio))
        holdout_items = rng.choice(nonzero_items, size=n_holdout, replace=False)

        X_input[user_idx, holdout_items] = 0.0
        X_target[user_idx, holdout_items] = 1.0

    return X_input.tocsr(), X_target.tocsr()


def evaluate_recommendations_batched(
    X_input_csr,
    X_target_csr,
    score_fn,
    ks: List[int] = [5, 10, 20],
    batch_size: int = 256,
) -> tuple[Dict[str, Dict[str, float]], int]:
    """Evaluate ranking metrics on a sparse holdout split in batches.

    Parameters
    ----------
    X_input_csr : scipy.sparse.csr_matrix
        Masked user history used as model input.
    X_target_csr : scipy.sparse.csr_matrix
        Held-out ground truth interactions.
    score_fn : Callable[[np.ndarray], np.ndarray]
        Function that maps a dense user batch to a dense score matrix.
    ks : List[int]
        Cutoff values to evaluate.
    batch_size : int
        Number of users to score per batch.

    Returns
    -------
    tuple[Dict[str, Dict[str, float]], int]
        Averaged metrics and the number of evaluated users.
    """
    totals = {
        "ndcg": {f"@{k}": 0.0 for k in ks},
        "recall": {f"@{k}": 0.0 for k in ks},
        "precision": {f"@{k}": 0.0 for k in ks},
        "mrr": {f"@{k}": 0.0 for k in ks},
        "hr": {f"@{k}": 0.0 for k in ks},
        "map": {f"@{k}": 0.0 for k in ks},
    }
    evaluated_users = 0

    n_users = X_input_csr.shape[0]
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        input_batch = X_input_csr[start:end].toarray().astype(np.float32)
        target_batch = X_target_csr[start:end].toarray().astype(np.float32)

        batch_user_mask = target_batch.sum(axis=1) > 0
        batch_evaluated = int(batch_user_mask.sum())
        if batch_evaluated == 0:
            continue

        score_batch = score_fn(input_batch)
        batch_metrics = compute_metrics_batch(target_batch, score_batch, ks)

        for metric_name, metric_values in batch_metrics.items():
            for k_str, value in metric_values.items():
                if not np.isnan(value):
                    totals[metric_name][k_str] += float(value) * batch_evaluated

        evaluated_users += batch_evaluated

    averaged = {
        metric_name: {
            k_str: (
                float(total / evaluated_users) if evaluated_users > 0 else float("nan")
            )
            for k_str, total in metric_values.items()
        }
        for metric_name, metric_values in totals.items()
    }

    return averaged, evaluated_users


def print_evaluation_report(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Format evaluation metrics as a readable report.

    Parameters
    ----------
    metrics : Dict[str, Dict[str, float]]
        Output from compute_metrics_batch or evaluate_recommendations

    Returns
    -------
    str
        Formatted report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("RANKING METRICS EVALUATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    metric_names = {
        "ndcg": "NDCG (Normalized Discounted Cumulative Gain)",
        "recall": "Recall",
        "precision": "Precision",
        "mrr": "MRR (Mean Reciprocal Rank)",
        "hr": "HR (Hit Rate)",
        "map": "MAP (Mean Average Precision)",
    }

    for metric_key, metric_display_name in metric_names.items():
        if metric_key in metrics:
            lines.append(f"{metric_display_name}:")
            for k_str, value in sorted(metrics[metric_key].items()):
                if not np.isnan(value):
                    lines.append(f"  {k_str}: {value:.4f}")
                else:
                    lines.append(f"  {k_str}: N/A")
            lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def compute_coverage(all_recommendations: Dict[int, np.ndarray], n_items: int) -> float:
    """
    Compute coverage: % of unique items recommended across all users.

    Parameters
    ----------
    all_recommendations : Dict[int, np.ndarray]
        {user_idx: [item1_idx, item2_idx, ...]} for each user
    n_items : int
        Total number of items in catalog

    Returns
    -------
    float
        Coverage ratio (0 to 1)
    """
    unique_items = set()
    for user_idx, items in all_recommendations.items():
        unique_items.update(items)

    coverage = len(unique_items) / n_items if n_items > 0 else 0.0
    return float(coverage)


def compute_average_popularity(
    all_recommendations: Dict[int, np.ndarray],
    item_popularity: Dict[int, float],
) -> float:
    """
    Compute average popularity of recommended items.

    Parameters
    ----------
    all_recommendations : Dict[int, np.ndarray]
        {user_idx: [item1_idx, item2_idx, ...]} for each user
    item_popularity : Dict[int, float]
        {item_idx: popularity_score} (normalized 0-1)

    Returns
    -------
    float
        Mean popularity score across all recommended items
    """
    popularities = []
    for user_idx, items in all_recommendations.items():
        for item_idx in items:
            pop = item_popularity.get(item_idx, 0.0)
            popularities.append(pop)

    avg_pop = np.mean(popularities) if popularities else 0.0
    return float(avg_pop)


def compute_entropy(all_recommendations: Dict[int, np.ndarray], n_items: int) -> float:
    """
    Compute recommendation distribution entropy (diversity measure).

    Higher entropy = more diverse recommendations
    Lower entropy = concentrated on popular items

    Parameters
    ----------
    all_recommendations : Dict[int, np.ndarray]
        {user_idx: [item1_idx, item2_idx, ...]} for each user
    n_items : int
        Total number of items in catalog

    Returns
    -------
    float
        Entropy value (normalized 0-1, where 1 = uniform distribution)
    """
    # Count how many times each item appears in recommendations
    item_counts = np.zeros(n_items)
    total_recs = 0

    for user_idx, items in all_recommendations.items():
        for item_idx in items:
            if 0 <= item_idx < n_items:
                item_counts[item_idx] += 1
                total_recs += 1

    if total_recs == 0:
        return 0.0

    # Compute probability distribution
    item_probs = item_counts / total_recs

    # Compute entropy: -sum(p * log(p)) for p > 0
    entropy = 0.0
    for p in item_probs:
        if p > 0:
            entropy -= p * np.log(p)

    # Normalize by maximum entropy (uniform distribution)
    max_entropy = np.log(n_items) if n_items > 0 else 1.0
    normalized_entropy = entropy / max_entropy

    return float(normalized_entropy)


def benchmark_inference(
    model_predictions: np.ndarray,
    n_samples: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Benchmark inference latency from model predictions.

    This estimates latency based on the number of predictions and their complexity.
    For more accurate benchmarking, profile the actual inference loop.

    Parameters
    ----------
    model_predictions : np.ndarray
        (n_users, n_items) prediction scores from model
    n_samples : int
        Number of inference samples to estimate
    device : str
        Device used ("cpu" or "cuda")

    Returns
    -------
    Dict[str, float]
        Keys: 'mean_ms', 'p95_ms', 'max_ms', 'p50_ms' (estimated latency in ms)
    """
    import time

    n_users, n_items = model_predictions.shape
    latencies = []

    # Simulate inference: get top-k predictions for random users
    for _ in range(n_samples):
        user_idx = np.random.randint(0, n_users)
        scores = model_predictions[user_idx]

        # Time the operations
        start = time.time()

        # Simulate: top-20 ranking
        _ = np.argsort(-scores)[:20]

        elapsed_ms = (time.time() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.array(latencies)

    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "max_ms": float(np.max(latencies)),
    }


def generate_recommendations(
    X_test_csr,
    model_predictions: np.ndarray,
    k: int = 20,
) -> Dict[int, np.ndarray]:
    """
    Generate top-k recommendations for all test users.

    Parameters
    ----------
    X_test_csr : scipy.sparse.csr_matrix
        Test CSR matrix (sparse interactions)
    model_predictions : np.ndarray
        (n_test_users, n_items) prediction scores
    k : int
        Number of recommendations per user

    Returns
    -------
    Dict[int, np.ndarray]
        {user_idx: [item1_idx, item2_idx, ...]} (top-k items per user)
    """
    recommendations = {}
    n_users = model_predictions.shape[0]

    for user_idx in range(n_users):
        scores = model_predictions[user_idx]
        top_items = np.argsort(-scores)[:k]
        recommendations[user_idx] = top_items

    return recommendations


def compare_model_performance(
    metrics_elsa: Dict[str, Dict[str, float]],
    metrics_sae: Dict[str, Dict[str, float]],
) -> str:
    """
    Compare ELSA vs SAE+ELSA performance and generate comparison table.

    Parameters
    ----------
    metrics_elsa : Dict[str, Dict[str, float]]
        Metrics from ELSA alone (e.g., {"ndcg": {"@5": 0.45, "@10": 0.40}})
    metrics_sae : Dict[str, Dict[str, float]]
        Metrics from SAE+ELSA (same structure)

    Returns
    -------
    str
        Formatted comparison table with % difference (negative = SAE worse, positive = SAE better)
    """
    lines = []
    lines.append("=" * 100)
    lines.append("ELSA vs SAE+ELSA PERFORMANCE COMPARISON (Test Set)")
    lines.append("=" * 100)
    lines.append("")

    # Metrics to compare
    metric_names = {
        "ndcg": "NDCG",
        "recall": "Recall",
        "precision": "Precision",
        "mrr": "MRR",
        "hr": "Hit Rate",
        "map": "MAP",
    }

    for metric_key, metric_display_name in metric_names.items():
        if metric_key not in metrics_elsa or metric_key not in metrics_sae:
            continue

        lines.append(f"{metric_display_name}:")
        lines.append("-" * 100)

        # Header
        header = f"{'K':>4} | {'ELSA':>12} | {'SAE+ELSA':>12} | {'Difference':>12} | {'Change %':>12}"
        lines.append(header)
        lines.append("-" * 100)

        elsa_vals = metrics_elsa[metric_key]
        sae_vals = metrics_sae[metric_key]

        # Sort by k value
        k_values = sorted(set(list(elsa_vals.keys()) + list(sae_vals.keys())))

        for k_str in k_values:
            elsa_val = elsa_vals.get(k_str, float("nan"))
            sae_val = sae_vals.get(k_str, float("nan"))

            if np.isnan(elsa_val) or np.isnan(sae_val):
                continue

            diff = sae_val - elsa_val
            change_pct = (diff / elsa_val * 100) if elsa_val != 0 else 0.0

            # Color indicator
            if change_pct > 0:
                indicator = "↑ (BETTER)"
            elif change_pct < 0:
                indicator = "↓ (WORSE)"
            else:
                indicator = "→ (EQUAL)"

            line = (
                f"{k_str:>4} | {elsa_val:>12.4f} | {sae_val:>12.4f} | "
                f"{diff:>12.4f} | {change_pct:>10.2f}% {indicator}"
            )
            lines.append(line)

        lines.append("")

    lines.append("=" * 100)
    lines.append(
        "Note: Positive % = SAE+ELSA is better, Negative % = SAE+ELSA is worse than ELSA alone"
    )
    lines.append("=" * 100)

    return "\n".join(lines)
