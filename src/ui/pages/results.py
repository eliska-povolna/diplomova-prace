"""Results page: strict, real-artifact evaluation views."""

from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)
SUPPORTED_K_VALUES = [5, 10, 20, 50]


def _arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize mixed object columns so Streamlit Arrow serialization is stable."""
    safe = df.copy()
    for col in safe.columns:
        series = safe[col]
        if not pd.api.types.is_object_dtype(series):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        value_types = {type(v) for v in non_null.tolist()}
        if len(value_types) > 1:
            safe[col] = series.map(lambda v: "" if pd.isna(v) else str(v))
    return safe


def _flatten_parameters(params: Any, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    flat: Dict[str, Any] = {}
    for key, value in params.items():
        key_str = str(key)
        flat_key = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(value, dict):
            flat.update(_flatten_parameters(value, flat_key))
        else:
            flat[flat_key] = value
    return flat


def _detect_varying_parameter_keys(runs: List[dict]) -> List[str]:
    relevant_prefixes = ("elsa.", "sae.", "data.")
    values_by_key: Dict[str, set] = {}
    for run in runs:
        flat = _flatten_parameters(run.get("parameters", {}))
        for key, value in flat.items():
            if not key.startswith(relevant_prefixes) or value is None:
                continue
            values_by_key.setdefault(key, set()).add(str(value))
    return sorted([k for k, v in values_by_key.items() if len(v) > 1])


def _format_param_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _get_param(flat: Dict[str, Any], candidates: List[str]) -> Any:
    for key in candidates:
        if key in flat and flat[key] is not None:
            return flat[key]
    return None


def _selector_primary_signature(run: dict) -> str:
    summary = run.get("summary") or {}
    ranking_metrics = summary.get("ranking_metrics", {})
    ndcg20 = ranking_metrics.get("ndcg", {}).get("@20")
    flat = _flatten_parameters(run.get("parameters", {}))

    e_lr = _get_param(flat, ["elsa.learning_rate", "elsa.lr", "elsa.optimizer.lr"])
    e_dim = _get_param(flat, ["elsa.latent_dim"])
    s_k = _get_param(flat, ["sae.k"])
    s_lr = _get_param(flat, ["sae.learning_rate", "sae.lr", "sae.optimizer.lr"])

    parts: List[str] = []
    if ndcg20 is not None:
        parts.append(f"NDCG@20={float(ndcg20):.3f}")
    if e_lr is not None:
        parts.append(f"elsa_lr={_format_param_value(e_lr)}")
    if e_dim is not None:
        parts.append(f"elsa_dim={_format_param_value(e_dim)}")
    if s_k is not None:
        parts.append(f"sae_k={_format_param_value(s_k)}")
    if s_lr is not None:
        parts.append(f"sae_lr={_format_param_value(s_lr)}")
    return ", ".join(parts)


def _run_label(run: dict) -> str:
    run_name = run.get("run_name", "run")
    parts = [f"{run_name}{' [best]' if run.get('is_best_run') else ''}"]
    primary = _selector_primary_signature(run)
    if primary:
        parts.append(primary)
    return " | ".join(parts)


def _run_id_from_dir(run_dir: str) -> str:
    if not run_dir:
        return "N/A"
    return Path(str(run_dir)).name or str(run_dir)


def _metric_at(ranking: Dict[str, Any], key: str, at: str = "@20") -> float | None:
    metric = ranking.get(key)
    if key == "hr" and metric is None:
        metric = ranking.get("hit_rate")
    if isinstance(metric, dict):
        value = metric.get(at)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _extract_available_k_values(results: Dict[str, Any]) -> List[int]:
    ks: set[int] = set()
    metric_sources = [
        results.get("ranking_metrics", {}) or {},
        results.get("ranking_metrics_elsa", {}) or {},
        results.get("ranking_metrics_sae", {}) or {},
    ]
    for run in results.get("experiment_runs", []) or []:
        summary = run.get("summary") or {}
        metric_sources.append(summary.get("ranking_metrics", {}) or {})

    for source in metric_sources:
        for metric_name in (
            "ndcg",
            "recall",
            "precision",
            "mrr",
            "map",
            "hr",
            "hit_rate",
        ):
            metric = source.get(metric_name)
            if not isinstance(metric, dict):
                continue
            for key in metric.keys():
                if isinstance(key, str) and key.startswith("@"):
                    try:
                        ks.add(int(key[1:]))
                    except ValueError:
                        continue
    return sorted(ks)


def _short_param_name(key: str) -> str:
    mapping = {
        "elsa.learning_rate": "e_lr",
        "elsa.lr": "e_lr",
        "elsa.optimizer.lr": "e_lr",
        "elsa.latent_dim": "e_dim",
        "elsa.num_epochs": "e_ep",
        "elsa.weight_decay": "e_wd",
        "elsa.patience": "e_pat",
        "elsa.batch_size": "e_bs",
        "sae.k": "s_k",
        "sae.width_ratio": "s_wr",
        "sae.learning_rate": "s_lr",
        "sae.lr": "s_lr",
        "sae.optimizer.lr": "s_lr",
        "sae.num_epochs": "s_ep",
        "sae.patience": "s_pat",
        "sae.min_delta": "s_md",
        "sae.l1_coef": "s_l1",
        "sae.batch_size": "s_bs",
    }
    if key in mapping:
        return mapping[key]
    if "." not in key:
        return key
    head, tail = key.split(".", 1)
    return f"{head[:1]}_{tail.replace('.', '_')}"


def _secondary_param_line(run: dict) -> str:
    flat = _flatten_parameters(run.get("parameters", {}))
    hidden = {
        "elsa.learning_rate",
        "elsa.lr",
        "elsa.optimizer.lr",
        "elsa.latent_dim",
        "sae.k",
        "sae.learning_rate",
        "sae.lr",
        "sae.optimizer.lr",
    }
    pairs = []
    for key in sorted(flat.keys()):
        if key in hidden or flat[key] is None:
            continue
        pairs.append(f"{_short_param_name(key)}={_format_param_value(flat[key])}")
    return ", ".join(pairs) if pairs else "No additional parameters."


def _build_metrics_table(results: Dict[str, Any], k_tags: List[str]) -> pd.DataFrame:
    ranking = results.get("ranking_metrics", {}) or {}
    model_sizes = results.get("model_sizes", {}) or {}
    rows: List[Dict[str, str]] = []
    for key, label in [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("hr", "Hit Rate"),
        ("ndcg", "NDCG"),
        ("mrr", "MRR"),
        ("map", "MAP"),
    ]:
        row: Dict[str, str] = {"Metric": label, "Value": ""}
        has_any = False
        for k_tag in k_tags:
            value = _metric_at(ranking, key, k_tag)
            if value is None:
                row[k_tag] = "N/A"
            else:
                row[k_tag] = f"{value:.4f}"
                has_any = True
        if has_any:
            rows.append(row)
    if "coverage" in ranking:
        rows.append(
            {
                "Metric": "Coverage",
                **{k_tag: "" for k_tag in k_tags},
                "Value": f"{float(ranking['coverage']) * 100:.2f}%",
            }
        )
    if "entropy" in ranking:
        rows.append(
            {
                "Metric": "Entropy",
                **{k_tag: "" for k_tag in k_tags},
                "Value": f"{float(ranking['entropy']):.4f}",
            }
        )
    if "total_mb" in model_sizes:
        rows.append(
            {
                "Metric": "Size (MB)",
                **{k_tag: "" for k_tag in k_tags},
                "Value": f"{float(model_sizes['total_mb']):.2f}",
            }
        )
    return pd.DataFrame(rows)


def _build_metrics_chart_df(results: Dict[str, Any], k_tags: List[str]) -> pd.DataFrame:
    ranking = results.get("ranking_metrics", {}) or {}
    rows: List[Dict[str, Any]] = []
    for key, label in [
        ("recall", "Recall"),
        ("precision", "Precision"),
        ("hr", "Hit Rate"),
        ("ndcg", "NDCG"),
        ("mrr", "MRR"),
        ("map", "MAP"),
    ]:
        for k_tag in k_tags:
            value = _metric_at(ranking, key, k_tag)
            if value is not None:
                rows.append({"Metric": label, "K": k_tag, "Score": value})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _simulate_random_baseline_metrics(
    n_items: int,
    avg_relevant_items: float,
    k_values: tuple[int, ...],
    n_trials: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    if n_items <= 0 or avg_relevant_items <= 0 or not k_values:
        return pd.DataFrame()

    rng = random.Random(seed)
    max_k = min(max(k_values), n_items)
    relevant_count = max(1, min(n_items, int(round(avg_relevant_items))))
    discount = [1.0 / math.log2(i + 2) for i in range(max_k)]

    agg = {
        k: {
            "Recall": 0.0,
            "Precision": 0.0,
            "Hit Rate": 0.0,
            "NDCG": 0.0,
            "MRR": 0.0,
            "MAP": 0.0,
        }
        for k in k_values
    }

    for _ in range(n_trials):
        relevant = set(rng.sample(range(n_items), relevant_count))
        ranked = rng.sample(range(n_items), max_k)
        hits = [1 if item in relevant else 0 for item in ranked]

        for k in k_values:
            kk = min(k, max_k)
            top_hits = hits[:kk]
            hit_count = sum(top_hits)

            precision = hit_count / float(kk)
            recall = hit_count / float(relevant_count)
            hr = 1.0 if hit_count > 0 else 0.0

            dcg = sum(top_hits[i] * discount[i] for i in range(kk))
            idcg = sum(discount[i] for i in range(min(relevant_count, kk)))
            ndcg = (dcg / idcg) if idcg > 0 else 0.0

            mrr = 0.0
            for idx, val in enumerate(top_hits, start=1):
                if val:
                    mrr = 1.0 / float(idx)
                    break

            cum_hits = 0
            ap_sum = 0.0
            for idx, val in enumerate(top_hits, start=1):
                if val:
                    cum_hits += 1
                    ap_sum += cum_hits / float(idx)
            ap = ap_sum / float(relevant_count)

            agg[k]["Recall"] += recall
            agg[k]["Precision"] += precision
            agg[k]["Hit Rate"] += hr
            agg[k]["NDCG"] += ndcg
            agg[k]["MRR"] += mrr
            agg[k]["MAP"] += ap

    rows: List[Dict[str, Any]] = []
    for k in k_values:
        rows.append(
            {
                "K": f"@{k}",
                "Recall": agg[k]["Recall"] / n_trials,
                "Precision": agg[k]["Precision"] / n_trials,
                "Hit Rate": agg[k]["Hit Rate"] / n_trials,
                "NDCG": agg[k]["NDCG"] / n_trials,
                "MRR": agg[k]["MRR"] / n_trials,
                "MAP": agg[k]["MAP"] / n_trials,
            }
        )
    return pd.DataFrame(rows)


def _extract_random_baseline_inputs(
    results: Dict[str, Any],
) -> tuple[int | None, float | None]:
    n_items = None
    avg_heldout = None

    data_block = results.get("data", {}) or {}
    if data_block.get("n_items") is not None:
        n_items = int(data_block.get("n_items"))

    proto = results.get("evaluation_protocol", {}) or {}
    diag = proto.get("holdout_diagnostics", {}) or {}
    if diag.get("avg_heldout_items_per_user") is not None:
        avg_heldout = float(diag.get("avg_heldout_items_per_user"))

    holdout = results.get("holdout", {}) or {}
    holdout_diag = holdout.get("diagnostics", {}) or {}
    if (
        avg_heldout is None
        and holdout_diag.get("avg_heldout_items_per_user") is not None
    ):
        avg_heldout = float(holdout_diag.get("avg_heldout_items_per_user"))

    return n_items, avg_heldout


@st.cache_data(show_spinner=False)
def _recompute_holdout_avg_from_run_config(
    config: dict,
) -> tuple[int | None, float | None]:
    """Recompute n_items and avg held-out items directly from canonical evaluation protocol."""
    try:
        from sklearn.model_selection import train_test_split

        from src.data.shared_preprocessing_cache import (
            prepare_shared_preprocessing_cache,
        )
        from src.utils.evaluation import (
            build_holdout_split_sparse,
            compute_holdout_diagnostics,
        )

        payload, _, _ = prepare_shared_preprocessing_cache(
            config, require_existing=False
        )
        X_csr = payload["final_dataset"].csr

        seed = int(config["data"]["seed"])
        train_test_ratio = float(config["data"]["train_test_split"])
        user_indices = list(range(X_csr.shape[0]))
        _, test_users = train_test_split(
            user_indices,
            test_size=1 - train_test_ratio,
            random_state=seed,
        )
        X_test_csr = X_csr[test_users]

        holdout_ratio = float(config.get("evaluation", {}).get("holdout_ratio", 0.2))
        min_interactions = int(config.get("evaluation", {}).get("min_interactions", 5))
        X_eval_input_csr, X_eval_target_csr = build_holdout_split_sparse(
            X_test_csr,
            holdout_ratio=holdout_ratio,
            min_interactions=min_interactions,
            seed=seed,
        )
        diag = compute_holdout_diagnostics(
            X_eval_input_csr,
            X_eval_target_csr,
            min_interactions=min_interactions,
        )
        return int(X_csr.shape[1]), float(diag.get("avg_heldout_items_per_user", 0.0))
    except Exception:
        return None, None


def _build_random_baseline_reference(
    results: Dict[str, Any], selected_k_values: List[int]
) -> tuple[pd.DataFrame | None, int | None, float | None, List[str]]:
    n_items, avg_heldout = _extract_random_baseline_inputs(results)
    if (not n_items or not avg_heldout or avg_heldout <= 0) and isinstance(
        results.get("config"), dict
    ):
        recomputed_n_items, recomputed_avg = _recompute_holdout_avg_from_run_config(
            results["config"]
        )
        if recomputed_n_items:
            n_items = recomputed_n_items
        if recomputed_avg and recomputed_avg > 0:
            avg_heldout = recomputed_avg

    missing: List[str] = []
    if not n_items:
        missing.append("data.n_items")
    if not avg_heldout or avg_heldout <= 0:
        missing.append(
            "evaluation_protocol.holdout_diagnostics.avg_heldout_items_per_user "
            "or holdout.diagnostics.avg_heldout_items_per_user"
        )
    if missing:
        return None, n_items, avg_heldout, missing
    baseline_df = _simulate_random_baseline_metrics(
        n_items=n_items,
        avg_relevant_items=avg_heldout,
        k_values=tuple(selected_k_values),
    )
    if baseline_df.empty:
        return (
            None,
            n_items,
            avg_heldout,
            ["random baseline simulation returned empty output"],
        )
    return baseline_df, n_items, avg_heldout, []


def _overlay_random_baseline(
    fig: go.Figure, baseline_df: pd.DataFrame | None, k_tags: List[str]
) -> None:
    if baseline_df is None:
        return

    metric_map = {
        "Recall": "Recall",
        "Precision": "Precision",
        "Hit Rate": "Hit Rate",
        "NDCG": "NDCG",
        "MRR": "MRR",
        "MAP": "MAP",
    }
    for k_tag in k_tags:
        row = baseline_df[baseline_df["K"] == k_tag]
        if row.empty:
            continue
        y_vals = [float(row.iloc[0][metric_map[m]]) for m in metric_map]
        fig.add_trace(
            go.Scatter(
                x=list(metric_map.keys()),
                y=y_vals,
                mode="lines+markers",
                name=f"Random {k_tag}",
                line={"dash": "dash", "width": 2},
                marker={"symbol": "x", "size": 8},
            )
        )


def _render_random_baseline_reference(
    baseline_df: pd.DataFrame | None,
    n_items: int | None,
    avg_heldout: float | None,
    missing_requirements: List[str],
) -> None:
    if baseline_df is None:
        st.error(
            "Random baseline unavailable in strict mode. Missing required artifacts: "
            + "; ".join(missing_requirements)
        )
        return

    shown_df = baseline_df.copy()
    for col in ["Recall", "Precision", "Hit Rate", "NDCG", "MRR", "MAP"]:
        shown_df[col] = shown_df[col].map(lambda v: f"{v:.4f}")

    st.markdown("#### Random-Recommender Reference")
    st.caption(
        f"Estimated from random ranking with sparse catalog assumptions: "
        f"{n_items} items, avg held-out relevant items/user ~= {avg_heldout:.2f}."
    )
    st.dataframe(_arrow_safe_df(shown_df), width="stretch", hide_index=True)


def _build_elsa_vs_sae_df(results: Dict[str, Any], k_tags: List[str]) -> pd.DataFrame:
    elsa = results.get("ranking_metrics_elsa", {}) or {}
    sae = results.get("ranking_metrics_sae", {}) or {}
    if not elsa or not sae:
        return pd.DataFrame()

    metric_names = {
        "ndcg": "NDCG",
        "recall": "Recall",
        "precision": "Precision",
        "mrr": "MRR",
        "hr": "Hit Rate",
        "map": "MAP",
    }
    rows: List[Dict[str, Any]] = []
    for metric_key, metric_name in metric_names.items():
        e = elsa.get(metric_key)
        s = sae.get(metric_key)
        if not isinstance(e, dict) or not isinstance(s, dict):
            continue
        for k_tag in k_tags:
            if k_tag not in e or k_tag not in s:
                continue
            e_val = float(e[k_tag])
            s_val = float(s[k_tag])
            diff = s_val - e_val
            pct = (diff / e_val * 100.0) if e_val != 0 else 0.0
            rows.append(
                {
                    "Metric": metric_name,
                    "K": k_tag,
                    "ELSA": e_val,
                    "ELSA+SAE": s_val,
                    "Diff": diff,
                    "Diff%": pct,
                }
            )
    return pd.DataFrame(rows)


def _build_ablation_table(
    runs: List[dict],
    varying_keys: List[str],
    selected_k_tags: List[str],
    primary_k_tag: str,
) -> tuple[pd.DataFrame, List[str], List[str]]:
    rows: List[Dict[str, Any]] = []
    ndcg_values: List[tuple[str, float]] = []
    for run in runs:
        summary = run.get("summary") or {}
        ranking = summary.get("ranking_metrics", {}) or {}
        flat = _flatten_parameters(run.get("parameters", {}))
        run_id = _run_id_from_dir(run.get("run_dir", ""))

        ndcg_primary = _metric_at(ranking, "ndcg", primary_k_tag)
        if ndcg_primary is not None:
            ndcg_values.append((run_id, ndcg_primary))

        row: Dict[str, Any] = {
            "Run": run.get("run_name", "run"),
            "RunID": run_id,
            "Best@10": bool(run.get("is_best_run")),
            "Cov": float(ranking["coverage"]) if "coverage" in ranking else None,
            "Ent": float(ranking["entropy"]) if "entropy" in ranking else None,
        }
        for k_tag in selected_k_tags:
            row[f"NDCG{k_tag}"] = _metric_at(ranking, "ndcg", k_tag)
            row[f"R{k_tag}"] = _metric_at(ranking, "recall", k_tag)
            row[f"P{k_tag}"] = _metric_at(ranking, "precision", k_tag)
            row[f"HR{k_tag}"] = _metric_at(ranking, "hr", k_tag)
            row[f"MRR{k_tag}"] = _metric_at(ranking, "mrr", k_tag)
            row[f"MAP{k_tag}"] = _metric_at(ranking, "map", k_tag)
        for key in varying_keys:
            row[_short_param_name(key)] = flat.get(key)
        row["SizeMB"] = (summary.get("model_sizes", {}) or {}).get("total_mb")
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols: List[str] = []
    for k_tag in selected_k_tags:
        metric_cols.extend(
            [
                f"NDCG{k_tag}",
                f"R{k_tag}",
                f"P{k_tag}",
                f"HR{k_tag}",
                f"MRR{k_tag}",
                f"MAP{k_tag}",
            ]
        )
    metric_cols.extend(["Cov", "Ent"])
    present_metric_cols = [c for c in metric_cols if c in df.columns]

    best_ndcg_run_id = None
    if ndcg_values:
        best_ndcg_run_id = max(ndcg_values, key=lambda x: x[1])[0]
    best_col = f"Best{primary_k_tag}"
    if best_ndcg_run_id and "RunID" in df.columns:
        df[best_col] = df["RunID"] == best_ndcg_run_id
    else:
        df[best_col] = False

    leading = ["Run", "RunID", "Best@10", best_col]
    param_cols = [
        _short_param_name(k) for k in varying_keys if _short_param_name(k) in df.columns
    ]
    ordered_cols = leading + present_metric_cols + param_cols + ["SizeMB"]
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    return df[ordered_cols], param_cols, present_metric_cols


def _influence_score(param_series: pd.Series, metric_series: pd.Series) -> float:
    valid = ~(param_series.isna() | metric_series.isna())
    x = param_series[valid]
    y = metric_series[valid]
    if len(x) < 3:
        return 0.0

    x_num = pd.to_numeric(x, errors="coerce")
    if x_num.notna().sum() == len(x) and x_num.nunique() > 1:
        corr = x_num.corr(pd.to_numeric(y, errors="coerce"), method="spearman")
        return 0.0 if pd.isna(corr) else abs(float(corr))

    groups = {}
    for xv, yv in zip(x.astype(str), pd.to_numeric(y, errors="coerce")):
        if pd.isna(yv):
            continue
        groups.setdefault(xv, []).append(float(yv))
    if len(groups) <= 1:
        return 0.0

    all_vals = [v for vals in groups.values() for v in vals]
    if len(all_vals) < 3:
        return 0.0
    grand_mean = sum(all_vals) / len(all_vals)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    if ss_total <= 1e-12:
        return 0.0
    ss_between = sum(
        len(vals) * ((sum(vals) / len(vals)) - grand_mean) ** 2
        for vals in groups.values()
    )
    eta_sq = ss_between / ss_total
    return max(0.0, min(1.0, float(eta_sq)))


def _render_influence_matrix(
    df: pd.DataFrame, param_cols: List[str], metric_cols: List[str]
) -> None:
    if not param_cols or not metric_cols:
        st.info("Not enough parameter variation for influence matrix.")
        return

    matrix_rows = []
    for p in param_cols:
        row = {"Parameter": p}
        for m in metric_cols:
            row[m] = _influence_score(df[p], df[m])
        matrix_rows.append(row)
    mdf = pd.DataFrame(matrix_rows)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=mdf[metric_cols].values,
                x=metric_cols,
                y=mdf["Parameter"].tolist(),
                colorscale="Viridis",
                colorbar={"title": "Influence"},
                zmin=0.0,
                zmax=1.0,
                hovertemplate="Param=%{y}<br>Metric=%{x}<br>Influence=%{z:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Parameter Influence Matrix (single-picture overview)",
        xaxis_title="Metrics",
        yaxis_title="Parameters",
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
        height=max(320, 70 + 36 * len(param_cols)),
    )
    st.plotly_chart(fig, width="stretch", key="ablation_influence_matrix")


def _render_ablation_section(runs: List[dict], primary_k_tag: str) -> None:
    selected_k_tags = st.session_state.get("results_selected_k_tags", [primary_k_tag])
    varying_keys = _detect_varying_parameter_keys(runs)

    if varying_keys:
        with st.expander("Parameter Key Legend", expanded=False):
            for key in varying_keys:
                st.markdown(f"`{_short_param_name(key)}`: `{key}`")

    table_df, param_cols, metric_cols = _build_ablation_table(
        runs, varying_keys, selected_k_tags, primary_k_tag
    )
    if table_df.empty:
        st.info("No ablation data available from loaded experiment runs.")
        return

    table_df_display = table_df.copy()
    for col in param_cols:
        if col in table_df_display.columns:
            table_df_display[col] = table_df_display[col].map(
                lambda v: "" if v is None else str(v)
            )

    table_df_display = _arrow_safe_df(table_df_display)
    styled = table_df_display.style
    if metric_cols:
        styled = styled.background_gradient(subset=metric_cols, cmap="Greens", axis=0)
    if "SizeMB" in table_df_display.columns:
        styled = styled.background_gradient(subset=["SizeMB"], cmap="Greys_r", axis=0)

    st.markdown(f"#### Parameter Study Table (primary cutoff {primary_k_tag})")
    st.dataframe(styled, width="stretch", hide_index=True)
    _render_influence_matrix(table_df, param_cols, metric_cols)


def _render_metric_formulas(k_tags: List[str]) -> None:
    shown_ks = ", ".join(k_tags)
    st.markdown("#### Metric Explanations")
    st.caption(f"Cutoffs shown on this page: {shown_ks}")
    st.markdown("All ranking metrics shown here are averaged across evaluated users.")
    st.markdown(
        "Symbols: `u`=user, `Rel_u`=held-out relevant items, `TopK_u`=top-`k` recommendations, `rel_i`=relevance at rank `i`."
    )

    st.markdown(
        "**Recall@k:** Of all relevant items for a user, how many were recovered in top-`k`."
    )
    st.markdown("`Recall@k = |Rel_u intersect TopK_u| / |Rel_u|`")

    st.markdown("**Precision@k:** Of the shown top-`k` items, how many are relevant.")
    st.markdown("`Precision@k = |Rel_u intersect TopK_u| / k`")

    st.markdown(
        "**HR@k (Hit Rate):** Whether at least one relevant item appears in top-`k`."
    )
    st.markdown("`HR@k = 1[|Rel_u intersect TopK_u| > 0]`")

    st.markdown(
        "**NDCG@k:** Ranking quality with position discount; higher-ranked relevant hits matter more."
    )
    st.markdown("`DCG@k = sum_{i=1..k} (rel_i / log2(i+1))`")
    st.markdown("`IDCG@k = sum_{i=1..min(|Rel_u|, k)} (1 / log2(i+1))`")
    st.markdown("`NDCG@k = DCG@k / IDCG@k`")

    st.markdown("**MRR@k:** How early the first relevant recommendation appears.")
    st.markdown("`MRR@k = 1 / rank_first_relevant` (or `0` if none in top-`k`)")

    st.markdown(
        "**MAP@k:** Average precision across relevant hit positions, then averaged across users."
    )
    st.markdown(
        "`AP@k = (1/|Rel_u|) * sum_{i=1..k} Precision@i * rel_i`, `MAP@k = mean_u(AP@k)`"
    )

    st.markdown(
        "**Coverage:** How much of the item catalog gets exposed by recommendations."
    )
    st.markdown("`Coverage = |union_u TopK_u| / |Catalog|`")

    st.markdown(
        "**Entropy:** How evenly recommendation exposure is distributed across items."
    )
    st.markdown("`Entropy = -sum_i p_i log(p_i) / log(|Catalog|)`")


def show() -> None:
    st.title("Model Evaluation Results")
    st.markdown("Comprehensive evaluation of ELSA + SAE using strict run artifacts.")

    try:
        config = st.session_state.get("config")
        training_results = st.session_state.get("training_results")
        if not training_results and config:
            from src.ui.cache import load_training_results

            training_results = load_training_results(
                config, st.session_state.get("selected_result_run_dir")
            )

        if training_results and training_results.get("runs"):
            runs = training_results["runs"]
            runtime_run_dir = training_results.get("default_run_dir")
            view_run_dir = (
                st.session_state.get("results_view_run_dir") or runtime_run_dir
            )

            selected_index_default = next(
                (
                    idx
                    for idx, run in enumerate(runs)
                    if run.get("run_dir") == view_run_dir
                ),
                0,
            )
            current_widget_index = st.session_state.get("results_run_selector")
            if current_widget_index is None or current_widget_index >= len(runs):
                st.session_state.results_run_selector = selected_index_default

            selected_index = st.selectbox(
                "Model / run",
                options=list(range(len(runs))),
                format_func=lambda idx: _run_label(runs[idx]),
                key="results_run_selector",
            )
            selected_run = runs[selected_index]
            st.session_state["results_view_run_dir"] = selected_run.get("run_dir")

            if selected_run.get("run_dir") == runtime_run_dir:
                st.info(
                    "This selected run is also the strict runtime run used by Live Demo and other pages."
                )
            else:
                st.info(
                    "This run selection affects only Results. Live Demo and other pages remain pinned to strict best run."
                )

            st.caption(f"Other run params: {_secondary_param_line(selected_run)}")
            selected_summary = dict(selected_run.get("summary") or {})
            selected_summary["experiment_runs"] = runs
            show_actual_results(selected_summary)
            return

        if training_results and training_results.get("summary"):
            show_actual_results(training_results["summary"])
            return

        st.error("Strict mode: no real results are available.")
    except Exception as e:
        logger.exception("Failed to load results")
        st.error(f"Strict mode: failed to load real results artifacts. {e}")


def show_actual_results(results: Dict[str, Any]) -> None:
    available_ks = _extract_available_k_values(results)
    ks_options = available_ks if available_ks else SUPPORTED_K_VALUES

    default_ks = st.session_state.get("results_global_k_values") or [20]
    default_ks = [k for k in default_ks if k in ks_options]
    if not default_ks:
        default_ks = [20] if 20 in ks_options else [ks_options[0]]

    selected_k_values = st.multiselect(
        "Global @k cutoffs for Results",
        options=ks_options,
        default=default_ks,
        key="results_global_k_values",
        help="Applies to Metrics, Model Comparison, and Ablations.",
    )
    if not selected_k_values:
        selected_k_values = [20] if 20 in ks_options else [ks_options[0]]
        st.session_state["results_global_k_values"] = selected_k_values

    selected_k_values = sorted(set(int(k) for k in selected_k_values))
    selected_k_tags = [f"@{k}" for k in selected_k_values]
    primary_k_tag = f"@{max(selected_k_values)}"
    st.session_state["results_selected_k_tags"] = selected_k_tags

    if available_ks:
        st.caption(
            "Available cutoffs in this run: " + ", ".join(f"@{k}" for k in available_ks)
        )

    tab_metrics, tab_comparison, tab_ablation, tab_speed = st.tabs(
        ["Metrics", "Model Comparison", "Ablations", "Performance"]
    )

    with tab_metrics:
        st.subheader("SAE + ELSA Recommendation Metrics")
        metrics_df = _build_metrics_table(results, selected_k_tags)
        if metrics_df.empty:
            st.info("Ranking metrics are not available in this run summary.")
        else:
            st.dataframe(_arrow_safe_df(metrics_df), width="stretch", hide_index=True)
            chart_df = _build_metrics_chart_df(results, selected_k_tags)
            (
                baseline_df,
                n_items,
                avg_heldout,
                missing_reqs,
            ) = _build_random_baseline_reference(results, selected_k_values)
            if not chart_df.empty:
                fig = px.bar(
                    chart_df,
                    x="Metric",
                    y="Score",
                    color="K",
                    barmode="group",
                    text_auto=".3f",
                    title=f"Core Ranking Metrics at {', '.join(selected_k_tags)}",
                )
                _overlay_random_baseline(fig, baseline_df, selected_k_tags)
                fig.update_layout(
                    margin={"l": 20, "r": 20, "t": 50, "b": 30},
                    height=380,
                    xaxis_title="Metric",
                )
                st.plotly_chart(
                    fig,
                    width="stretch",
                    key=f"metrics_core_bar_{'_'.join(selected_k_tags)}",
                )
            _render_random_baseline_reference(
                baseline_df, n_items, avg_heldout, missing_reqs
            )
            _render_metric_formulas(selected_k_tags)

    with tab_comparison:
        st.subheader("ELSA vs ELSA+SAE")
        comp_df = _build_elsa_vs_sae_df(results, selected_k_tags)
        if comp_df.empty:
            st.info(
                "Comparison metrics are unavailable in this run summary for selected @k cutoffs."
            )
        else:
            show_df = comp_df.copy()
            show_df["ELSA"] = show_df["ELSA"].map(lambda v: f"{v:.4f}")
            show_df["ELSA+SAE"] = show_df["ELSA+SAE"].map(lambda v: f"{v:.4f}")
            show_df["Diff"] = show_df["Diff"].map(lambda v: f"{v:+.4f}")
            show_df["Diff%"] = show_df["Diff%"].map(lambda v: f"{v:+.2f}%")
            st.dataframe(_arrow_safe_df(show_df), width="stretch", hide_index=True)

            long_df = comp_df.melt(
                id_vars=["Metric", "K"],
                value_vars=["ELSA", "ELSA+SAE"],
                var_name="Model",
                value_name="Score",
            )
            fig = px.bar(
                long_df,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                facet_col="K",
                title=f"ELSA vs ELSA+SAE across {', '.join(selected_k_tags)}",
                text_auto=".3f",
            )
            fig.update_layout(margin={"l": 20, "r": 10, "t": 50, "b": 30}, height=420)
            st.plotly_chart(
                fig,
                width="stretch",
                key=f"comparison_grouped_bar_{'_'.join(selected_k_tags)}",
            )

    with tab_ablation:
        st.subheader("Ablations")
        experiment_runs = results.get("experiment_runs", []) or []
        if not experiment_runs:
            st.info("No experiment runs available for ablation analysis.")
        else:
            _render_ablation_section(experiment_runs, primary_k_tag)

    with tab_speed:
        st.subheader("Inference Latency")
        ranking = results.get("ranking_metrics", {}) or {}
        latency = ranking.get("latency", {})
        if not isinstance(latency, dict) or not latency:
            latency = results.get("latency", {}) or {}
        if not latency:
            st.info("Latency data is unavailable in this run summary.")
            return

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean Latency", f"{float(latency.get('mean_ms', 0)):.2f} ms")
        c2.metric("Median Latency", f"{float(latency.get('p50_ms', 0)):.2f} ms")
        c3.metric("P95 Latency", f"{float(latency.get('p95_ms', 0)):.2f} ms")
        c4.metric("Max Latency", f"{float(latency.get('max_ms', 0)):.2f} ms")

        samples = latency.get("samples_ms", [])
        if isinstance(samples, list):
            samples = [float(v) for v in samples if isinstance(v, (int, float))]
        else:
            samples = []

        if samples:
            sample_df = pd.DataFrame(
                {"Sample": list(range(1, len(samples) + 1)), "LatencyMs": samples}
            )
            lfig = px.scatter(
                sample_df,
                x="Sample",
                y="LatencyMs",
                title="Latency Samples with Mean / Median / P95 Markers",
                opacity=0.7,
            )
            lfig.update_traces(marker={"size": 6, "color": "#1f77b4"})
            mean_ms = float(latency.get("mean_ms", 0.0))
            p50_ms = float(latency.get("p50_ms", 0.0))
            p95_ms = float(latency.get("p95_ms", 0.0))
            lfig.add_hline(
                y=mean_ms,
                line_dash="solid",
                line_color="#2ca02c",
                annotation_text=f"Mean {mean_ms:.2f}ms",
            )
            lfig.add_hline(
                y=p50_ms,
                line_dash="dot",
                line_color="#ff7f0e",
                annotation_text=f"Median {p50_ms:.2f}ms",
            )
            lfig.add_hline(
                y=p95_ms,
                line_dash="dash",
                line_color="#d62728",
                annotation_text=f"P95 {p95_ms:.2f}ms",
            )
            lfig.update_layout(
                margin={"l": 20, "r": 20, "t": 50, "b": 30},
                height=360,
                xaxis_title="Sample Index",
                yaxis_title="Latency (ms)",
            )
            st.plotly_chart(lfig, width="stretch", key="latency_samples_scatter")
        else:
            latency_df = pd.DataFrame(
                [
                    ("Mean", float(latency.get("mean_ms", 0))),
                    ("Median (P50)", float(latency.get("p50_ms", 0))),
                    ("P95", float(latency.get("p95_ms", 0))),
                    ("Max", float(latency.get("max_ms", 0))),
                ],
                columns=["Statistic", "LatencyMs"],
            )
            lfig = px.bar(
                latency_df,
                x="Statistic",
                y="LatencyMs",
                color="LatencyMs",
                color_continuous_scale="Blues",
                text_auto=".1f",
                title="Latency Summary (milliseconds)",
            )
            lfig.update_layout(
                margin={"l": 20, "r": 20, "t": 50, "b": 30},
                height=330,
                coloraxis_showscale=False,
            )
            st.plotly_chart(lfig, width="stretch", key="latency_summary_bar")
            st.caption(
                "Per-sample latency points are not stored in this run summary. Run evaluation/training with updated code to populate latency.samples_ms."
            )
