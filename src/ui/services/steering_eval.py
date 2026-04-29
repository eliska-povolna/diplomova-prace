"""Helpers for logging and plotting steering evaluation data.

Example CSV row (dummy):
2026-04-29T12:00:00Z,run_123,user_42,12,weighted-category,neuron:17,0.500,0.1432,0.1821,0.2500,0.3333,0.0840,0.0412,0.0833,0.0100,0.0234,"{\"17\": 0.5}"
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

DEFAULT_STEERING_EVAL_CSV = Path("outputs") / "steering_eval.csv"
DEFAULT_STEERING_EVAL_OUTDIR = Path("img") / "generated"
STEERING_EVAL_TRADEOFF_PLOT = "steering_tradeoff_cpr_vs_ndcg.png"
STEERING_EVAL_STRENGTH_PLOT = "steering_vs_strength.png"
DEFAULT_STEERING_EVAL_TABLE = "steering_eval"


def _ensure_steering_eval_table(engine) -> None:
    """Create the steering_eval table if it does not exist yet."""

    from sqlalchemy import text

    cols_ddl = (
        "timestamp_iso TIMESTAMP WITH TIME ZONE,"
        "run_id TEXT, user_id TEXT, k INTEGER, method TEXT, label TEXT,"
        "strength DOUBLE PRECISION, ndcg_before DOUBLE PRECISION, ndcg_after DOUBLE PRECISION,"
        "cpr_before DOUBLE PRECISION, cpr_after DOUBLE PRECISION, activation_before DOUBLE PRECISION,"
        "activation_after DOUBLE PRECISION, delta_ndcg DOUBLE PRECISION, delta_cpr DOUBLE PRECISION,"
        "delta_activation DOUBLE PRECISION, weights_changed_json TEXT"
    )
    create_sql = f"CREATE TABLE IF NOT EXISTS {DEFAULT_STEERING_EVAL_TABLE} ({cols_ddl})"
    with engine.begin() as conn:
        conn.execute(text(create_sql))

CSV_COLUMNS = [
    "timestamp_iso",
    "run_id",
    "user_id",
    "k",
    "method",
    "label",
    "strength",
    "ndcg_before",
    "ndcg_after",
    "cpr_before",
    "cpr_after",
    "activation_before",
    "activation_after",
    "delta_ndcg",
    "delta_cpr",
    "delta_activation",
    "weights_changed_json",
]


def append_steering_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    csv_path: Path | str = DEFAULT_STEERING_EVAL_CSV,
) -> None:
    """Append steering evaluation rows to CSV, creating the file and header if needed."""
    if not rows:
        return

    # Try Cloud SQL first (if configured). If anything fails, fall back to CSV.
    try:
        from .secrets_helper import get_cloudsql_config
        from .cloud_sql_helper import CloudSQLHelper

        cfg = get_cloudsql_config()
        if all([cfg.get("instance"), cfg.get("database"), cfg.get("user"), cfg.get("password")]):
            sql_helper = CloudSQLHelper(
                instance_connection_name=cfg.get("instance"),
                database=cfg.get("database"),
                user=cfg.get("user"),
                password=cfg.get("password"),
            )
            engine = getattr(sql_helper, "engine", None)
            if engine is not None:
                from sqlalchemy import text

                _ensure_steering_eval_table(engine)

                with engine.begin() as conn:
                    # Prepare rows for insert: normalize keys and ensure timestamp present
                    insert_cols = ", ".join(CSV_COLUMNS)
                    placeholders = ", ".join([f":" + c for c in CSV_COLUMNS])
                    insert_sql = text(f"INSERT INTO {DEFAULT_STEERING_EVAL_TABLE} ({insert_cols}) VALUES ({placeholders})")

                    normalized_rows = []
                    for r in rows:
                        normalized = {col: (r.get(col) if r.get(col) is not None else None) for col in CSV_COLUMNS}
                        normalized_rows.append(normalized)

                    if normalized_rows:
                        conn.execute(insert_sql, normalized_rows)
                        logger.info(
                            "Wrote %d steering eval row(s) to Cloud SQL table %s",
                            len(normalized_rows),
                            DEFAULT_STEERING_EVAL_TABLE,
                        )
                        return
    except Exception:
        logger.debug("Cloud SQL write failed; falling back to local CSV", exc_info=True)

    # Fallback: local CSV
    output_path = Path(csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            normalized = {column: row.get(column, "") for column in CSV_COLUMNS}
            writer.writerow(normalized)

    logger.info(
        "Wrote %d steering eval row(s) to local CSV %s",
        len(rows),
        output_path,
    )


def load_steering_eval_dataframe(
    csv_path: Path | str = DEFAULT_STEERING_EVAL_CSV,
    max_rows: int = 500,
) -> pd.DataFrame:
    """Load steering evaluation data and keep only the most recent rows for plotting/UI.

    The default max_rows limit keeps the charts responsive when the CSV grows large.
    """

    # Try loading from Cloud SQL first; fallback to CSV file
    try:
        from .secrets_helper import get_cloudsql_config
        from .cloud_sql_helper import CloudSQLHelper

        cfg = get_cloudsql_config()
        if all([cfg.get("instance"), cfg.get("database"), cfg.get("user"), cfg.get("password")]):
            sql_helper = CloudSQLHelper(
                instance_connection_name=cfg.get("instance"),
                database=cfg.get("database"),
                user=cfg.get("user"),
                password=cfg.get("password"),
            )
            engine = getattr(sql_helper, "engine", None)
            if engine is not None:
                _ensure_steering_eval_table(engine)
                query = f"SELECT * FROM {DEFAULT_STEERING_EVAL_TABLE} ORDER BY timestamp_iso DESC LIMIT {int(max_rows) if max_rows>0 else 1000000}"
                df = pd.read_sql(query, con=engine)
                # SQL returned newest-first; invert to have chronological order like CSV loader
                if not df.empty:
                    df = df.iloc[::-1].reset_index(drop=True)
                else:
                    df = df.reindex(columns=CSV_COLUMNS)
                # Normalize types below
                for column in CSV_COLUMNS:
                    if column not in df.columns:
                        df[column] = pd.NA
                df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], errors="coerce", utc=True)
                df["k"] = pd.to_numeric(df["k"], errors="coerce")
                numeric_columns = [
                    "strength",
                    "ndcg_before",
                    "ndcg_after",
                    "cpr_before",
                    "cpr_after",
                    "activation_before",
                    "activation_after",
                    "delta_ndcg",
                    "delta_cpr",
                    "delta_activation",
                ]
                for column in numeric_columns:
                    df[column] = pd.to_numeric(df[column], errors="coerce")
                return df.reset_index(drop=True)
    except Exception:
        logger.debug("Cloud SQL read failed; falling back to CSV", exc_info=True)

    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.read_csv(path)
    if df.empty:
        return df.reindex(columns=CSV_COLUMNS)

    for column in CSV_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[CSV_COLUMNS].copy()
    df["timestamp_iso"] = pd.to_datetime(df["timestamp_iso"], errors="coerce", utc=True)
    df["k"] = pd.to_numeric(df["k"], errors="coerce")
    numeric_columns = [
        "strength",
        "ndcg_before",
        "ndcg_after",
        "cpr_before",
        "cpr_after",
        "activation_before",
        "activation_after",
        "delta_ndcg",
        "delta_cpr",
        "delta_activation",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("timestamp_iso", na_position="first")
    if max_rows > 0 and len(df) > max_rows:
        # Keep only the most recent rows so the plots stay fast even when the CSV grows.
        df = df.tail(max_rows)
    return df.reset_index(drop=True)


def filter_steering_eval_dataframe(
    df: pd.DataFrame,
    *,
    k_value: int | str = "All",
    method_value: str = "All",
    label_contains: str = "",
) -> pd.DataFrame:
    """Apply UI-friendly filters to the steering evaluation dataframe."""

    if df.empty:
        return df

    filtered = df.copy()

    if k_value != "All":
        filtered = filtered[filtered["k"] == pd.to_numeric(k_value, errors="coerce")]

    if method_value and method_value != "All":
        filtered = filtered[filtered["method"].astype(str) == str(method_value)]

    label_query = str(label_contains or "").strip()
    if label_query:
        filtered = filtered[
            filtered["label"].astype(str).str.contains(label_query, case=False, na=False)
        ]

    return filtered.reset_index(drop=True)


def _annotate_no_data(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()


def _safe_group_means(
    df: pd.DataFrame,
    x_column: str,
    y_columns: Sequence[str],
    round_digits: int = 3,
) -> pd.DataFrame:
    if df.empty or x_column not in df.columns:
        return pd.DataFrame(columns=[x_column, *y_columns])

    work = df[[x_column, *y_columns]].copy()
    work = work.dropna(subset=[x_column])
    if work.empty:
        return pd.DataFrame(columns=[x_column, *y_columns])

    work[x_column] = pd.to_numeric(work[x_column], errors="coerce").round(round_digits)
    work = work.dropna(subset=[x_column])
    if work.empty:
        return pd.DataFrame(columns=[x_column, *y_columns])

    grouped = work.groupby(x_column, as_index=False)[list(y_columns)].mean()
    return grouped.sort_values(x_column)


def generate_steering_eval_plots(
    df: pd.DataFrame,
    outdir: Path | str = DEFAULT_STEERING_EVAL_OUTDIR,
    *,
    k_filter: int | None = 12,
    lang: str = "en",
) -> dict[str, Path]:
    """Create steering evaluation plots and save them as PNG files.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Steering evaluation dataframe
    outdir : Path | str
        Output directory for PNG files
    k_filter : int | None
        Filter rows by k value before plotting
    lang : str
        Language for labels ('en' or 'cs')
    """

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    if k_filter is not None:
        plot_df = plot_df[pd.to_numeric(plot_df["k"], errors="coerce") == int(k_filter)]

    # Choose filename based on language
    strength_plot = "steering_vs_strength.png" if lang == "en" else "steering_vs_strength_cz.png"
    strength_path = output_dir / strength_plot

    _plot_strength(plot_df, strength_path, k_filter=k_filter, lang=lang)

    return {"strength": strength_path}


def _plot_tradeoff(df: pd.DataFrame, outpath: Path, *, k_filter: int | None) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df = df[["cpr_after", "ndcg_after", "strength", "method"]].copy()
    
    # Separate neuron-based (has CPR) and LLM-based (no CPR) rows
    neuron_df = plot_df[
        plot_df["method"].astype(str).str.startswith("neuron", na=False)
    ].copy()
    neuron_df = neuron_df.dropna(subset=["cpr_after", "ndcg_after"])
    
    llm_df = plot_df[
        plot_df["method"].astype(str).str.startswith("llm", na=False)
    ].copy()
    llm_df = llm_df.dropna(subset=["ndcg_after"])

    if neuron_df.empty and llm_df.empty:
        _annotate_no_data(ax, "No steering rows available for this filter")
    else:
        # Plot neuron-based rows (has CPR on X-axis)
        if not neuron_df.empty:
            scatter1 = ax.scatter(
                neuron_df["cpr_after"],
                neuron_df["ndcg_after"],
                c=neuron_df["strength"],
                cmap="viridis",
                alpha=0.85,
                s=60,
                edgecolors="none",
                label="neuron-based",
            )
            cbar = fig.colorbar(scatter1, ax=ax)
            cbar.set_label("Strength")
        
        # Plot LLM-based rows (no CPR, show on left side at x=0)
        if not llm_df.empty:
            ax.scatter(
                [0] * len(llm_df),
                llm_df["ndcg_after"],
                c="red",
                alpha=0.5,
                s=60,
                marker="x",
                edgecolors="darkred",
                linewidths=2,
                label="LLM-based (CPR N/A)",
            )
        
        ax.set_xlabel("CPR after")
        ax.set_ylabel("NDCG after")
        ax.set_title(
            f"Steering trade-off: CPR vs NDCG"
            + (f" (k={k_filter})" if k_filter is not None else "")
        )
        ax.grid(True, alpha=0.25)
        if not neuron_df.empty and not llm_df.empty:
            ax.legend(loc="best")
        ax.set_xlim(-0.1, 1.1)

    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_strength(df: pd.DataFrame, outpath: Path, *, k_filter: int | None, lang: str = "en") -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 5.5))

    left_df = _safe_group_means(df, "strength", ["cpr_after", "ndcg_after"])
    right_df = _safe_group_means(df, "delta_activation", ["delta_cpr", "delta_ndcg"])

    # Language-specific labels
    labels = {
        "en": {
            "strength": "Strength",
            "cpr_after": "mean CPR after",
            "ndcg_after": "mean NDCG after",
            "metric": "Mean metric",
            "delta_activation": "Delta activation",
            "delta_cpr": "mean delta CPR",
            "delta_ndcg": "mean delta NDCG",
            "delta_metric": "Mean delta metric",
            "title_strength": "Mean metrics by strength",
            "title_delta": "Metric deltas vs activation shift",
        },
        "cs": {
            "strength": "Síla steeringu",
            "cpr_after": "průměr CPR po",
            "ndcg_after": "průměr NDCG po",
            "metric": "Průměrná metrika",
            "delta_activation": "Změna aktivace",
            "delta_cpr": "průměr Δ CPR",
            "delta_ndcg": "průměr Δ NDCG",
            "delta_metric": "Průměrná změna metriky",
            "title_strength": "Průměrné metriky podle síly steeringu",
            "title_delta": "Změny metrik vs posun aktivace",
        },
    }
    
    l = labels.get(lang, labels["en"])

    if left_df.empty:
        _annotate_no_data(ax_left, "No data for strength grouping" if lang == "en" else "Žádná data pro seskupení podle síly")
    else:
        ax_left.plot(left_df["strength"], left_df["cpr_after"], marker="o", label=l["cpr_after"])
        ax_left.plot(left_df["strength"], left_df["ndcg_after"], marker="o", label=l["ndcg_after"])
        
        # Add regression lines for both metrics
        if len(left_df) >= 2:
            # CPR regression line
            cpr_poly = np.polyfit(left_df["strength"].dropna(), left_df["cpr_after"].dropna(), 1)
            cpr_line = np.poly1d(cpr_poly)
            x_range_cpr = np.linspace(left_df["strength"].min(), left_df["strength"].max(), 100)
            ax_left.plot(x_range_cpr, cpr_line(x_range_cpr), "--", alpha=0.5, color="C0", linewidth=1.5)
            
            # NDCG regression line
            ndcg_poly = np.polyfit(left_df["strength"].dropna(), left_df["ndcg_after"].dropna(), 1)
            ndcg_line = np.poly1d(ndcg_poly)
            x_range_ndcg = np.linspace(left_df["strength"].min(), left_df["strength"].max(), 100)
            ax_left.plot(x_range_ndcg, ndcg_line(x_range_ndcg), "--", alpha=0.5, color="C1", linewidth=1.5)
        
        ax_left.set_xlabel(l["strength"])
        ax_left.set_ylabel(l["metric"])
        ax_left.set_title(
            l["title_strength"] + (f" (k={k_filter})" if k_filter is not None else "")
        )
        ax_left.grid(True, alpha=0.25)
        ax_left.legend()

    if right_df.empty:
        _annotate_no_data(ax_right, "No data for delta activation grouping" if lang == "en" else "Žádná data pro seskupení podle změny aktivace")
    else:
        ax_right.plot(
            right_df["delta_activation"],
            right_df["delta_cpr"],
            marker="o",
            label=l["delta_cpr"],
        )
        ax_right.plot(
            right_df["delta_activation"],
            right_df["delta_ndcg"],
            marker="o",
            label=l["delta_ndcg"],
        )
        
        # Add regression lines for both metrics
        if len(right_df) >= 2:
            # Delta CPR regression line
            cpr_poly = np.polyfit(right_df["delta_activation"].dropna(), right_df["delta_cpr"].dropna(), 1)
            cpr_line = np.poly1d(cpr_poly)
            x_range_cpr = np.linspace(right_df["delta_activation"].min(), right_df["delta_activation"].max(), 100)
            ax_right.plot(x_range_cpr, cpr_line(x_range_cpr), "--", alpha=0.5, color="C0", linewidth=1.5)
            
            # Delta NDCG regression line
            ndcg_poly = np.polyfit(right_df["delta_activation"].dropna(), right_df["delta_ndcg"].dropna(), 1)
            ndcg_line = np.poly1d(ndcg_poly)
            x_range_ndcg = np.linspace(right_df["delta_activation"].min(), right_df["delta_activation"].max(), 100)
            ax_right.plot(x_range_ndcg, ndcg_line(x_range_ndcg), "--", alpha=0.5, color="C1", linewidth=1.5)
        
        ax_right.set_xlabel(l["delta_activation"])
        ax_right.set_ylabel(l["delta_metric"])
        ax_right.set_title(l["title_delta"])
        ax_right.grid(True, alpha=0.25)
        ax_right.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_presence_ratio(
    recommendations: Iterable[Mapping[str, Any]],
    target_neuron_ids: Iterable[int],
    *,
    top_features: int = 3,
) -> float:
    """Compute the fraction of recommendations where the target feature appears among top contributors."""

    target_set = {int(idx) for idx in target_neuron_ids if idx is not None}
    if not target_set:
        return float("nan")

    recommendations_list = list(recommendations or [])
    if not recommendations_list:
        return float("nan")

    hits = 0
    considered = 0
    for recommendation in recommendations_list:
        contributors = recommendation.get("contributing_neurons") or []
        if not contributors:
            continue
        considered += 1
        top_contributors = contributors[:top_features]
        contributor_ids = {
            int(entry.get("neuron_idx"))
            for entry in top_contributors
            if entry.get("neuron_idx") is not None
        }
        if target_set.intersection(contributor_ids):
            hits += 1

    if considered == 0:
        return float("nan")
    return float(hits / considered)


def build_ndcg_from_recommendations(
    recommendations: Sequence[Mapping[str, Any]],
    relevant_item_ids: Iterable[int],
    *,
    k: int,
) -> float:
    """Compute NDCG@k from a ranked recommendation list and binary relevance set."""

    relevant_set = {int(item_id) for item_id in relevant_item_ids if item_id is not None}
    ranked_items = [
        int(row["item_id"] if row.get("item_id") is not None else row.get("poi_idx"))
        for row in recommendations[:k]
        if row.get("item_id") is not None or row.get("poi_idx") is not None
    ]
    if not ranked_items:
        return float("nan")

    hits = np.array([1.0 if item_id in relevant_set else 0.0 for item_id in ranked_items])
    if float(hits.sum()) <= 0:
        return 0.0

    discounts = np.log2(np.arange(2, len(hits) + 2))
    dcg = float((hits / discounts).sum())

    ideal_hits = np.array([1.0] * min(len(relevant_set), k) + [0.0] * max(0, k - len(relevant_set)))
    ideal_discounts = np.log2(np.arange(2, len(ideal_hits) + 2))
    idcg = float((ideal_hits / ideal_discounts).sum())
    return float(dcg / idcg) if idcg > 0 else 0.0
