"""Generate thesis visualization charts from Yelp database.

This script creates and saves figures used in the thesis (e.g. dataset statistics
charts). It queries the Yelp database directly and exports PNG images to a 
specified output directory (typically the LaTeX figures folder).

The charts show:
- State and city distribution of businesses
- Top business categories
- Rating distribution
- Activity metrics

Usage:
    python scripts/generate_thesis_charts.py
"""

import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    repo = Path(r"c:\Users\elisk\Desktop\2024-25\Diplomka\Github\Diplomov-pr-ce")
    latex_img = Path(
        r"c:\Users\elisk\Desktop\2024-25\Diplomka\Github\Diplomov-pr-ce-latex\img\generated"
    )
    latex_img.mkdir(parents=True, exist_ok=True)

    db_path = Path(r"c:\Users\elisk\Desktop\2024-25\Diplomka\Yelp-JSON\yelp.duckdb")
    con = duckdb.connect(str(db_path))

    plt.style.use("seaborn-v0_8-whitegrid")

    q1 = """
    SELECT state, COUNT(*) AS n_businesses
    FROM yelp_business
    WHERE state IS NOT NULL AND state <> ''
    GROUP BY state
    ORDER BY n_businesses DESC
    LIMIT 15
    """
    df1 = con.execute(q1).df().sort_values("n_businesses")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df1["state"], df1["n_businesses"], color="#3b82f6")
    ax.set_xlabel("Počet podniků")
    ax.set_ylabel("Stát")
    ax.set_title("Globální distribuce podniků podle státu (Top 15)")
    fig.tight_layout()
    fig.savefig(latex_img / "dataset_state_distribution_global.png", dpi=220)
    plt.close(fig)

    q2 = """
    SELECT city, COUNT(*) AS n_businesses
    FROM yelp_business
    WHERE city IS NOT NULL AND city <> ''
    GROUP BY city
    ORDER BY n_businesses DESC
    LIMIT 15
    """
    df2 = con.execute(q2).df().sort_values("n_businesses")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df2["city"], df2["n_businesses"], color="#0ea5a4")
    ax.set_xlabel("Počet podniků")
    ax.set_ylabel("Město")
    ax.set_title("Globální distribuce podniků podle měst (Top 15)")
    fig.tight_layout()
    fig.savefig(latex_img / "dataset_city_distribution_global.png", dpi=220)
    plt.close(fig)

    q3 = """
    WITH exploded AS (
      SELECT TRIM(t.cat) AS category
      FROM yelp_business b,
           UNNEST(string_split(COALESCE(b.categories, ''), ',')) AS t(cat)
    )
    SELECT category, COUNT(*) AS n_businesses
    FROM exploded
    WHERE category IS NOT NULL AND category <> ''
    GROUP BY category
    ORDER BY n_businesses DESC
    LIMIT 15
    """
    df3 = con.execute(q3).df().sort_values("n_businesses")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df3["category"], df3["n_businesses"], color="#f59e0b")
    ax.set_xlabel("Počet podniků")
    ax.set_ylabel("Kategorie")
    ax.set_title("Globální nejčastější kategorie podniků (Top 15)")
    fig.tight_layout()
    fig.savefig(latex_img / "dataset_top_categories_global.png", dpi=220)
    plt.close(fig)

    q4 = """
    SELECT CAST(TRY_CAST(stars AS DOUBLE) AS INTEGER) AS stars, COUNT(*) AS review_count
    FROM yelp_review
    GROUP BY stars
    ORDER BY stars
    """
    df4 = con.execute(q4).df()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(df4["stars"].astype(str), df4["review_count"], color="#8b5cf6")
    ax.set_xlabel("Hodnocení (hvězdy)")
    ax.set_ylabel("Počet recenzí")
    ax.set_title("Globální distribuce hodnocení recenzí")
    fig.tight_layout()
    fig.savefig(latex_img / "dataset_rating_distribution_global.png", dpi=220)
    plt.close(fig)

    q5 = """
    SELECT EXTRACT(YEAR FROM date) AS year, COUNT(*) AS review_count
    FROM yelp_review
    GROUP BY year
    ORDER BY year
    """
    df5 = con.execute(q5).df()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df5["year"], df5["review_count"], marker="o", color="#ef4444")
    ax.set_xlabel("Rok")
    ax.set_ylabel("Počet recenzí")
    ax.set_title("Globální objem recenzí v čase")
    fig.tight_layout()
    fig.savefig(latex_img / "dataset_review_volume_global.png", dpi=220)
    plt.close(fig)

    summary_path = repo / "outputs" / "20260427_030923" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["ranking_metrics"]
    kvals = ["@10", "@20", "@50"]
    metric_names = ["recall", "precision", "hr", "ndcg", "mrr", "map"]
    labels = ["Recall", "Precision", "HR", "NDCG", "MRR", "MAP"]

    rows = []
    for k in kvals:
        for metric_name, label in zip(metric_names, labels):
            rows.append({"K": k, "Metric": label, "Value": float(metrics[metric_name][k])})
    dfm = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for i, k in enumerate(kvals):
        sub = dfm[dfm["K"] == k]
        x = range(len(labels))
        width = 0.22
        offs = (i - 1) * width
        ax.bar([v + offs for v in x], sub["Value"].values, width=width, label=k)
    ax.set_xticks(list(range(len(labels))))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Hodnota metriky")
    ax.set_title("Offline metriky modelu ELSA+SAE pro různé hodnoty K")
    ax.legend(title="Top-K")
    fig.tight_layout()
    fig.savefig(latex_img / "results_offline_metrics_by_k.png", dpi=220)
    plt.close(fig)

    k = "@20"
    comp = pd.DataFrame(
        {
            "Metric": ["Recall", "Precision", "HR", "NDCG", "MRR", "MAP"],
            "ELSA": [
                summary["ranking_metrics_elsa"]["recall"][k],
                summary["ranking_metrics_elsa"]["precision"][k],
                summary["ranking_metrics_elsa"]["hr"][k],
                summary["ranking_metrics_elsa"]["ndcg"][k],
                summary["ranking_metrics_elsa"]["mrr"][k],
                summary["ranking_metrics_elsa"]["map"][k],
            ],
            "ELSA+SAE": [
                summary["ranking_metrics_sae"]["recall"][k],
                summary["ranking_metrics_sae"]["precision"][k],
                summary["ranking_metrics_sae"]["hr"][k],
                summary["ranking_metrics_sae"]["ndcg"][k],
                summary["ranking_metrics_sae"]["mrr"][k],
                summary["ranking_metrics_sae"]["map"][k],
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = range(len(comp))
    w = 0.35
    ax.bar([v - w / 2 for v in x], comp["ELSA"], width=w, label="ELSA", color="#94a3b8")
    ax.bar(
        [v + w / 2 for v in x],
        comp["ELSA+SAE"],
        width=w,
        label="ELSA+SAE",
        color="#22c55e",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(comp["Metric"])
    ax.set_ylabel("Hodnota metriky")
    ax.set_title("Srovnání ELSA vs ELSA+SAE (K=20)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(latex_img / "results_elsa_vs_sae_k20.png", dpi=220)
    plt.close(fig)

    print(f"Generated charts in {latex_img}")


if __name__ == "__main__":
    main()
