"""Results page — Evaluation metrics and ablations."""

from pathlib import Path
import json
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def _run_label(run: dict) -> str:
    run_name = run.get("run_name", "run")
    summary = run.get("summary") or {}
    ranking_metrics = summary.get("ranking_metrics", {})
    recall = ranking_metrics.get("recall", {}).get("@20")
    ndcg10 = ranking_metrics.get("ndcg", {}).get("@10")
    ndcg = ranking_metrics.get("ndcg", {}).get("@20")
    size_mb = summary.get("model_sizes", {}).get("total_mb")

    parts = [f"{run_name}{' [best]' if run.get('is_best_run') else ''}"]
    if ndcg10 is not None:
        parts.append(f"NDCG@10={ndcg10:.3f}")
    if recall is not None:
        parts.append(f"R@20={recall:.3f}")
    if ndcg is not None:
        parts.append(f"NDCG@20={ndcg:.3f}")
    if size_mb is not None:
        parts.append(f"{size_mb:.2f}MB")
    return " | ".join(parts)


def _show_experiment_overview(runs: list[dict]) -> None:
    rows = []
    for run in runs:
        summary = run.get("summary") or {}
        ranking_metrics = summary.get("ranking_metrics", {})
        params = run.get("parameters", {})
        sae_params = params.get("sae", {}) if isinstance(params, dict) else {}
        elsa_params = params.get("elsa", {}) if isinstance(params, dict) else {}

        rows.append(
            {
                "Run": run.get("run_name", "run"),
                "Experiment": run.get("experiment_name", ""),
                "latent_dim": elsa_params.get("latent_dim"),
                "width_ratio": sae_params.get("width_ratio"),
                "k": sae_params.get("k"),
                "l1_coef": sae_params.get("l1_coef"),
                "Recall@20": ranking_metrics.get("recall", {}).get("@20"),
                "NDCG@20": ranking_metrics.get("ndcg", {}).get("@20"),
                "Model Size MB": summary.get("model_sizes", {}).get("total_mb"),
            }
        )

    if not rows:
        st.info("No experiment runs available for comparison.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    scatter_df = df.dropna(subset=["Model Size MB", "Recall@20"])
    if len(scatter_df) >= 2:
        fig = px.scatter(
            scatter_df,
            x="Model Size MB",
            y="Recall@20",
            color="Experiment" if scatter_df["Experiment"].nunique() > 1 else None,
            hover_name="Run",
            size="NDCG@20" if scatter_df["NDCG@20"].notna().any() else None,
            title="Experiment Comparison: Quality vs Size",
        )
        st.plotly_chart(fig, use_container_width=True)


def _single_summary(results: dict) -> dict:
    summary = dict(results)
    return summary


def show():
    """Display results page."""
    st.title("📊 Model Evaluation Results")

    st.markdown(
        """
    Comprehensive evaluation of the ELSA+SAE recommendation system
    across multiple metrics and ablation studies.
    """
    )

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
            experiment = training_results.get("experiment", {})
            source = training_results.get("source") or "cached results"
            st.info(f"Showing experiment results from: `{source}`")
            st.caption(
                "Runs are sorted by NDCG@10, and the default selection is the best run from the latest experiment."
            )

            current_run_dir = (
                st.session_state.get("selected_result_run_dir")
                or training_results.get("selected_run_dir")
                or training_results.get("default_run_dir")
            )
            selected_index_default = 0
            if current_run_dir:
                selected_index_default = next(
                    (
                        idx
                        for idx, run in enumerate(runs)
                        if run.get("run_dir") == current_run_dir
                    ),
                    0,
                )

            current_widget_index = st.session_state.get("results_run_selector")
            if (
                current_widget_index is None
                or current_widget_index >= len(runs)
                or runs[current_widget_index].get("run_dir") != current_run_dir
            ):
                st.session_state.results_run_selector = selected_index_default

            selected_index = st.selectbox(
                "Model / run",
                options=list(range(len(runs))),
                format_func=lambda idx: _run_label(runs[idx]),
                index=st.session_state.results_run_selector,
                key="results_run_selector",
            )
            selected_run = runs[selected_index]
            st.session_state.selected_result_run_dir = selected_run.get("run_dir")
            if selected_run.get("experiment_dir"):
                st.session_state.selected_result_experiment_dir = selected_run.get(
                    "experiment_dir"
                )
            selected_summary = _single_summary(selected_run.get("summary") or {})
            selected_summary["experiment_runs"] = runs
            selected_summary["experiment"] = experiment

            st.caption(
                f"Selected run directory: `{selected_run.get('run_dir', 'N/A')}`"
            )

            if len(runs) > 1:
                with st.expander("Experiment overview", expanded=True):
                    _show_experiment_overview(runs)

            show_actual_results(selected_summary)
        elif training_results and training_results.get("summary"):
            source = training_results.get("source") or "cached results"
            st.info(f"Showing results from: `{source}`")
            show_actual_results(training_results["summary"])
        else:
            st.warning("No training results available. Using placeholder metrics.")
            show_placeholder_results()

    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        st.error(f"Error loading results: {e}")
        show_placeholder_results()


def show_actual_results(results: dict):
    """Display actual evaluation results from training."""

    tab_metrics, tab_comparison, tab_ablation, tab_speed = st.tabs(
        ["📈 Metrics", "⚖️ Model Comparison", "🔧 Ablations", "⚡ Performance"]
    )

    with tab_metrics:
        st.subheader("SAE+ELSA Recommendation Quality Metrics")

        # Extract ranking metrics from summary
        ranking_metrics = results.get("ranking_metrics", {})
        model_sizes = results.get("model_sizes", {})

        if ranking_metrics:
            # Create metrics table
            metrics_data = []

            # Ranking metrics at k=20
            metric_sources = [
                ("recall", "RECALL@20"),
                ("precision", "PRECISION@20"),
                ("hr", "HIT RATE@20"),  # Primary key from evaluation pipeline
                ("ndcg", "NDCG@20"),
                ("mrr", "MRR@20"),
                ("map", "MAP@20"),
            ]

            for metric_name, metric_label in metric_sources:
                metric_values = ranking_metrics.get(metric_name)
                # Backward-compatible fallback
                if metric_values is None and metric_name == "hr":
                    metric_values = ranking_metrics.get("hit_rate")

                if isinstance(metric_values, dict):
                    value = metric_values.get("@20", 0)
                    metrics_data.append(
                        {
                            "Metric": metric_label,
                            "Value": f"{value:.4f}",
                            "Score": value,
                        }
                    )

            # Coverage and entropy
            if "coverage" in ranking_metrics:
                coverage_val = ranking_metrics["coverage"]
                metrics_data.append(
                    {
                        "Metric": "Coverage",
                        "Value": f"{coverage_val*100:.2f}%",
                        "Score": coverage_val,
                    }
                )

            if "entropy" in ranking_metrics:
                entropy_val = ranking_metrics["entropy"]
                metrics_data.append(
                    {
                        "Metric": "Entropy (Diversity)",
                        "Value": f"{entropy_val:.4f}",
                        "Score": entropy_val,
                    }
                )

            # Model size
            if model_sizes:
                total_size = model_sizes.get("total_mb", 0)
                metrics_data.append(
                    {
                        "Metric": "Total Model Size",
                        "Value": f"{total_size:.2f} MB",
                        "Score": -total_size,  # Negative so lower is better in sorting
                    }
                )

            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)

                # Display metrics in two columns
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Key Metrics")
                    # Show top metrics
                    for idx, row in metrics_df.head(4).iterrows():
                        st.metric(row["Metric"], row["Value"])

                with col2:
                    st.markdown("#### Additional Metrics")
                    # Show remaining metrics
                    for idx, row in metrics_df.iloc[4:].iterrows():
                        st.metric(row["Metric"], row["Value"])

                st.markdown("---")

                # Detailed table
                st.markdown("#### Detailed Metrics")
                st.dataframe(metrics_df[["Metric", "Value"]], use_container_width=True)
            else:
                st.info("Metrics data not available in results.")
        else:
            st.info("Ranking metrics not available. Model may still be training.")

        # Metric explanations in expandable sections
        with st.expander("📖 Metric Explanations", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                **Recall@20** 
                - % of user's liked items appearing in top-20 recommendations
                - Range: 0-1 (higher = better)
                - Good value: > 0.20
                - Measures: Coverage of user preferences
                
                **Precision@20**
                - % of top-20 recommendations that user actually likes
                - Range: 0-1 (higher = better)
                - Good value: > 0.25
                - Measures: Accuracy of recommendations
                
                **Hit Rate@20**
                - % of users who got ≥1 relevant item in top-20
                - Range: 0-1 (higher = better)
                - Good value: > 0.60
                - Measures: Utility for typical users
                
                **NDCG@20**
                - Normalized Discounted Cumulative Gain (ranking quality)
                - Range: 0-1 (higher = better)
                - Good value: > 0.25
                - Penalizes wrong ranking order
                """
                )

            with col2:
                st.markdown(
                    """
                **MRR@20** (Mean Reciprocal Rank)
                - Average position of first relevant item (reciprocal)
                - Range: 0-1 (higher = better)
                - Good value: > 0.40
                - Rewards early matching
                
                **MAP@20** (Mean Average Precision)
                - Average precision across all relevant items in top-20
                - Range: 0-1 (higher = better)
                - Good value: > 0.15
                - Combines precision and recall
                
                **Coverage**
                - % of total items recommended across all users
                - Range: 0-1 (higher = better for discovery)
                - Good value: > 0.20
                - Measures: Item variety in recommendations
                
                **Entropy (Diversity)**
                - Measures how uniformly distributed recommendations are
                - Range: 0-1 (higher = more diverse)
                - Good value: > 0.50
                - Low entropy = concentrated on popular items
                """
                )

    with tab_comparison:
        st.subheader("ELSA vs SAE+ELSA Performance Comparison")

        st.markdown(
            """
        This tab compares the performance of ELSA (collaborative filtering alone)
        vs SAE+ELSA (ELSA with sparse autoencoder compression).
        """
        )

        # Get both metrics
        ranking_metrics_elsa = results.get("ranking_metrics_elsa", {})
        ranking_metrics_sae = results.get("ranking_metrics_sae", {})

        if ranking_metrics_elsa and ranking_metrics_sae:
            # Create comparison table for each metric
            comparison_data = []

            metric_names = {
                "ndcg": "NDCG",
                "recall": "Recall",
                "precision": "Precision",
                "mrr": "MRR",
                "hr": "Hit Rate",
                "map": "MAP",
            }

            for metric_key, metric_display_name in metric_names.items():
                if (
                    metric_key in ranking_metrics_elsa
                    and metric_key in ranking_metrics_sae
                ):
                    elsa_metrics = ranking_metrics_elsa[metric_key]
                    sae_metrics = ranking_metrics_sae[metric_key]

                    for k_str in ["@5", "@10", "@20"]:
                        if k_str in elsa_metrics and k_str in sae_metrics:
                            elsa_val = elsa_metrics[k_str]
                            sae_val = sae_metrics[k_str]
                            diff = sae_val - elsa_val
                            pct_change = (
                                (diff / elsa_val * 100) if elsa_val != 0 else 0.0
                            )

                            comparison_data.append(
                                {
                                    "Metric": metric_display_name,
                                    "K": k_str,
                                    "ELSA": f"{elsa_val:.4f}",
                                    "SAE+ELSA": f"{sae_val:.4f}",
                                    "Difference": f"{diff:+.4f}",
                                    "Change %": f"{pct_change:+.2f}%",
                                    "Better": (
                                        "SAE+ELSA ↑"
                                        if pct_change > 0
                                        else ("ELSA ↓" if pct_change < 0 else "Equal →")
                                    ),
                                }
                            )

            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)

                # Summary statistics
                st.markdown("#### Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Count wins for SAE
                    wins = len(
                        [
                            row
                            for _, row in comparison_df.iterrows()
                            if "↑" in row["Better"]
                        ]
                    )
                    st.metric("SAE+ELSA Wins", f"{wins}/{len(comparison_df)}")

                with col2:
                    # Count wins for ELSA
                    wins_elsa = len(
                        [
                            row
                            for _, row in comparison_df.iterrows()
                            if "↓" in row["Better"]
                        ]
                    )
                    st.metric("ELSA Wins", f"{wins_elsa}/{len(comparison_df)}")

                with col3:
                    # Model sizes
                    model_sizes = results.get("model_sizes", {})
                    if model_sizes:
                        elsa_size = model_sizes.get("elsa_mb", 0)
                        sae_size = model_sizes.get("sae_mb", 0)
                        compression = (
                            (1 - sae_size / elsa_size) * 100 if elsa_size > 0 else 0
                        )
                        st.metric("Compression Gain", f"{compression:.1f}%")
            else:
                st.info(
                    "Comparison data not available. Make sure both ELSA and SAE+ELSA metrics are saved."
                )
        else:
            st.info(
                "⚠️ Comparison metrics not available. Training must save both `ranking_metrics_elsa` and `ranking_metrics_sae`."
            )

    with tab_ablation:
        st.subheader("Ablation: Effect of SAE Sparsity")

        st.info(
            """
        **What is Ablation?** 
        Ablation studies show how model performance changes when we vary hyperparameters.
        Here we test: As SAE sparsity (k = number of active features) increases:
        - Does recommendation quality improve?
        - How much does model size increase?
        - What's the sweet spot trade-off?
        """
        )

        experiment_runs = results.get("experiment_runs", [])

        if experiment_runs:
            st.markdown("#### Experiment runs")
            _show_experiment_overview(experiment_runs)
        elif "ablations" in results:
            ablations_df = pd.DataFrame(results["ablations"])

            # Plot: Quality vs Sparsity
            fig = px.line(
                ablations_df,
                x="k",
                y=["recall@20", "ndcg@100"],
                markers=True,
                title="Recommendation Quality vs SAE Sparsity (k)",
                labels={
                    "k": "Number of Active Features",
                    "value": "Score",
                    "variable": "Metric",
                },
            )
            st.plotly_chart(fig, use_container_width=True, key="ablation_sparsity_line")

            # Table
            st.dataframe(ablations_df, width="stretch")
        else:
            st.info(
                """
            ℹ️ **Ablation data not available yet**
            
            To generate ablation studies, run the experiment sweep config:
            ```bash
            python -m src.train --config configs/experiments.yaml
            ```
            Results are loaded from the latest run artifacts, with local outputs/ used only as an offline fallback.
            """
            )

    with tab_speed:
        st.subheader("Inference Latency & Performance")

        # Try to get latency from ranking metrics
        ranking_metrics = results.get("ranking_metrics", {})
        latency_data = (
            ranking_metrics.get("latency", {})
            if isinstance(ranking_metrics.get("latency"), dict)
            else None
        )

        # Also check top-level latency key
        if not latency_data:
            latency_data = results.get("latency", {})

        if latency_data:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                mean_ms = latency_data.get("mean_ms", 0)
                st.metric("Mean Latency", f"{mean_ms:.2f}ms")
            with col2:
                p50_ms = latency_data.get("p50_ms", 0)
                st.metric("Median Latency", f"{p50_ms:.2f}ms")
            with col3:
                p95_ms = latency_data.get("p95_ms", 0)
                st.metric("P95 Latency", f"{p95_ms:.2f}ms")
            with col4:
                max_ms = latency_data.get("max_ms", 0)
                st.metric("Max Latency", f"{max_ms:.2f}ms")

            st.markdown(
                f"""
            **Latency Interpretation:**
            - **Mean**: Average inference time (typical user experience)
            - **Median (P50)**: Middle-ground representative performance
            - **P95**: 95th percentile (worst 5% of requests)
            - **Max**: Longest single inference
            
            **Performance Targets:**
            - Mean < 100ms = Excellent (real-time responsiveness)
            - Mean < 500ms = Good (interactive)
            - P95 < 1000ms = Acceptable for web apps
            - P95 < 2000ms = Acceptable for pilot
            
            **Current Status:**
            - Mean: {latency_data.get('mean_ms', 'N/A')} ms
            - P95: {latency_data.get('p95_ms', 'N/A')} ms
            """
            )

            # Show estimation note
            st.info(
                "📊 **Note**: These latency measurements are estimates based on numpy operations. "
                "For more accurate benchmarking in production, profile the actual deployment inference pipeline."
            )
        else:
            st.info(
                """
            ℹ️ **Latency data not available yet**
            
            Latency is measured automatically during training evaluation:
            1. After model training completes
            2. Latency is estimated for top-20 inference operations
            3. Results appear in this Performance tab
            
            Each inference (encoding + ranking) is measured in milliseconds.
            """
            )


def show_placeholder_results():
    """Display placeholder results (demo data)."""

    st.markdown("### 📊 Sample Results (Placeholder Data)")

    tab_metrics, tab_ablation, tab_speed = st.tabs(
        ["📈 Metrics", "🔧 Ablations", "⚡ Performance"]
    )

    with tab_metrics:
        st.subheader("Recommendation Quality")

        metrics_data = {
            "Model": [
                "ELSA (Dense)",
                "SAE k=32",
                "SAE k=64",
                "SAE k=128",
            ],
            "Recall@20": [0.450, 0.420, 0.445, 0.450],
            "Precision@20": [0.380, 0.360, 0.375, 0.385],
            "Hit Rate@20": [0.720, 0.680, 0.710, 0.725],
            "NDCG@20": [0.580, 0.550, 0.570, 0.580],
            "Coverage (%)": [85.2, 45.8, 62.3, 78.5],
            "Model Size (MB)": [12.5, 1.2, 2.4, 4.8],
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, width="stretch")

        # Metric explanations in expandable sections
        with st.expander("📖 Metric Explanations", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                **Recall@20** 
                - % of user's liked items appearing in top-20 recommendations
                - Range: 0-100% (higher = better)
                - Target: > 20%
                
                **Precision@20**
                - % of recommended items that are actually relevant
                - Range: 0-100% (higher = better)
                - Target: > 25%
                
                **Hit Rate@20**
                - % of users who got ≥1 relevant item in top-20
                - Range: 0-100% (higher = better)
                - Target: > 60%
                """
                )

            with col2:
                st.markdown(
                    """
                **NDCG@20** (Normalized Discounted Cumulative Gain)
                - Quality of ranking (best items ranked first)
                - Range: 0-1 (higher = better)
                - Target: > 0.25
                
                **Coverage (%)**
                - % of total items recommended across all users
                - Range: 0-100% (higher = better discovery)
                - Target: > 20%
                
                **Model Size**
                - Compressed model size (MB)
                - Smaller = faster inference, larger = better accuracy
                """
                )

        # Quality vs Model Size trade-off
        fig = px.scatter(
            metrics_df,
            x="Model Size (MB)",
            y="Recall@20",
            size="NDCG@20",
            hover_name="Model",
            title="Quality vs Model Size (bubble area = NDCG@20)",
            labels={
                "Model Size (MB)": "Model Size (MB)",
                "Recall@20": "Recall@20",
            },
        )
        st.plotly_chart(fig, use_container_width=True, key="quality_vs_size_scatter")

        st.markdown(
            """
        **Key Insights:**
        - SAE substantially reduces model size (12.5MB → 1.2MB for k=32)
        - Quality remains good (Recall@20: 0.42-0.45)
        - Coverage improves with larger k (more features = more POIs)
        - Sweet spot: k=64 balances sparsity and performance
        """
        )

    with tab_ablation:
        st.subheader("Ablation: SAE Sparsity Analysis")

        st.info(
            """
        **What is Ablation?** 
        Ablation studies show how model performance changes when we vary hyperparameters.
        Here we test: As SAE sparsity (k = number of active features) increases:
        - Does recommendation quality improve?
        - How much does model size increase?
        - What's the sweet spot trade-off?
        """
        )

        ablation_data = {
            "k": [16, 32, 64, 128, 256],
            "Recall@20": [0.410, 0.420, 0.445, 0.450, 0.448],
            "Precision@20": [0.350, 0.360, 0.375, 0.385, 0.382],
            "Hit Rate@20": [0.680, 0.700, 0.720, 0.735, 0.730],
            "NDCG@20": [0.540, 0.550, 0.570, 0.580, 0.575],
            "Coverage (%)": [20.5, 45.8, 62.3, 78.5, 91.2],
            "Model Size (MB)": [0.6, 1.2, 2.4, 4.8, 9.6],
        }

        ablation_df = pd.DataFrame(ablation_data)
        st.dataframe(ablation_df, width="stretch")

        # Dual-axis plot
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=ablation_df["k"],
                y=ablation_df["Recall@20"],
                mode="lines+markers",
                name="Recall@20",
                yaxis="y1",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=ablation_df["k"],
                y=ablation_df["Model Size (MB)"],
                mode="lines+markers",
                name="Model Size (MB)",
                yaxis="y2",
            )
        )

        fig.update_layout(
            title="SAE Sparsity: Quality and Size Trade-off",
            xaxis=dict(title="Number of Active Features (k)"),
            yaxis=dict(
                title="Recall@20",
                title_font=dict(color="#1f77b4"),
            ),
            yaxis2=dict(
                title="Model Size (MB)",
                title_font=dict(color="#ff7f0e"),
                anchor="x",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )

        st.plotly_chart(
            fig, use_container_width=True, key="sparsity_tradeoff_dual_axis"
        )

        st.markdown(
            """
        **Interpretation:**
        - **k=32** (blue dot): Good balance - quality 0.42 with small 1.2MB model
        - **k=64** (sweet spot): Slightly better quality (0.45) at 2.4MB
        - **k=128+**: Diminishing returns - model grows 2x vs 5% quality gain
        """
        )

    with tab_speed:
        st.subheader("Inference Latency & Performance")

        st.info(
            """
        **What to know about latency:**
        - Measured in milliseconds (ms)
        - Includes: user encoding + feature steering + item ranking
        - Critical for interactive demos (target: <1000ms)
        """
        )

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Latency", "847 ms")
        with col2:
            st.metric("P95 Latency", "1234 ms")
        with col3:
            st.metric("P99 Latency", "1650 ms")
        with col4:
            st.metric("Max Latency", "1899 ms")

        # Latency histogram (milliseconds)
        import numpy as np

        latencies = np.random.normal(847, 300, 1000)
        latencies = np.clip(latencies, 100, 3000)

        fig = px.histogram(
            x=latencies,
            nbins=30,
            title="Inference Time Distribution (1000 samples)",
            labels={"x": "Latency (milliseconds)"},
            color_discrete_sequence=["#636EFA"],
        )

        fig.add_vline(
            847,
            line_dash="dash",
            line_color="green",
            annotation_text="Mean",
            annotation_position="top right",
        )
        fig.add_vline(
            1234,
            line_dash="dash",
            line_color="orange",
            annotation_text="P95",
            annotation_position="top right",
        )

        st.plotly_chart(fig, use_container_width=True, key="latency_histogram")

        st.markdown(
            """
        **Performance Breakdown:**
        - User encoding (ELSA): ~150ms
        - Feature steering (SAE): ~50ms  
        - Item scoring (matrix mult): ~650ms
        - Total: ~847ms per recommendation
        
        **Latency Goals:**
        - ✅ Mean < 1000ms: **Interactive** (current)
        - ✅ P95 < 1500ms: **Good** (current)
        - Can optimize further with batching/GPU
        """
        )
