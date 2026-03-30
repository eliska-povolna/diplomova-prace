"""Results page — Evaluation metrics and ablations."""

from pathlib import Path
import json
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)


def show():
    """Display results page."""
    st.title("📊 Model Evaluation Results")

    st.markdown(
        """
    Comprehensive evaluation of the ELSA+SAE recommendation system
    across multiple metrics and ablation studies.
    """
    )

    # Try to load results from latest run
    try:
        results_dir = Path(__file__).parent.parent.parent.parent / "outputs"

        # Find latest run directory
        if not results_dir.exists():
            st.warning("No outputs directory found. Using placeholder metrics.")
            show_placeholder_results()
            return

        run_dirs = sorted(
            [d for d in results_dir.glob("*") if d.is_dir()],
            key=lambda x: x.name,
            reverse=True,
        )

        if not run_dirs:
            st.warning("No training runs found. Using placeholder metrics.")
            show_placeholder_results()
            return

        latest_run = run_dirs[0]
        st.info(f"Showing results from: `{latest_run.name}`")

        # Load results JSON
        results_file = latest_run / "training_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            show_actual_results(results)
        else:
            st.warning("No training_results.json found. Using placeholder data.")
            show_placeholder_results()

    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        st.error(f"Error loading results: {e}")
        show_placeholder_results()


def show_actual_results(results: dict):
    """Display actual evaluation results from training."""

    tab_metrics, tab_ablation, tab_speed = st.tabs(
        ["📈 Metrics", "🔧 Ablations", "⚡ Performance"]
    )

    with tab_metrics:
        st.subheader("Recommendation Quality")

        # Metrics table
        if "metrics" in results:
            metrics_df = pd.DataFrame(results["metrics"])
            st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("Metrics data not available in results.")

    with tab_ablation:
        st.subheader("Ablation: Effect of SAE Sparsity")

        if "ablations" in results:
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
            st.plotly_chart(fig, use_container_width=True)

            # Table
            st.dataframe(ablations_df, use_container_width=True)
        else:
            st.info("Ablation data not available.")

    with tab_speed:
        st.subheader("Inference Latency")

        if "latency" in results:
            latency = results["latency"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Latency", f"{latency.get('mean', 0):.3f}s")
            with col2:
                st.metric("P95 Latency", f"{latency.get('p95', 0):.3f}s")
            with col3:
                st.metric("Max Latency", f"{latency.get('max', 0):.3f}s")
        else:
            st.info("Latency data not available.")


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
            "NDCG@100": [0.580, 0.550, 0.570, 0.580],
            "Model Size (MB)": [12.5, 1.2, 2.4, 4.8],
        }

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

        # Quality vs Model Size trade-off
        fig = px.scatter(
            metrics_df,
            x="Model Size (MB)",
            y="Recall@20",
            size="NDCG@100",
            hover_name="Model",
            title="Quality vs Model Size (bubble area = NDCG@100)",
            labels={
                "Model Size (MB)": "Model Size (MB)",
                "Recall@20": "Recall@20",
            },
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
        **Key Insights:**
        - SAE substantially reduces model size (12.5MB → 1.2MB for k=32)
        - Quality remains competitive (Recall@20: 0.42-0.45)
        - Sweet spot: k=64 balances sparsity and performance
        """
        )

    with tab_ablation:
        st.subheader("Ablation: SAE Sparsity Analysis")

        ablation_data = {
            "k": [16, 32, 64, 128, 256],
            "Recall@20": [0.410, 0.420, 0.445, 0.450, 0.448],
            "NDCG@100": [0.540, 0.550, 0.570, 0.580, 0.575],
            "Model Size (MB)": [0.6, 1.2, 2.4, 4.8, 9.6],
        }

        ablation_df = pd.DataFrame(ablation_data)
        st.dataframe(ablation_df, use_container_width=True)

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
                titlefont=dict(color="#1f77b4"),
            ),
            yaxis2=dict(
                title="Model Size (MB)",
                titlefont=dict(color="#ff7f0e"),
                anchor="x",
                overlaying="y",
                side="right",
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab_speed:
        st.subheader("Inference Latency Distribution")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Latency", "0.847 s")
        with col2:
            st.metric("P95 Latency", "1.234 s")
        with col3:
            st.metric("Max Latency", "1.899 s")

        # Latency histogram (synthetic)
        import numpy as np

        latencies = np.random.normal(0.847, 0.3, 1000)
        latencies = np.clip(latencies, 0.1, 3.0)

        fig = px.histogram(
            x=latencies,
            nbins=30,
            title="Inference Time Distribution (1000 samples)",
            labels={"x": "Latency (seconds)"},
            color_discrete_sequence=["#636EFA"],
        )

        fig.add_vline(
            0.847,
            line_dash="dash",
            line_color="green",
            annotation_text="Mean",
            annotation_position="top right",
        )
        fig.add_vline(
            1.234,
            line_dash="dash",
            line_color="orange",
            annotation_text="P95",
            annotation_position="top right",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            """
        **Performance Notes:**
        - Target: <2s latency for interactive steering ✅
        - Mean encoding: ~0.15s per user
        - Mean scoring: ~0.70s for 10K items
        - Total: ~0.85s per recommendation
        """
        )
