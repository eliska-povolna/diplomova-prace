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

        # Metric explanations in expandable sections
        with st.expander("📖 Metric Explanations", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    """
                **Recall@20** 
                - % of user's liked items appearing in top-20 recommendations
                - Range: 0-100% (higher = better)
                - Good value: > 20%
                - Measures: Coverage of user preferences
                
                **Precision@20**
                - % of recommendations that are actually relevant to user
                - Range: 0-100% (higher = better)
                - Good value: > 25%
                - Measures: Accuracy of recommendations
                
                **Hit Rate@20**
                - % of users who got ≥1 relevant item in top-20
                - Range: 0-100% (higher = better)
                - Good value: > 60%
                - Measures: Utility for typical users
                """
                )

            with col2:
                st.markdown(
                    """
                **NDCG@20** (Normalized Discounted Cumulative Gain)
                - Quality of ranking (best items ranked first)
                - Range: 0-1 (higher = better)
                - Good value: > 0.25
                - Penalizes wrong ranking order
                
                **Coverage**
                - % of total items recommended across all users
                - Range: 0-100% (higher = better discovery)
                - Good value: > 20%
                - Measures: Item variety in recommendations
                
                **Model Size**
                - Compressed model size in MB
                - Trade-off: smaller = faster, but may reduce accuracy
                - Shows SAE (sparse) vs ELSA (dense) compression
                """
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
            st.info(
                """
            ℹ️ **Ablation data not available yet**
            
            To generate ablation studies, train SAE models with different k values:
            ```bash
            for k in 16 32 64 128 256; do
              python src/train.py --config configs/default.yaml --sae-k=$k
            done
            ```
            Results will appear in the outputs/ directory.
            """
            )

    with tab_speed:
        st.subheader("Inference Latency & Performance")

        if "latency" in results:
            latency = results["latency"]

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Latency", f"{latency.get('mean', 0):.1f}ms")
            with col2:
                st.metric("P95 Latency", f"{latency.get('p95', 0):.1f}ms")
            with col3:
                st.metric("Max Latency", f"{latency.get('max', 0):.1f}ms")

            st.metric("Inference Count", f"{latency.get('count', 0)} requests")

            st.markdown(
                f"""
            **Latency Interpretation:**
            - Mean: Average inference time (typical user experience)
            - P95: 95th percentile (worst 5% of requests)
            - Max: Longest single inference
            
            **Performance Goals:**
            - Mean < 500ms = Interactive (Live Demo acceptable)
            - P95 < 1000ms = Good for web apps
            - P95 < 2000ms = Acceptable for pilot
            """
            )
        else:
            st.info(
                """
            ℹ️ **Latency data not available yet**
            
            Latency is measured automatically when using the Live Demo:
            1. Go to **🎛️ Live Demo** page
            2. Select a user and generate recommendations
            3. Latency will be recorded and displayed here
            
            Each inference (encoding + steering + ranking) is measured in milliseconds.
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
        st.dataframe(metrics_df, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
