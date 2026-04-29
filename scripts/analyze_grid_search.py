"""Analyze grid search results and generate comparison plots.

Usage:
    python scripts/analyze_grid_search.py              # Compare all runs
    python scripts/analyze_grid_search.py --plot       # Generate plots
    python scripts/analyze_grid_search.py --best 5     # Show top 5 runs
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_results(results_dir: str | Path = "results") -> pd.DataFrame:
    """Load grid search summary CSV.
    
    Returns: DataFrame with all runs and their metrics
    """
    summary_path = Path(results_dir) / "grid_search_summary.csv"
    
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        print("   Run grid search first: python scripts/run_grid_search.py")
        return None
    
    df = pd.read_csv(summary_path)
    return df


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print basic statistics about grid search results."""
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\nTotal runs: {len(df)}")
    print(f"Experiments: {df['run_id'].nunique()}")
    
    # Metrics available
    metric_cols = [
        col for col in df.columns
        if col not in ['run_id', 'run_dir', 'timestamp', 
                       'elsa_latent_dim', 'sae_k', 'sae_l1_coef']
    ]
    print(f"\nMetrics computed: {', '.join(metric_cols[:5])}...")
    
    # Parameter ranges
    print("\nParameter Ranges Tested:")
    if 'elsa_latent_dim' in df.columns:
        print(f"  ELSA latent_dim: {df['elsa_latent_dim'].min()} - {df['elsa_latent_dim'].max()}")
    if 'elsa_weight_decay' in df.columns:
        print(f"  ELSA weight_decay: {df['elsa_weight_decay'].unique()}")
    if 'sae_k' in df.columns:
        print(f"  SAE k: {sorted(df['sae_k'].unique())}")
    if 'sae_l1_coef' in df.columns:
        print(f"  SAE l1_coef: {sorted(df['sae_l1_coef'].unique())}")
    if 'sae_width_ratio' in df.columns:
        print(f"  SAE width_ratio: {sorted(df['sae_width_ratio'].unique())}")


def print_best_runs(df: pd.DataFrame, metric: str = "ndcg_20", top_n: int = 10) -> None:
    """Print top N runs by specified metric."""
    if metric not in df.columns:
        print(f"❌ Metric '{metric}' not found in results")
        print(f"   Available metrics: {[c for c in df.columns if c not in ['run_id', 'run_dir', 'timestamp']]}")
        return
    
    print(f"\n{'=' * 70}")
    print(f"TOP {top_n} RUNS BY {metric.upper()}")
    print("=" * 70)
    
    top = df.nlargest(top_n, metric)
    
    for rank, (idx, row) in enumerate(top.iterrows(), 1):
        print(f"\n{rank}. {row['run_id']}")
        print(f"   {metric}: {row[metric]:.6f}")
        
        # Show hyperparameters for this run
        hp_keys = ['elsa_latent_dim', 'elsa_weight_decay', 'sae_k', 'sae_l1_coef', 'sae_width_ratio']
        for hp in hp_keys:
            if hp in row and pd.notna(row[hp]):
                print(f"   {hp}: {row[hp]}")
        
        print(f"   Results dir: {row['run_dir']}")


def print_metric_statistics(df: pd.DataFrame) -> None:
    """Print statistics for key metrics."""
    print(f"\n{'=' * 70}")
    print("METRIC STATISTICS")
    print("=" * 70)
    
    metrics = [
        col for col in df.columns
        if col not in ['run_id', 'run_dir', 'timestamp',
                       'elsa_latent_dim', 'elsa_weight_decay',
                       'sae_k', 'sae_l1_coef', 'sae_width_ratio']
    ]
    
    for metric in metrics[:10]:  # Show first 10 metrics
        if df[metric].dtype in ['float64', 'int64']:
            print(f"\n{metric}:")
            print(f"  Mean: {df[metric].mean():.6f}")
            print(f"  Std:  {df[metric].std():.6f}")
            print(f"  Min:  {df[metric].min():.6f}")
            print(f"  Max:  {df[metric].max():.6f}")


def compare_parameter_sweeps(df: pd.DataFrame) -> None:
    """Compare performance across parameter values."""
    print(f"\n{'=' * 70}")
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 70)
    
    # Identify which parameters were swept
    swept_params = {}
    for param in ['elsa_latent_dim', 'elsa_weight_decay', 'sae_k', 'sae_l1_coef', 'sae_width_ratio']:
        if param in df.columns and len(df[param].unique()) > 1:
            swept_params[param] = df[param].unique()
    
    if not swept_params:
        print("No parameter sweeps found in results")
        return
    
    # For each swept parameter, show average performance
    for param, values in swept_params.items():
        print(f"\n{param}:")
        
        for val in sorted(values):
            subset = df[df[param] == val]
            
            # Find metrics to average
            metric_cols = [
                col for col in df.columns
                if col not in ['run_id', 'run_dir', 'timestamp',
                              'elsa_latent_dim', 'elsa_weight_decay',
                              'sae_k', 'sae_l1_coef', 'sae_width_ratio']
            ]
            
            if metric_cols:
                # Show first metric (e.g., NDCG)
                primary_metric = metric_cols[0]
                avg_val = subset[primary_metric].mean()
                print(f"  {param}={val:>8} → avg {primary_metric}={avg_val:.6f} ({len(subset)} runs)")


def export_comparison_table(df: pd.DataFrame, output_path: str = "results/comparison.csv") -> None:
    """Export simplified comparison table."""
    # Select key columns for comparison
    key_cols = ['run_id', 'timestamp']
    hp_cols = ['elsa_latent_dim', 'elsa_weight_decay', 'sae_k', 'sae_l1_coef', 'sae_width_ratio']
    
    # Find available metric columns
    metric_cols = [
        col for col in df.columns
        if col not in key_cols + hp_cols
    ]
    
    selected_cols = key_cols + [col for col in hp_cols if col in df.columns] + metric_cols
    selected_df = df[[col for col in selected_cols if col in df.columns]]
    
    selected_df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison table exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Results directory",
    )
    parser.add_argument(
        "--metric",
        default="ndcg_20",
        help="Metric to use for ranking (default: ndcg_20)",
    )
    parser.add_argument(
        "--best",
        type=int,
        default=5,
        help="Show top N runs (default: 5)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export comparison table to CSV",
    )

    args = parser.parse_args()

    # Load results
    df = load_results(args.results_dir)
    if df is None:
        return

    # Print analyses
    print_summary_stats(df)
    print_best_runs(df, metric=args.metric, top_n=args.best)
    print_metric_statistics(df)
    compare_parameter_sweeps(df)

    # Export if requested
    if args.export:
        export_comparison_table(df, Path(args.results_dir) / "comparison.csv")

    print(f"\n{'=' * 70}")
    print("Analysis complete!")
    print(f"Full results: {Path(args.results_dir) / 'grid_search_summary.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
