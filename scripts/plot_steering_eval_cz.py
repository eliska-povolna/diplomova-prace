"""Generate Czech-language steering evaluation charts from outputs/steering_eval.csv.

Usage:
  python scripts/plot_steering_eval_cz.py --outdir img/generated_cz
  python scripts/plot_steering_eval_cz.py --input outputs/steering_eval.csv --outdir img/generated_cz --k 20
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.ui.services.steering_eval import (
    DEFAULT_STEERING_EVAL_CSV,
    DEFAULT_STEERING_EVAL_OUTDIR,
    generate_steering_eval_plots,
    load_steering_eval_dataframe,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Czech-language steering evaluation plots.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STEERING_EVAL_CSV,
        help="Path to outputs/steering_eval.csv (or Cloud SQL if configured)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_STEERING_EVAL_OUTDIR,
        help="Directory for generated PNG charts (default: img/generated)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Filter rows by k value before plotting (optional, plots all k if omitted)",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=500,
        help="Keep only the latest N rows from source before plotting (default: 500)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    
    logger.info("Loading steering evaluation data from %s", args.input)
    df = load_steering_eval_dataframe(args.input, max_rows=args.limit_rows)
    
    if df.empty:
        logger.warning("No steering evaluation data found in %s", args.input)
        return 1

    logger.info("Generating Czech-language plots (k=%s)", args.k if args.k else "all")
    paths = generate_steering_eval_plots(df, args.outdir, k_filter=args.k, lang="cs")
    
    logger.info("✅ Saved Czech steering plots to %s", args.outdir)
    for name, path in paths.items():
        logger.info("  %s -> %s", name, path)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
