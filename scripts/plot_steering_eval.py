"""Generate steering evaluation charts from outputs/steering_eval.csv.

Example CSV row (dummy):
2026-04-29T12:00:00Z,run_123,user_42,12,weighted-category,neuron:17,0.500,0.1432,0.1821,0.2500,0.3333,0.0840,0.0412,0.0833,0.0100,0.0234,"{\"17\": 0.5}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate steering evaluation plots.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_STEERING_EVAL_CSV,
        help="Path to outputs/steering_eval.csv",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_STEERING_EVAL_OUTDIR,
        help="Directory for generated PNG charts",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=12,
        help="Filter rows by k before plotting (default: 12)",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=500,
        help="Keep only the latest N rows from the CSV before plotting",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = load_steering_eval_dataframe(args.input, max_rows=args.limit_rows)
    if df.empty:
        logger.warning("No steering evaluation data found in %s", args.input)
        return 1

    paths = generate_steering_eval_plots(df, args.outdir, k_filter=args.k)
    logger.info("Saved steering plots to %s", args.outdir)
    for name, path in paths.items():
        logger.info("%s -> %s", name, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
