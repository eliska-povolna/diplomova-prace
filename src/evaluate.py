"""Evaluation entry point for SAE-CF POI recommender.

Usage
-----
    python src/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best.ckpt
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SAE-CF POI recommender")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--split",
        default="test",
        choices=["val", "test"],
        help="Which split to evaluate on",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Evaluating checkpoint %s on %s split", args.checkpoint, args.split)
    # TODO: load model, run evaluation, print Recall@K, NDCG@K, HR@K
    raise NotImplementedError("Evaluation pipeline not yet implemented.")


if __name__ == "__main__":
    main()
