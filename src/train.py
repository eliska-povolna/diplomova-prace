"""Training entry point for SAE-CF POI recommender.

Usage
-----
    python src/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAE-CF POI recommender")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading config from %s", args.config)
    # TODO: load config, build data pipeline, train CF, train SAE, save checkpoints
    raise NotImplementedError("Training pipeline not yet implemented. See src/models/.")


if __name__ == "__main__":
    main()
