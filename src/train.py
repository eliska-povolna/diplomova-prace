"""Training entry point for ELSA + TopK SAE POI recommender.

Pipeline:
  1. Build CSR matrix from Yelp Parquet data (via src/data/)
  2. Train ELSA model     → saves models/elsa_model_best.pt
  3. Encode all users     → L2-normalised latent vectors
  4. Train TopK SAE       → saves models/sae_model_r{R}_k{K}.pt

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
    parser = argparse.ArgumentParser(description="Train ELSA + TopK SAE POI recommender")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Loading config from %s", args.config)
    # TODO: load config, build CSR matrix, train ELSA, encode users, train TopK SAE
    # See src/yelp_initial_exploration/train_elsa.py and train_sae.py for reference.
    raise NotImplementedError("Training pipeline not yet implemented. See src/models/.")


if __name__ == "__main__":
    main()

