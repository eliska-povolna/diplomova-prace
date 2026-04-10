"""Model loader service for discovering and loading checkpoints."""

from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Discover and load ELSA+SAE model checkpoints from outputs directory."""

    @staticmethod
    def find_latest_checkpoint(outputs_dir: Path) -> Optional[Path]:
        """
        Find the most recent checkpoint directory by timestamp.

        Expected structure:
            outputs/
            ├── 20260326_092110/
            │   ├── checkpoints/
            │   │   ├── elsa_best.pt
            │   │   └── sae_best.pt
            │   └── ...
            └── 20260326_093131/
                ├── checkpoints/
                └── ...

        Returns:
            Path to checkpoint directory or None if not found.
        """
        outputs_dir = Path(outputs_dir)

        if not outputs_dir.exists():
            logger.warning(f"Outputs directory not found: {outputs_dir}")
            return None

        # Find directories with timestamp pattern (YYYYMMDD_HHMMSS)
        checkpoint_dirs = [
            d for d in outputs_dir.glob("*/checkpoints") if d.parent.name[0].isdigit()
        ]

        if not checkpoint_dirs:
            logger.warning("No checkpoint directories found")
            return None

        # Return most recent (by directory name = timestamp)
        latest = max(checkpoint_dirs, key=lambda x: x.parent.name)
        logger.info(f"Found latest checkpoint: {latest.parent.name}")
        return latest

    @staticmethod
    def load_models(
        elsa_ckpt_path: Path, device: str = "cpu"
    ):
        """
        Load ELSA model from checkpoint path.

        Args:
            elsa_ckpt_path: Path to elsa_best.pt
            device: 'cpu' or 'cuda'

        Returns:
            Loaded ELSA model
        """
        from src.models.sae_cf_model import ELSASAEModel

        logger.info(f"Loading model from {elsa_ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(elsa_ckpt_path, map_location=device)

        # Initialize model (config should be in checkpoint)
        config = checkpoint.get("config", {})
        model = ELSASAEModel(
            n_users=config.get("n_users", 1000),
            n_items=config.get("n_items", 10000),
            embedding_dim=config.get("embedding_dim", 128),
            sae_k=config.get("sae_k", 64),
            device=device,
        )

        # Load state
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        logger.info("✅ Model loaded successfully")
        return model

    @staticmethod
    def get_checkpoint_metadata(checkpoint_dir: Path) -> Dict:
        """
        Extract metadata from checkpoint directory.

        Returns:
            Dict with keys: timestamp, elsa_path, sae_path, config
        """
        checkpoint_dir = Path(checkpoint_dir)

        metadata = {
            "timestamp": checkpoint_dir.parent.name,
            "elsa_path": checkpoint_dir / "elsa_best.pt",
            "sae_path": checkpoint_dir / "sae_best.pt",
            "checkpoint_dir": checkpoint_dir,
        }

        # Try to load config
        config_path = checkpoint_dir.parent / "config.yaml"
        if config_path.exists():
            import yaml

            with open(config_path) as f:
                metadata["config"] = yaml.safe_load(f)

        return metadata
