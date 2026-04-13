"""Service for loading and accessing neuron co-activation relationships."""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class CoactivationService:
    """Load and provide access to neuron co-activation data."""

    def __init__(self, coactivation_path: Optional[Path] = None):
        """
        Initialize co-activation service.

        Args:
            coactivation_path: Path to neuron_coactivation.json file
        """
        self.coactivation_data = {}
        
        if coactivation_path:
            self.load_coactivation_data(coactivation_path)

    def load_coactivation_data(self, coactivation_path: Path):
        """Load co-activation data from JSON file."""
        try:
            coactivation_path = Path(coactivation_path)
            if coactivation_path.exists():
                with open(coactivation_path, 'r', encoding='utf-8') as f:
                    self.coactivation_data = json.load(f)
                logger.info(f"Loaded co-activation data for {len(self.coactivation_data)} neurons")
            else:
                logger.warning(f"Co-activation file not found: {coactivation_path}")
        except Exception as e:
            logger.error(f"Error loading co-activation data: {e}")

    def get_highly_coactivated(self, neuron_id: int) -> List[Dict[str, Any]]:
        """
        Get neurons that are frequently co-activated with the given neuron.

        Args:
            neuron_id: Index of the neuron

        Returns:
            List of dicts with neuron_id, label, and correlation
        """
        try:
            neuron_data = self.coactivation_data.get(str(neuron_id), {})
            return neuron_data.get('highly_coactivated', [])
        except Exception as e:
            logger.error(f"Error getting highly co-activated neurons: {e}")
            return []

    def get_rarely_coactivated(self, neuron_id: int) -> List[Dict[str, Any]]:
        """
        Get neurons that are rarely co-activated with the given neuron.

        Args:
            neuron_id: Index of the neuron

        Returns:
            List of dicts with neuron_id, label, and correlation
        """
        try:
            neuron_data = self.coactivation_data.get(str(neuron_id), {})
            return neuron_data.get('rarely_coactivated', [])
        except Exception as e:
            logger.error(f"Error getting rarely co-activated neurons: {e}")
            return []
