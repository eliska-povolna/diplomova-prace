"""
Run Registry System - Tracks all training runs with metadata
Helps users manage multiple training experiments and their results
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RunRegistry:
    """Manages a registry of training runs with metadata"""

    def __init__(self, registry_file: Path = None):
        """
        Initialize run registry

        Args:
            registry_file: Path to registry JSON file (default: outputs/manifest.json)
        """
        if registry_file is None:
            registry_file = Path.cwd() / "outputs" / "manifest.json"

        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        self.registry: Dict = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load registry from disk or create new one"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load registry: {e}. Creating new one.")

        return {"version": "1.0", "created": datetime.now().isoformat(), "runs": {}}

    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
        logger.info(f"Registry saved to {self.registry_file}")

    def register_run(
        self,
        run_id: str,
        stage: str,
        config: Dict = None,
        status: str = "pending",
        metadata: Dict = None,
    ) -> None:
        """
        Register a new training run

        Args:
            run_id: Run identifier (YYYYMMDD_HHMMSS format)
            stage: Stage name ('preprocess', 'train', or 'label')
            config: Configuration parameters used
            status: Completion status ('pending', 'completed', 'failed')
            metadata: Additional metadata
        """
        if run_id not in self.registry["runs"]:
            self.registry["runs"][run_id] = {
                "created": datetime.now().isoformat(),
                "stages": {},
            }

        stage_info = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "config": config or {},
            "metadata": metadata or {},
        }

        self.registry["runs"][run_id]["stages"][stage] = stage_info
        self._save_registry()
        logger.info(f"Registered run {run_id} - stage {stage}: {status}")

    def update_run_status(
        self, run_id: str, stage: str, status: str, metadata: Dict = None
    ) -> None:
        """
        Update the status of a run stage

        Args:
            run_id: Run identifier
            stage: Stage name
            status: New status
            metadata: Additional metadata to store
        """
        if run_id not in self.registry["runs"]:
            logger.warning(f"Run {run_id} not found in registry")
            return

        if stage not in self.registry["runs"][run_id]["stages"]:
            logger.warning(f"Stage {stage} not found in run {run_id}")
            return

        self.registry["runs"][run_id]["stages"][stage]["status"] = status
        self.registry["runs"][run_id]["stages"][stage][
            "completed"
        ] = datetime.now().isoformat()

        if metadata:
            self.registry["runs"][run_id]["stages"][stage]["metadata"].update(metadata)

        self._save_registry()

    def get_runs(self) -> List[str]:
        """Get list of all registered run IDs"""
        return sorted(self.registry["runs"].keys(), reverse=True)

    def get_latest_run(self) -> Optional[str]:
        """Get the most recent run ID"""
        runs = self.get_runs()
        return runs[0] if runs else None

    def get_run_info(self, run_id: str) -> Optional[Dict]:
        """Get complete info for a run"""
        return self.registry["runs"].get(run_id)

    def get_runs_by_stage(self, stage: str) -> List[str]:
        """Get all runs that have completed a specific stage"""
        completed_runs = []
        for run_id, run_info in self.registry["runs"].items():
            if stage in run_info["stages"]:
                stage_info = run_info["stages"][stage]
                if stage_info["status"] == "completed":
                    completed_runs.append(run_id)
        return sorted(completed_runs, reverse=True)

    def get_full_pipeline_runs(self) -> List[str]:
        """Get runs that have completed all three stages (preprocess, train, label)"""
        required_stages = {"preprocess", "train", "label"}
        complete_runs = []

        for run_id, run_info in self.registry["runs"].items():
            stages_present = set(run_info["stages"].keys())
            completed_stages = {
                stage
                for stage in stages_present
                if run_info["stages"][stage]["status"] == "completed"
            }

            if required_stages.issubset(completed_stages):
                complete_runs.append(run_id)

        return sorted(complete_runs, reverse=True)

    def print_summary(self):
        """Print human-readable summary of registry"""
        print("\n" + "=" * 80)
        print("RUN REGISTRY SUMMARY")
        print("=" * 80)

        if not self.registry["runs"]:
            print("No runs registered yet.")
            return

        for run_id in self.get_runs():
            run_info = self.registry["runs"][run_id]
            print(f"\n📦 Run ID: {run_id}")
            print(f"   Created: {run_info['created']}")

            stages = run_info.get("stages", {})
            if not stages:
                print("   (No stages completed)")
                continue

            for stage, stage_info in stages.items():
                status = stage_info.get("status", "unknown")
                status_symbol = {"completed": "✓", "pending": "⏳", "failed": "✗"}.get(
                    status, "?"
                )
                print(f"   {status_symbol} {stage}: {status}")

                if stage_info.get("metadata"):
                    metadata = stage_info["metadata"]
                    for key, value in list(metadata.items())[:3]:  # Show first 3 items
                        if isinstance(value, (int, float)):
                            print(f"      - {key}: {value}")

        print("\n" + "=" * 80)


def create_run_id() -> str:
    """Generate a run ID in YYYYMMDD_HHMMSS format"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    # Example usage
    registry = RunRegistry()

    # Example: Register a preprocessing run
    run_id = create_run_id()
    registry.register_run(run_id, "preprocess", status="pending")
    registry.update_run_status(
        run_id,
        "preprocess",
        "completed",
        {"num_restaurants": 12345, "num_reviews": 45678},
    )

    # Example: Register training for same run
    registry.register_run(run_id, "train", status="pending")
    registry.update_run_status(
        run_id, "train", "completed", {"epochs": 10, "final_loss": 0.052}
    )

    registry.print_summary()
