"""Utility to clean up incomplete training runs.

An incomplete training run is one that has only ELSA checkpoint but is missing
SAE checkpoint (typically due to training interruption before SAE phase).

Usage
-----
    # List incomplete runs without deleting
    python -m src.cleanup_incomplete_runs --dry-run

    # Delete all incomplete runs
    python -m src.cleanup_incomplete_runs --delete

    # Delete all incomplete runs except the most recent one
    python -m src.cleanup_incomplete_runs --delete --keep-latest 1
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def is_training_complete(training_dir: Path) -> bool:
    """Check if training run has both ELSA and SAE checkpoints.

    Parameters
    ----------
    training_dir : Path
        Training output directory

    Returns
    -------
    bool
        True if both ELSA and SAE checkpoints exist
    """
    checkpoints_dir = training_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return False

    # Check for ELSA checkpoint
    elsa_exists = (checkpoints_dir / "elsa_best.pt").exists()

    # Check for SAE checkpoint (with or without suffix)
    sae_exists = (checkpoints_dir / "sae_best.pt").exists()
    if not sae_exists:
        sae_candidates = list(checkpoints_dir.glob("sae_*_best.pt"))
        sae_exists = len(sae_candidates) > 0

    return elsa_exists and sae_exists


def find_incomplete_runs(outputs_base: Path = Path("outputs")) -> list[Path]:
    """Find all incomplete training runs.

    Parameters
    ----------
    outputs_base : Path
        Base outputs directory

    Returns
    -------
    list[Path]
        List of incomplete training run directories, sorted by name (newest first)
    """
    if not outputs_base.exists():
        logger.warning(f"Outputs directory not found: {outputs_base}")
        return []

    training_runs = [
        d
        for d in outputs_base.iterdir()
        if d.is_dir() and len(d.name) == 15 and not is_training_complete(d)
    ]

    return sorted(training_runs, key=lambda x: x.name, reverse=True)


def cleanup_incomplete_runs(
    outputs_base: Path = Path("outputs"),
    delete: bool = False,
    keep_latest: int = 0,
) -> dict:
    """Clean up incomplete training runs.

    Parameters
    ----------
    outputs_base : Path
        Base outputs directory
    delete : bool
        If True, delete incomplete runs. If False, just list them (dry-run)
    keep_latest : int
        Number of most recent incomplete runs to keep (useful to preserve recent failures)

    Returns
    -------
    dict
        Summary with keys: 'total_incomplete', 'deleted', 'kept', 'space_freed_mb'
    """
    incomplete_runs = find_incomplete_runs(outputs_base)

    if not incomplete_runs:
        logger.info("✓ No incomplete training runs found")
        return {"total_incomplete": 0, "deleted": 0, "kept": 0, "space_freed_mb": 0}

    logger.info(f"Found {len(incomplete_runs)} incomplete training runs")

    space_freed = 0
    deleted_count = 0
    kept_count = 0

    for idx, run_dir in enumerate(incomplete_runs):
        is_recent = idx < keep_latest
        action = (
            "KEEP (recent)" if is_recent else "DELETE" if delete else "would delete"
        )

        # Calculate size
        try:
            size_mb = sum(f.stat().st_size for f in run_dir.rglob("*")) / (1024 * 1024)
        except OSError:
            size_mb = 0

        logger.info(f"  [{action}] {run_dir.name} ({size_mb:.1f} MB)")

        if delete and not is_recent:
            try:
                logger.info(f"     Deleting {run_dir.name}...")
                shutil.rmtree(run_dir)
                deleted_count += 1
                space_freed += size_mb
            except Exception as e:
                logger.error(f"     Error deleting {run_dir.name}: {e}")
        elif is_recent:
            kept_count += 1

    summary = {
        "total_incomplete": len(incomplete_runs),
        "deleted": deleted_count,
        "kept": kept_count,
        "space_freed_mb": round(space_freed, 1),
    }

    if delete:
        logger.info(f"\n✓ Cleanup complete:")
        logger.info(f"  Deleted: {deleted_count} runs")
        logger.info(f"  Kept: {kept_count} runs")
        logger.info(f"  Space freed: {space_freed:.1f} MB")
    else:
        logger.info(
            f"\n(Dry-run mode) Would delete {deleted_count} runs and free {space_freed:.1f} MB"
        )
        logger.info("Run with --delete to actually remove the incomplete runs")

    return summary


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Clean up incomplete training runs (missing SAE checkpoint)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List incomplete runs (dry-run)
  python -m src.cleanup_incomplete_runs --dry-run
  
  # Delete all incomplete runs
  python -m src.cleanup_incomplete_runs --delete
  
  # Delete but keep the 2 most recent (for reference)
  python -m src.cleanup_incomplete_runs --delete --keep-latest 2
        """,
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Base outputs directory (default: outputs)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete incomplete runs (default: dry-run only)",
    )
    parser.add_argument(
        "--keep-latest",
        type=int,
        default=0,
        help="Number of most recent incomplete runs to keep (default: 0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List incomplete runs without deleting (default behavior)",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 80)
    logger.info("INCOMPLETE TRAINING RUN CLEANUP")
    logger.info("=" * 80)
    logger.info(f"Outputs directory: {args.outputs_dir}")
    logger.info("")

    cleanup_incomplete_runs(
        outputs_base=args.outputs_dir,
        delete=args.delete,
        keep_latest=args.keep_latest,
    )


if __name__ == "__main__":
    main()
