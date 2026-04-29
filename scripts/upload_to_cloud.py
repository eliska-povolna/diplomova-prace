#!/usr/bin/env python3
"""
Upload completed experiment runs to GCS.

This script ensures that the strict best-run contract artifacts are available
on cloud storage for Streamlit deployment. Run this after training completes
to synchronize artifacts to GCS.

Usage:
    python scripts/upload_to_cloud.py --run-dir outputs/20260423_123055
    python scripts/upload_to_cloud.py --latest  # Upload the latest completed run
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_cloud_storage_helper(bucket_name: str):
    """Initialize cloud storage helper."""
    try:
        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        return CloudStorageHelper(bucket_name=bucket_name)
    except ImportError as e:
        logger.error(f"Could not import CloudStorageHelper: {e}")
        return None


def get_required_artifact_files():
    """Return list of required artifact files for strict best-run contract."""
    return [
        "summary.json",
        "mappings/item2index.pkl",
        "precomputed/user_csr_matrices.pkl",
        "data/test_users_top50.json",
        "checkpoints/elsa_best.pt",
        "neuron_coactivation.json",
        "neuron_category_metadata.json",
        "neuron_interpretations/",  # Directory marker
    ]


def validate_run_artifacts(run_dir: Path) -> bool:
    """Check if run directory has all required artifacts."""
    run_dir = Path(run_dir)
    required_files = get_required_artifact_files()
    missing = []

    for artifact in required_files:
        if artifact.endswith("/"):
            # Directory marker
            artifact_path = run_dir / artifact.rstrip("/")
            if not artifact_path.is_dir():
                missing.append(artifact)
        else:
            artifact_path = run_dir / artifact
            if not artifact_path.exists():
                missing.append(artifact)

    if missing:
        logger.error(f"Missing required artifacts in {run_dir}:")
        for item in missing:
            logger.error(f"  - {item}")
        return False

    logger.info(f"✅ All required artifacts validated in {run_dir}")
    return True


def find_experiment_manifest_dir_for_run(run_dir: Path) -> Optional[Path]:
    """Find outputs/experiments/<experiment_id> directory that references run_dir."""
    run_dir = Path(run_dir)
    outputs_dir = run_dir.parent
    experiments_dir = outputs_dir / "experiments"
    if not experiments_dir.exists() or not experiments_dir.is_dir():
        return None

    run_name = run_dir.name
    for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        try:
            with open(manifest_path, encoding="utf-8") as f:
                payload = json.load(f)
            runs = payload.get("runs", []) if isinstance(payload, dict) else []
            for run in runs:
                run_dir_str = str((run or {}).get("run_dir") or "")
                if not run_dir_str:
                    continue
                candidate = Path(run_dir_str)
                if candidate.name == run_name:
                    return exp_dir
        except Exception:
            continue

    return None


def upload_run_to_gcs(
    run_dir: Path,
    bucket_name: str,
    remote_prefix: str = "experiments",
    dry_run: bool = False,
) -> bool:
    """
    Upload run artifacts to GCS.

    Args:
        run_dir: Local run directory (e.g., outputs/20260423_123055)
        bucket_name: GCS bucket name
        remote_prefix: Remote path prefix in GCS (default: experiments)
        dry_run: If True, just show what would be uploaded without uploading

    Returns:
        True if upload succeeded or dry_run=True with no errors
    """
    run_dir = Path(run_dir)
    run_name = run_dir.name

    if not run_dir.exists():
        logger.error(f"Run directory not found: {run_dir}")
        return False

    if not validate_run_artifacts(run_dir):
        logger.error(f"Artifact validation failed for {run_dir}")
        return False

    # Get cloud helper
    cloud_helper = get_cloud_storage_helper(bucket_name)
    if not cloud_helper:
        logger.error("Failed to initialize cloud storage helper")
        return False

    logger.info(
        f"Uploading {run_dir} to gs://{bucket_name}/{remote_prefix}/{run_name}/"
    )

    required_files = get_required_artifact_files()
    uploaded_count = 0
    failed_count = 0

    for artifact in required_files:
        local_path = run_dir / artifact.rstrip("/")
        remote_path = f"{remote_prefix}/{run_name}/{artifact.rstrip('/')}"

        if artifact.endswith("/"):
            # Directory - upload all contents recursively
            if not local_path.is_dir():
                logger.warning(f"Skipping non-existent directory: {local_path}")
                continue

            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(run_dir)
                    remote_file = f"{remote_prefix}/{run_name}/{rel_path}".replace(
                        "\\", "/"
                    )

                    try:
                        if dry_run:
                            logger.info(
                                f"  [DRY RUN] Would upload: {file_path} → {remote_file}"
                            )
                        else:
                            logger.info(f"  Uploading: {file_path}")
                            ok = cloud_helper.upload_file(str(file_path), remote_file)
                            if not ok:
                                raise RuntimeError("upload_file returned False")
                        uploaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to upload {file_path}: {e}")
                        failed_count += 1
        else:
            # Single file
            try:
                if dry_run:
                    logger.info(
                        f"  [DRY RUN] Would upload: {local_path} → {remote_path}"
                    )
                else:
                    logger.info(f"  Uploading: {local_path}")
                    ok = cloud_helper.upload_file(str(local_path), remote_path)
                    if not ok:
                        raise RuntimeError("upload_file returned False")
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Failed to upload {artifact}: {e}")
                failed_count += 1

    # Upload experiment manifest if present
    manifest_dir = find_experiment_manifest_dir_for_run(run_dir)
    manifest_path = manifest_dir / "manifest.json" if manifest_dir else None
    if manifest_path and manifest_path.exists() and manifest_dir:
        try:
            remote_manifest = f"experiments/{manifest_dir.name}/manifest.json"
            if dry_run:
                logger.info(
                    f"  [DRY RUN] Would upload: {manifest_path} → {remote_manifest}"
                )
            else:
                logger.info(f"  Uploading: {manifest_path}")
                ok = cloud_helper.upload_file(str(manifest_path), remote_manifest)
                if not ok:
                    raise RuntimeError("upload_file returned False")
            uploaded_count += 1
        except Exception as e:
            logger.warning(f"Could not upload manifest: {e}")

    logger.info(
        f"Upload complete: {uploaded_count} files uploaded, {failed_count} failed"
    )

    if dry_run:
        logger.info("(This was a dry run - no files were actually uploaded)")

    return failed_count == 0


def find_latest_completed_run() -> Path:
    """Find the latest run with all required artifacts."""
    outputs_dir = Path("outputs")
    experiments_dir = outputs_dir / "experiments"

    if not experiments_dir.exists():
        logger.error(f"Experiments directory not found: {experiments_dir}")
        return None

    # List experiment manifests
    for exp_dir in sorted(experiments_dir.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        manifest_path = exp_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        # Try to read manifest and find best run
        try:
            import json

            with open(manifest_path) as f:
                manifest = json.load(f)

            runs = manifest.get("runs", [])
            for run in runs:
                run_dir = Path(run.get("run_dir", ""))
                if run_dir.exists() and validate_run_artifacts(run_dir):
                    logger.info(f"Found latest completed run: {run_dir}")
                    return run_dir
        except Exception as e:
            logger.warning(f"Could not read manifest {manifest_path}: {e}")
            continue

    logger.error("No completed runs found")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Upload experiment runs to GCS for Streamlit deployment"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory to upload (e.g., outputs/20260423_123055)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Upload the latest completed run",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="diplomova-prace",
        help="GCS bucket name (default: diplomova-prace)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )

    args = parser.parse_args()

    if args.latest:
        run_dir = find_latest_completed_run()
        if not run_dir:
            logger.error("Could not find latest completed run")
            return 1
    elif args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        parser.print_help()
        return 1

    success = upload_run_to_gcs(
        run_dir,
        args.bucket,
        dry_run=args.dry_run,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
