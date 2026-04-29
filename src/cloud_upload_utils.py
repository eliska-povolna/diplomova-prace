"""Utilities for uploading training artifacts to cloud storage after completion."""

import json
import logging
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


def get_cloud_helper():
    """Get CloudStorageHelper if GCS is configured."""
    import os

    bucket_name = os.getenv("GCS_BUCKET_NAME") or os.getenv("CLOUD_STORAGE_BUCKET")
    if not bucket_name:
        return None

    try:
        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        return CloudStorageHelper(bucket_name=bucket_name)
    except Exception as e:
        logger.warning(f"Could not initialize cloud storage: {e}")
        return None


def get_required_artifacts() -> List[str]:
    """Return list of artifact paths required by strict best-run contract."""
    return [
        "summary.json",
        "mappings/item2index.pkl",
        "precomputed/user_csr_matrices.pkl",
        "data/test_users_top50.json",
        "checkpoints/elsa_best.pt",
        "neuron_coactivation.json",
        "neuron_category_metadata.json",
        "neuron_interpretations/",
    ]


def validate_artifacts(run_dir: Path) -> bool:
    """Validate that all required artifacts exist."""
    run_dir = Path(run_dir)
    missing = []

    for artifact in get_required_artifacts():
        if artifact.endswith("/"):
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

    logger.info(f"✅ All required artifacts present in {run_dir}")
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
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
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


def upload_run_artifacts(
    run_dir: Path,
    experiment_manifest_dir: Optional[Path] = None,
    skip_if_exists: bool = False,
) -> bool:
    """
    Upload run artifacts to cloud storage after training completes.

    Args:
        run_dir: Path to completed run directory
        experiment_manifest_dir: Optional path to experiment manifest directory
        skip_if_exists: If True, skip upload if artifacts already exist in cloud

    Returns:
        True if upload successful or skipped, False on error
    """
    cloud_helper = get_cloud_helper()
    if not cloud_helper:
        logger.info("Cloud storage not configured, skipping upload")
        return True

    run_dir = Path(run_dir)
    run_name = run_dir.name

    if not validate_artifacts(run_dir):
        logger.error("Artifact validation failed, skipping cloud upload")
        return False

    remote_prefix = f"experiments/{run_name}"

    if experiment_manifest_dir is None:
        experiment_manifest_dir = find_experiment_manifest_dir_for_run(run_dir)

    # Check if already exists
    if skip_if_exists:
        try:
            summary_exists = cloud_helper.exists(f"{remote_prefix}/summary.json")

            manifest_exists = True
            if experiment_manifest_dir:
                manifest_path = experiment_manifest_dir / "manifest.json"
                if manifest_path.exists():
                    remote_manifest = (
                        f"experiments/{experiment_manifest_dir.name}/manifest.json"
                    )
                    manifest_exists = cloud_helper.exists(remote_manifest)

            if summary_exists and manifest_exists:
                logger.info(f"Run and manifest already exist in cloud: {remote_prefix}")
                return True
        except Exception:
            pass  # Doesn't exist yet, proceed with upload

    logger.info(
        f"Uploading run artifacts to gs://{cloud_helper.bucket_name}/{remote_prefix}/"
    )

    uploaded_count = 0
    failed_count = 0

    for artifact in get_required_artifacts():
        local_path = run_dir / artifact.rstrip("/")

        if artifact.endswith("/"):
            # Directory - upload all contents
            if not local_path.is_dir():
                logger.warning(f"Skipping non-existent directory: {local_path}")
                continue

            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(run_dir)
                    remote_file = f"{remote_prefix}/{rel_path}".replace("\\", "/")

                    try:
                        logger.debug(f"Uploading {file_path}")
                        ok = cloud_helper.upload_file(str(file_path), remote_file)
                        if not ok:
                            raise RuntimeError("upload_file returned False")
                        uploaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to upload {file_path}: {e}")
                        failed_count += 1
        else:
            # Single file
            remote_file = f"{remote_prefix}/{artifact}"
            try:
                logger.debug(f"Uploading {local_path}")
                ok = cloud_helper.upload_file(str(local_path), remote_file)
                if not ok:
                    raise RuntimeError("upload_file returned False")
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Failed to upload {artifact}: {e}")
                failed_count += 1

    # Upload experiment manifest if provided
    if experiment_manifest_dir:
        manifest_path = experiment_manifest_dir / "manifest.json"
        if manifest_path.exists():
            try:
                remote_manifest = (
                    f"experiments/{experiment_manifest_dir.name}/manifest.json"
                )
                logger.debug(f"Uploading manifest {manifest_path}")
                ok = cloud_helper.upload_file(str(manifest_path), remote_manifest)
                if not ok:
                    raise RuntimeError("upload_file returned False")
                uploaded_count += 1
            except Exception as e:
                logger.warning(f"Could not upload manifest: {e}")

    logger.info(
        f"Upload complete: {uploaded_count} files uploaded, {failed_count} failed"
    )

    if failed_count > 0:
        logger.error(
            f"Upload had {failed_count} failures - run may be incomplete in cloud"
        )
        return False

    logger.info(
        f"✅ Run successfully uploaded to cloud: gs://{cloud_helper.bucket_name}/{remote_prefix}/"
    )
    return True
