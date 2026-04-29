#!/usr/bin/env python
"""Resume photo upload to GCS with checkpoint support and memory optimization."""

import logging
import sys
import json
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

CHECKPOINT_FILE = Path("gcs_upload_checkpoint.json")


def load_checkpoint() -> dict:
    """Load checkpoint data if it exists."""
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE) as f:
                data = json.load(f)
            logger.info(
                f"✅ Loaded checkpoint: Uploaded={data['uploaded']}, Skipped={data['skipped']}, Failed={data['failed']}"
            )
            return data
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return {"uploaded": 0, "skipped": 0, "failed": 0, "last_processed": 0}
    return {"uploaded": 0, "skipped": 0, "failed": 0, "last_processed": 0}


def detect_gcs_resume_point(bucket, photos_dir: Path) -> int:
    """
    Detect resume point by checking what's already in GCS.

    Returns the index of the first file that hasn't been uploaded yet.
    """
    try:
        logger.info("🔍 Scanning GCS to find resume point...")

        # Get list of local photos (sorted)
        photo_files = sorted(photos_dir.glob("*.jpg"))
        if not photo_files:
            logger.warning("No local photos found")
            return 0

        # List all files in GCS photos/ folder
        blobs = bucket.list_blobs(prefix="photos/")
        gcs_photos = {
            blob.name.replace("photos/", "") for blob in blobs if blob.name != "photos/"
        }

        logger.info(f"Found {len(gcs_photos)} files in GCS")

        # Find last uploaded file index
        for i, photo_path in enumerate(photo_files):
            if photo_path.name not in gcs_photos:
                # This file hasn't been uploaded yet
                logger.info(
                    f"✅ Last uploaded file: {photo_files[i-1].name if i > 0 else 'none'}"
                )
                logger.info(
                    f"📍 Resuming from file #{i+1}/{len(photo_files)}: {photo_path.name}"
                )
                return i

        # All files already uploaded
        logger.info(f"✅ All {len(photo_files)} files already in GCS!")
        return len(photo_files)

    except Exception as e:
        logger.warning(f"⚠️  Could not scan GCS for resume point: {e}")
        logger.info("Falling back to checkpoint file or starting from beginning")
        return 0


def save_checkpoint(stats: dict):
    """Save checkpoint data."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def get_bucket_from_config() -> Optional[str]:
    """Read bucket name from secrets/env (same as app does)."""
    try:
        from src.ui.services.secrets_helper import get_cloud_storage_bucket

        return get_cloud_storage_bucket()
    except Exception as e:
        logger.debug(f"Could not load secrets: {e}")
        return None


def upload_photos_to_gcs_resume(
    photos_dir: Path,
    photos_json: Optional[Path],
    bucket_name: str,
    project_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
    dry_run: bool = False,
    batch_size: int = 100,  # Process in smaller batches to reduce memory
) -> int:
    """
    Upload photos from local directory to GCS with resume support.

    Args:
        photos_dir: Local directory containing photos
        photos_json: Optional path to photos.json metadata file
        bucket_name: GCS bucket name
        project_id: Optional GCP project ID
        credentials_path: Optional path to service account JSON key
        dry_run: If True, only count files without uploading
        batch_size: Number of files to process before GC cleanup (reduces memory)

    Returns:
        Number of photos uploaded
    """
    try:
        from google.cloud import storage
        from google.oauth2 import service_account
        import gc
    except ImportError:
        logger.error(
            "google-cloud-storage not installed. Install with: pip install google-cloud-storage"
        )
        return 0

    if not photos_dir.exists():
        logger.error(f"Photos directory not found: {photos_dir}")
        return 0

    # Load checkpoint
    checkpoint = load_checkpoint()
    start_index = checkpoint["last_processed"]
    uploaded = checkpoint["uploaded"]
    skipped = checkpoint["skipped"]
    failed = checkpoint["failed"]

    # Initialize credentials
    credentials = None
    if credentials_path and Path(credentials_path).exists():
        logger.info(f"Loading GCS credentials from: {credentials_path}")
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )

    # Initialize client
    if credentials:
        client = storage.Client(credentials=credentials, project=project_id)
    else:
        logger.info("Using Application Default Credentials")
        client = storage.Client(project=project_id)

    bucket = client.bucket(bucket_name)

    # Verify bucket exists
    if not bucket.exists():
        logger.error(f"Bucket {bucket_name} does not exist")
        return 0

    logger.info(f"Uploading photos from {photos_dir} to gs://{bucket_name}/photos/")

    # Load checkpoint or detect from GCS
    checkpoint = load_checkpoint()
    start_index = checkpoint["last_processed"]

    # If no checkpoint, try to detect from GCS
    if start_index == 0 and not CHECKPOINT_FILE.exists():
        gcs_resume_point = detect_gcs_resume_point(bucket, photos_dir)
        if gcs_resume_point > 0:
            start_index = gcs_resume_point
            logger.info(f"✅ Auto-detected resume point from GCS: {gcs_resume_point}")

    uploaded = checkpoint["uploaded"]
    skipped = checkpoint["skipped"]
    failed = checkpoint["failed"]

    if start_index == 0:
        # First, upload photos.json if provided
        if photos_json and photos_json.exists():
            try:
                blob = bucket.blob("photos.json")
                blob.upload_from_filename(
                    str(photos_json), content_type="application/json"
                )
                logger.info(
                    f"✓ Uploaded photos.json metadata file ({photos_json.stat().st_size / 1024:.1f} KB)"
                )
            except Exception as e:
                logger.error(f"Failed to upload photos.json: {e}")
                return 0

    # Collect all photos
    photo_files = sorted(photos_dir.glob("*.jpg"))
    total = len(photo_files)

    if total == 0:
        logger.error(f"No .jpg files found in {photos_dir}")
        return 0

    logger.info(f"Found {total} photos total")
    if start_index > 0:
        logger.info(f"Resuming from photo {start_index + 1}/{total}")

    if dry_run:
        logger.info(f"DRY RUN: Would upload {total} photos")
        return total

    # Upload with progress tracking and memory management
    batch_count = 0
    for i, photo_path in enumerate(photo_files[start_index:], start_index + 1):
        try:
            photo_name = photo_path.name
            gcs_path = f"photos/{photo_name}"
            blob = bucket.blob(gcs_path)

            # Skip if already exists (resume safety)
            if blob.exists():
                skipped += 1
            else:
                blob.upload_from_filename(str(photo_path), content_type="image/jpeg")
                uploaded += 1

            # Progress report every 1000 files
            if i % 1000 == 0:
                pct = 100 * i / total
                logger.info(
                    f"Progress: {i}/{total} ({pct:.1f}%) - "
                    f"Uploaded: {uploaded}, Skipped: {skipped}, Failed: {failed}"
                )
                # Save checkpoint frequently
                checkpoint = {
                    "uploaded": uploaded,
                    "skipped": skipped,
                    "failed": failed,
                    "last_processed": i,
                }
                save_checkpoint(checkpoint)

            # Memory management - force garbage collection in batches
            batch_count += 1
            if batch_count >= batch_size:
                gc.collect()
                batch_count = 0

        except Exception as e:
            failed += 1
            if failed <= 10:  # Log first 10 errors
                logger.warning(f"Failed to upload {photo_path.name}: {e}")

    # Final checkpoint
    checkpoint = {
        "uploaded": uploaded,
        "skipped": skipped,
        "failed": failed,
        "last_processed": total,
    }
    save_checkpoint(checkpoint)

    logger.info(
        f"✓ Upload complete: {uploaded} new, {skipped} already existed, {failed} failed (out of {total})"
    )
    return uploaded


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload photos to Google Cloud Storage with resume support"
    )
    parser.add_argument(
        "--photos-dir",
        type=Path,
        default=Path("yelp_photos/photos"),
        help="Local photos directory (default: yelp_photos/photos)",
    )
    parser.add_argument(
        "--photos-json",
        type=Path,
        default=Path("yelp_photos/photos.json"),
        help="Optional photos.json metadata file (default: yelp_photos/photos.json)",
    )
    parser.add_argument(
        "--bucket",
        help="GCS bucket name (reads from CLOUD_STORAGE_BUCKET secret if not provided)",
    )
    parser.add_argument(
        "--project-id",
        help="GCP project ID (optional)",
    )
    parser.add_argument(
        "--credentials",
        help="Path to service account JSON key (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count files, don't upload",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Files to process before garbage collection (default: 100)",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete checkpoint and start over",
    )

    args = parser.parse_args()

    # Handle reset
    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        logger.info("✓ Checkpoint deleted, starting fresh")

    # Get bucket from config if not provided
    bucket_name = args.bucket or get_bucket_from_config()

    if not bucket_name:
        logger.error(
            "Bucket name not provided and CLOUD_STORAGE_BUCKET not configured.\n"
            "Either:\n"
            "  1. Set CLOUD_STORAGE_BUCKET env variable\n"
            "  2. Set CLOUD_STORAGE_BUCKET in .streamlit/secrets.toml\n"
            "  3. Pass --bucket argument"
        )
        return 1

    logger.info(f"Using bucket: {bucket_name}")

    uploaded = upload_photos_to_gcs_resume(
        photos_dir=args.photos_dir,
        photos_json=args.photos_json,
        bucket_name=bucket_name,
        project_id=args.project_id,
        credentials_path=args.credentials,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
    )

    return 0 if uploaded >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
