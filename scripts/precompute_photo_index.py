#!/usr/bin/env python3
"""
Precompute photo index from photos.json and save locally + to cloud.

This script:
1. Reads photos.json and builds business_id -> [PhotoMetadata] index
2. Saves locally as photo_index.pkl for fast local startup
3. Uploads to Cloud Storage for Streamlit Cloud deployments

Run once during setup, then the app will use the precomputed index.
"""

import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ui.services.data_service import PhotoMetadata
from src.ui.services.secrets_helper import get_cloud_storage_bucket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_photo_index_from_json(photos_json_path: Path) -> Dict[str, List[PhotoMetadata]]:
    """
    Build photo index from photos.json file.
    
    Expected format: JSONL with records:
    {
        "photo_id": "xxx",
        "business_id": "yyy",
        "label": "outside|inside|food|drink|menu|other",
        "caption": "..."
    }
    """
    if not photos_json_path.exists():
        logger.error(f"photos.json not found at {photos_json_path}")
        return {}
    
    photo_index: Dict[str, List[PhotoMetadata]] = {}
    loaded_count = 0
    
    logger.info(f"Building photo index from {photos_json_path}...")
    
    try:
        with open(photos_json_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    photo_record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                    continue
                
                business_id = str(photo_record.get("business_id", ""))
                photo_id = str(photo_record.get("photo_id", ""))
                
                if not business_id or not photo_id:
                    logger.debug(f"Line {line_num}: Missing business_id or photo_id")
                    continue
                
                label = str(photo_record.get("label", "other"))
                caption = str(photo_record.get("caption", ""))
                
                # Note: we don't know local vs cloud path yet, will be determined at runtime
                photo_meta = PhotoMetadata(
                    photo_id=photo_id,
                    business_id=business_id,
                    path=f"photos/{photo_id}.jpg",  # Relative path, will be resolved at runtime
                    label=label,
                    caption=caption,
                )
                
                photo_index.setdefault(business_id, []).append(photo_meta)
                loaded_count += 1
        
        logger.info(f"✅ Loaded {loaded_count} photo records")
    
    except Exception as e:
        logger.error(f"Failed to build photo index: {e}")
        return {}
    
    # Sort photos by label priority for each business
    for business_id in photo_index:
        photo_index[business_id].sort(
            key=lambda p: PhotoMetadata.label_priority(p.label), reverse=True
        )
    
    total_photos = sum(len(photos) for photos in photo_index.values())
    logger.info(f"✅ Built index with {total_photos} photos for {len(photo_index)} businesses")
    
    # Log label distribution
    label_dist = {}
    for photos in photo_index.values():
        for photo in photos:
            label_dist[photo.label] = label_dist.get(photo.label, 0) + 1
    logger.info(f"Label distribution: {label_dist}")
    
    return photo_index


def save_photo_index_locally(photo_index: Dict, output_path: Path) -> bool:
    """Save photo index as pickle file locally."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(photo_index, f)
        logger.info(f"✅ Saved photo index to {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"Failed to save photo index locally: {e}")
        return False


def upload_photo_index_to_cloud(photo_index: Dict, bucket_name: str) -> bool:
    """Upload photo index to Cloud Storage."""
    try:
        from src.ui.services.cloud_storage_helper import CloudStorageHelper
        
        helper = CloudStorageHelper(bucket_name)
        
        # Serialize to JSON for cloud storage (more portable than pickle)
        photo_index_json = {}
        for business_id, photos_list in photo_index.items():
            photo_index_json[business_id] = [
                {
                    "photo_id": p.photo_id,
                    "business_id": p.business_id,
                    "label": p.label,
                    "caption": p.caption,
                }
                for p in photos_list
            ]
        
        # Upload as JSON
        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(photo_index_json, f)
            temp_path = f.name
        
        try:
            success = helper.upload_json(Path(temp_path), "metadata/photo_index.json")
            if success:
                logger.info(f"✅ Uploaded photo index to gs://{bucket_name}/metadata/photo_index.json")
                return True
        finally:
            Path(temp_path).unlink()
        
        return False
    
    except Exception as e:
        logger.error(f"Failed to upload photo index to cloud: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    # Find local photos directory
    yelp_photos_paths = [
        project_root / "yelp_photos",
        project_root / "Yelp-Photos",
    ]
    
    local_photos_dir = None
    for path in yelp_photos_paths:
        if path.exists():
            local_photos_dir = path
            break
    
    if not local_photos_dir:
        logger.error(f"Could not find local photos directory. Checked: {yelp_photos_paths}")
        sys.exit(1)
    
    photos_json_path = local_photos_dir / "photos.json"
    if not photos_json_path.exists():
        logger.error(f"photos.json not found at {photos_json_path}")
        sys.exit(1)
    
    logger.info(f"📷 Precomputing photo index...")
    logger.info(f"   Source: {photos_json_path}")
    
    # Step 1: Build photo index from JSON
    photo_index = build_photo_index_from_json(photos_json_path)
    if not photo_index:
        logger.error("Failed to build photo index")
        sys.exit(1)
    
    # Step 2: Save locally
    local_index_path = project_root / "data" / "photo_index.pkl"
    if not save_photo_index_locally(photo_index, local_index_path):
        logger.warning("Failed to save locally, but continuing...")
    
    # Step 3: Upload to cloud
    bucket_name = get_cloud_storage_bucket()
    if bucket_name:
        logger.info(f"Uploading to Cloud Storage (bucket: {bucket_name})...")
        if upload_photo_index_to_cloud(photo_index, bucket_name):
            logger.info("✅ Photo index precomputation complete!")
        else:
            logger.warning("Saved locally but failed to upload to cloud")
    else:
        logger.warning("Cloud Storage not configured, saved locally only")
    
    logger.info("")
    logger.info("✅ Photo index precomputation complete!")
    logger.info(f"   Total: {sum(len(p) for p in photo_index.values())} photos for {len(photo_index)} businesses")
    logger.info(f"   Next app startup will load from precomputed index (fast!)")
