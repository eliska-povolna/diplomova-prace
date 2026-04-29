#!/usr/bin/env python
"""
Precompute CSR matrices for all users from interaction history.

This script:
1. Loads all users from the database
2. Builds a CSR matrix for each user (1 row, n_items columns)
3. Saves all matrices to a single pickle file locally
4. Uploads to Cloud Storage

This eliminates the need to build matrices on every app load.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def precompute_user_matrices(
    data_service,
    inference_service,
    output_dir: Path = Path("data"),
    upload_to_cloud: bool = True,
):
    """
    Precompute CSR matrices for all users.

    Args:
        data_service: DataService instance
        inference_service: InferenceService instance (for n_items)
        output_dir: Directory to save pickle file
        upload_to_cloud: Whether to upload to Cloud Storage after saving
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_path = output_dir / "user_csr_matrices.pkl"

    logger.info("🔍 Loading all users from database...")

    # Get all unique users from the database
    try:
        # Query all unique user IDs
        all_users = data_service.get_all_users()
        logger.info(f"✅ Found {len(all_users)} users in database")
    except Exception as e:
        logger.error(f"❌ Failed to get users: {e}")
        raise

    if not all_users:
        logger.warning("⚠️ No users found in database!")
        return

    # Ensure inference service has n_items
    if inference_service.n_items is None:
        logger.error("❌ Inference service n_items not initialized!")
        raise ValueError("n_items not set in inference service")

    n_items = inference_service.n_items
    logger.info(f"📐 Building matrices with shape (1, {n_items}) for each user...")

    user_matrices = {}
    failed_users = []
    matrices_built = 0

    for user_idx, user_id in enumerate(all_users, 1):
        try:
            # Get POI indices for this user
            poi_indices = data_service.get_user_interactions(user_id)

            if not poi_indices:
                logger.debug(
                    f"[{user_idx}/{len(all_users)}] User {user_id}: no interactions"
                )
                failed_users.append((user_id, "no_interactions"))
                continue

            # Validate POI indices are within bounds
            max_poi_idx = max(poi_indices)
            if max_poi_idx >= n_items:
                logger.debug(
                    f"[{user_idx}/{len(all_users)}] User {user_id}: POI index {max_poi_idx} >= {n_items}"
                )
                failed_users.append((user_id, "index_out_of_bounds"))
                continue

            # Build CSR matrix
            row = np.zeros(len(poi_indices), dtype=int)  # All row 0
            col = np.array(poi_indices, dtype=int)  # POI indices as columns
            data_vals = np.ones(len(poi_indices), dtype=np.float32)

            user_csr = csr_matrix((data_vals, (row, col)), shape=(1, n_items))

            user_matrices[user_id] = user_csr
            matrices_built += 1

            if user_idx % 100 == 0 or user_idx == len(all_users):
                logger.info(
                    f"[{user_idx}/{len(all_users)}] Built {matrices_built} matrices..."
                )

        except Exception as e:
            logger.debug(
                f"[{user_idx}/{len(all_users)}] Failed to build matrix for user {user_id}: {e}"
            )
            failed_users.append((user_id, str(e)))

    logger.info(f"✅ Successfully built {matrices_built} CSR matrices")

    if failed_users:
        logger.warning(f"⚠️ Failed to build matrices for {len(failed_users)} users:")
        for user_id, reason in failed_users[:10]:  # Show first 10
            logger.debug(f"  - {user_id}: {reason}")
        if len(failed_users) > 10:
            logger.debug(f"  ... and {len(failed_users) - 10} more")

    # Save locally
    logger.info(f"💾 Saving {matrices_built} matrices to {local_path}...")
    try:
        with open(local_path, "wb") as f:
            pickle.dump(user_matrices, f)
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Saved locally: {local_path} ({file_size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"❌ Failed to save locally: {e}")
        raise

    # Upload to cloud
    if upload_to_cloud:
        try:
            logger.info("☁️ Uploading to Cloud Storage...")
            cloud_path = "metadata/user_csr_matrices.pkl"
            data_service.upload_to_cloud(str(local_path), cloud_path)
            logger.info(
                f"✅ UPLOADED TO: gs://{data_service.config.get('GCS_BUCKET_NAME', 'bucket')}/{cloud_path}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Cloud upload failed (app will use local file): {e}")

    return user_matrices


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add src to path so we can import our modules
    repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root))

    from src.ui.services.data_service import DataService
    from src.ui.services.inference_service import InferenceService

    import streamlit as st

    # Try to load config from streamlit secrets
    try:
        config = {
            k: st.secrets[k]
            for k in ["GCS_BUCKET_NAME", "DUCKDB_PATH", "PARQUET_DIR"]
            if k in st.secrets
        }
    except:
        config = {}

    # Fallback to env / defaults
    if not config.get("GCS_BUCKET_NAME"):
        config["GCS_BUCKET_NAME"] = "diplomova-prace"

    duckdb_path = Path(config.get("DUCKDB_PATH", "yelp.duckdb"))
    parquet_dir = Path(config.get("PARQUET_DIR", "../Yelp-JSON/yelp_parquet"))

    # Ensure paths are absolute or relative to repo root
    if not duckdb_path.is_absolute():
        duckdb_path = repo_root / duckdb_path
    if not parquet_dir.is_absolute():
        parquet_dir = repo_root / parquet_dir

    logger.info(f"📁 DuckDB: {duckdb_path}")
    logger.info(f"📁 Parquet: {parquet_dir}")

    # Find latest model checkpoints from outputs directory FIRST
    # (so we can use the item2index from the same training run)
    outputs_dir = repo_root / "outputs"
    if outputs_dir.exists():
        # Get latest timestamped directory
        latest_run = sorted(outputs_dir.glob("202*"), key=lambda p: p.name)[-1]
        checkpoints_dir = latest_run / "checkpoints"
        
        # Use item2index from the same training run
        item2index_path = latest_run / "data" / "item2index.pkl"
        
        elsa_ckpt = checkpoints_dir / "elsa_best.pt"
        sae_ckpt = checkpoints_dir / "sae_r4_k32_best.pt"

        logger.info(f"📁 Using models from latest run: {latest_run.name}")
        logger.info(f"   ELSA: {elsa_ckpt}")
        logger.info(f"   SAE:  {sae_ckpt}")
        logger.info(f"📁 item2index: {item2index_path}")
    else:
        raise FileNotFoundError(f"No outputs directory found at {outputs_dir}")

    # Initialize services
    logger.info("🚀 Initializing services...")
    logger.info(f"📋 item2index_path exists: {item2index_path.exists()}")
    logger.info(f"📋 item2index_path: {item2index_path}")
    
    data_service = DataService(
        duckdb_path=duckdb_path,
        parquet_dir=parquet_dir,
        config=config,
        item2index_path=item2index_path,
    )

    inference_service = InferenceService(
        elsa_checkpoint_path=elsa_ckpt,
        sae_checkpoint_path=sae_ckpt,
        data_service=data_service,
    )

    # Precompute matrices
    logger.info("=" * 80)
    logger.info("PRECOMPUTING USER CSR MATRICES")
    logger.info("=" * 80)

    precompute_user_matrices(
        data_service,
        inference_service,
        output_dir=repo_root / "data",
        upload_to_cloud=True,
    )

    logger.info("=" * 80)
    logger.info("✅ PRECOMPUTATION COMPLETE")
    logger.info("=" * 80)
