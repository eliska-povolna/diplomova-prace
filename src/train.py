"""Training entry point for ELSA + TopK SAE POI recommender.

Pipeline:
    1. Load Yelp review/business data from DuckDB
  2. Build CSR matrix from interactions
  3. Train ELSA model on CSR matrix
  4. Encode users with ELSA (frozen), get latent vectors
  5. Train TopK SAE on latent vectors
  6. Save models and metrics to output directory

Usage
-----
    python src/train.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset

from src.data.preprocessing import (
    apply_kcore_filtering,
    build_csr,
    load_dataset,
    save_dataset,
)
from src.data.yelp_loader import load_businesses, load_reviews
from src.models.collaborative_filtering import ELSA, NMSELoss
from src.models.sparse_autoencoder import TopKSAE
from src.run_registry import RunRegistry, create_run_id
from src.ui.services.secrets_helper import get_cloud_storage_bucket
from src.utils import CheckpointManager, Config, load_config, setup_logger
from src.utils.evaluation import evaluate_recommendations, print_evaluation_report

logger = logging.getLogger(__name__)


def upload_results_to_cloud(output_dir: Path, timestamp: str) -> bool:
    """
    Upload training results to GCS if configured.

    Args:
        output_dir: Local output directory with results
        timestamp: Training timestamp (YYYYMMDD_HHMMSS)

    Returns:
        True if uploaded or not configured, False if upload failed
    """
    try:
        gcs_bucket_name = get_cloud_storage_bucket()
        if not gcs_bucket_name:
            logger.info("GCS_BUCKET_NAME not set, skipping cloud upload")
            return True

        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
        logger.info(f"Uploading results to GCS bucket: {gcs_bucket_name}")

        gcs_prefix = f"models/{timestamp}"

        # Upload summary.json
        summary_path = output_dir / "summary.json"
        if summary_path.exists():
            cloud_storage.upload_json(
                summary_path,
                f"{gcs_prefix}/summary.json",
                metadata={"timestamp": timestamp, "type": "training_summary"},
            )

        # Upload training_results.json
        results_path = output_dir / "training_results.json"
        if results_path.exists():
            cloud_storage.upload_json(
                results_path,
                f"{gcs_prefix}/training_results.json",
                metadata={"timestamp": timestamp, "type": "training_results"},
            )

        # Upload ranking metrics report (text)
        report_path = output_dir / "ranking_metrics_report.txt"
        if report_path.exists():
            blob = cloud_storage.bucket.blob(f"{gcs_prefix}/ranking_metrics_report.txt")
            blob.upload_from_filename(str(report_path), content_type="text/plain")
            logger.info(
                f"✅ Uploaded ranking metrics report → gs://{gcs_bucket_name}/{gcs_prefix}/ranking_metrics_report.txt"
            )

        logger.info(
            f"✅ Training results uploaded to gs://{gcs_bucket_name}/{gcs_prefix}/"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to upload results to cloud: {e}")
        logger.warning(
            "Continuing with local-only results (app won't access them on cloud)"
        )
        return False


def precompute_user_csr_matrices(
    reviews_df,
    item_map_after_kcore,
    output_dir: Path,
    upload_to_cloud: bool = True,
    top_n_users: int = 50,
):
    """
    Precompute CSR matrices for top-N users (for Streamlit app).

    Builds a 1×n_items sparse matrix for each user representing their interaction history.
    Only computes for the top N users by interaction count (default: 50 for Streamlit demo).
    This enables fast lookup in the Streamlit app without querying the database each time.

    Args:
        reviews_df: DataFrame with user_id and business_id columns
        item_map_after_kcore: business_id -> model_index mapping (filtered)
        output_dir: Training output directory for saving results
        upload_to_cloud: Whether to upload to Cloud Storage
        top_n_users: Number of top users to precompute (default: 50)

    Returns:
        Dict of {user_id: csr_matrix}
    """
    logger.info("=" * 60)
    logger.info("PRECOMPUTING USER CSR MATRICES")
    logger.info("=" * 60)

    precomp_dir = output_dir / "precomputed"
    precomp_dir.mkdir(parents=True, exist_ok=True)

    local_path = precomp_dir / "user_csr_matrices.pkl"

    n_items = len(item_map_after_kcore)

    # Find top N users by interaction count
    user_interaction_counts = (
        reviews_df.groupby("user_id").size().sort_values(ascending=False)
    )
    top_users = user_interaction_counts.head(top_n_users).index.tolist()

    logger.info(
        f"Total users in dataset: {len(user_interaction_counts)}, using top {len(top_users)} by interaction count"
    )
    logger.info(
        f"Interaction distribution - min: {user_interaction_counts[top_users].min()}, max: {user_interaction_counts[top_users].max()}"
    )

    # Filter reviews to only top users
    reviews_filtered = reviews_df[reviews_df["user_id"].isin(top_users)]
    all_users = reviews_filtered["user_id"].unique()

    logger.info(f"Building CSR matrices for {len(all_users)} users, {n_items} items...")

    user_matrices = {}
    failed_users = []
    matrices_built = 0

    for user_idx, user_id in enumerate(all_users, 1):
        try:
            # Get all interactions for this user (from filtered reviews)
            user_reviews = reviews_filtered[reviews_filtered["user_id"] == user_id]
            business_ids = user_reviews["business_id"].values

            # Map business IDs to model indices (skip if not in filtered set)
            poi_indices = [
                item_map_after_kcore[bid]
                for bid in business_ids
                if bid in item_map_after_kcore
            ]

            if not poi_indices:
                failed_users.append((user_id, "no_valid_interactions"))
                continue

            # Build CSR matrix: 1 row (user), n_items columns
            row = np.zeros(len(poi_indices), dtype=int)
            col = np.array(poi_indices, dtype=int)
            data_vals = np.ones(len(poi_indices), dtype=np.float32)

            user_csr = csr_matrix((data_vals, (row, col)), shape=(1, n_items))
            user_matrices[user_id] = user_csr
            matrices_built += 1

            if user_idx % 500 == 0 or user_idx == len(all_users):
                logger.info(
                    f"[{user_idx}/{len(all_users)}] Built {matrices_built} matrices..."
                )

        except Exception as e:
            logger.debug(f"Failed to build matrix for user {user_id}: {e}")
            failed_users.append((user_id, str(e)))

    logger.info(f"✅ Successfully built {matrices_built} user CSR matrices")

    if failed_users:
        logger.warning(f"⚠️ Failed for {len(failed_users)} users")

    # Save locally
    logger.info(f"💾 Saving {matrices_built} matrices to {local_path}...")
    try:
        with open(local_path, "wb") as f:
            pickle.dump(user_matrices, f)
        file_size_mb = local_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Saved locally: {local_path} ({file_size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"❌ Failed to save locally: {e}")
        return {}

    # Upload to cloud
    if upload_to_cloud:
        try:
            gcs_bucket_name = get_cloud_storage_bucket()
            if not gcs_bucket_name:
                logger.info(
                    "GCS not configured, skipping cloud upload of user matrices"
                )
                return user_matrices

            from src.ui.services.cloud_storage_helper import CloudStorageHelper

            cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
            gcs_path = f"metadata/user_csr_matrices.pkl"
            blob = cloud_storage.bucket.blob(gcs_path)
            blob.upload_from_filename(str(local_path))
            logger.info(f"✅ Uploaded to gs://{gcs_bucket_name}/{gcs_path}")
        except Exception as e:
            logger.warning(f"⚠️ Cloud upload failed (app will use local file): {e}")

    return user_matrices


class SparseDataset(Dataset):
    """Dataset wrapper for sparse CSR matrices that converts rows on-the-fly."""

    def __init__(self, csr_matrix):
        self.data = csr_matrix

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data[idx].toarray().squeeze()
        return torch.tensor(row, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Examples
    --------
    # Full pipeline (default)
    python -m src.train --config configs/default.yaml

    # Only train ELSA (reuse existing preprocessing)
    python -m src.train --config configs/default.yaml --skip-sae

    # Only train SAE (reuse existing ELSA + preprocessing)
    python -m src.train --config configs/default.yaml --skip-elsa

    # Skip preprocessing, use cached data
    python -m src.train --config configs/default.yaml --use-cached-preprocessing

    # Experiments: try different SAE hyperparams without retraining ELSA
    python -m src.train --config configs/default.yaml --skip-elsa --skip-preprocessing

    """
    parser = argparse.ArgumentParser(
        description="Train ELSA + TopK SAE POI recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Component selection:\n"
        "  --skip-preprocessing  Skip data loading/filtering (use cached data from last run)\n"
        "  --skip-elsa           Skip ELSA training (reuse best checkpoint)\n"
        "  --skip-sae            Skip SAE training (reuse best checkpoint)\n"
        "\nFor grid search experiments, use --skip-elsa --skip-preprocessing to\n"
        "only train SAE with different hyperparameters while reusing ELSA outputs.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data loading/filtering. Requires cached data from previous run. "
        "Useful for quick SAE experiments without reloading large datasets.",
    )
    parser.add_argument(
        "--skip-elsa",
        action="store_true",
        help="Skip ELSA training. Loads best ELSA checkpoint from previous run. "
        "Useful for SAE hyperparameter tuning without retraining ELSA.",
    )
    parser.add_argument(
        "--skip-sae",
        action="store_true",
        help="Skip SAE training. Useful for testing data pipeline or ELSA only.",
    )
    parser.add_argument(
        "--elsa-checkpoint",
        default=None,
        help="Path to specific ELSA checkpoint to load (overrides 'latest' search). "
        "Format: path/to/outputs/YYYYMMDD_HHMMSS/checkpoints/elsa_best.pt",
    )
    parser.add_argument(
        "--sae-checkpoint",
        default=None,
        help="Path to specific SAE checkpoint to load (for SAE-only experiments).",
    )
    return parser.parse_args()


class MetricsCollector:
    """Collect and report metrics during training."""

    def __init__(self) -> None:
        self.data: dict[str, list] = {}

    def record(self, epoch: int, **kwargs: float) -> None:
        """Record metrics for an epoch."""
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)

    def to_dict(self) -> dict[str, list]:
        """Get metrics as dictionary."""
        return self.data

    def get_summary(self) -> str:
        """Get formatted summary of latest metrics."""
        if not self.data:
            return "No metrics recorded"

        lines = []
        for key, values in self.data.items():
            if values:
                latest = values[-1]
                lines.append(f"  {key}: {latest:.6f}")
        return "\n".join(lines)


def cosine_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cosine reconstruction loss (1 - cosine_similarity)."""
    recon_norm = torch.nn.functional.normalize(recon, dim=-1)
    target_norm = torch.nn.functional.normalize(target, dim=-1)
    return (
        1.0
        - torch.nn.functional.cosine_similarity(recon_norm, target_norm, dim=-1).mean()
    )


def compute_model_sizes(elsa_model: ELSA, sae_model: TopKSAE, output_dir: Path) -> dict:
    """Compute and save model file sizes in bytes and MB."""
    elsa_temp = output_dir / "elsa_temp_size.pt"
    sae_temp = output_dir / "sae_temp_size.pt"

    try:
        # Save model state dicts to temp files
        torch.save(elsa_model.state_dict(), elsa_temp)
        torch.save(sae_model.state_dict(), sae_temp)

        # Get file sizes
        elsa_size_bytes = elsa_temp.stat().st_size
        sae_size_bytes = sae_temp.stat().st_size

        sizes = {
            "elsa_bytes": int(elsa_size_bytes),
            "elsa_mb": float(elsa_size_bytes / 1e6),
            "sae_bytes": int(sae_size_bytes),
            "sae_mb": float(sae_size_bytes / 1e6),
            "total_bytes": int(elsa_size_bytes + sae_size_bytes),
            "total_mb": float((elsa_size_bytes + sae_size_bytes) / 1e6),
        }

        logger.info(
            f"Model sizes: ELSA={sizes['elsa_mb']:.2f}MB, "
            f"SAE={sizes['sae_mb']:.2f}MB, Total={sizes['total_mb']:.2f}MB"
        )

        return sizes
    finally:
        # Clean up temp files
        elsa_temp.unlink(missing_ok=True)
        sae_temp.unlink(missing_ok=True)


def compute_sparsity_stats(
    sae_model: TopKSAE, Z_data: torch.Tensor, device: str, batch_size: int = 256
) -> dict:
    """Compute sparsity statistics (active neurons, sparsity ratio)."""
    sae_model.eval()
    all_active_counts = []

    loader = DataLoader(TensorDataset(Z_data), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (z_batch,) in loader:
            z_batch = z_batch.to(device)
            _, h_sparse, _ = sae_model(z_batch)

            # Count active neurons per sample
            active_per_sample = (h_sparse != 0).sum(dim=1).float()
            all_active_counts.append(active_per_sample)

    all_active = torch.cat(all_active_counts)
    avg_active = all_active.mean().item()
    max_active = all_active.max().item()

    # Total neurons in SAE hidden layer (dictionary size)
    total_neurons = sae_model.hidden_dim

    return {
        "avg_active_neurons": float(avg_active),
        "max_active_neurons": float(max_active),
        "total_neurons": int(total_neurons),
        "sparsity_ratio": float(1.0 - (avg_active / total_neurons)),
    }


def train_elsa(
    config: Config,
    X_train,
    X_val,
    n_items: int,
    checkpoint_mgr: CheckpointManager,
) -> tuple[ELSA, float, dict]:
    """Train ELSA model.

    Parameters
    ----------
    config : Config
        Training configuration.
    X_train : torch.Tensor or Dataset
        Training interaction matrix (dense tensor or Dataset).
    X_val : torch.Tensor or Dataset
        Validation interaction matrix.
    n_items : int
        Number of items in the interaction matrix.
    checkpoint_mgr : CheckpointManager
        Checkpoint manager for saving.

    Returns
    -------
    tuple[ELSA, float, dict]
        Trained ELSA model, best validation loss, and training statistics
    """
    logger.info("=" * 60)
    logger.info("TRAINING ELSA")
    logger.info("=" * 60)

    elsa_cfg = config["elsa"]

    device = elsa_cfg["device"]
    model = ELSA(n_items, latent_dim=elsa_cfg["latent_dim"]).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=elsa_cfg["learning_rate"],
        betas=(0.9, 0.99),
        weight_decay=elsa_cfg["weight_decay"],
    )
    criterion = NMSELoss()

    train_loader = torch.utils.data.DataLoader(
        X_train, batch_size=elsa_cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        X_val, batch_size=elsa_cfg["batch_size"], shuffle=False
    )

    metrics = MetricsCollector()
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_reason = None
    epoch_started = 0

    logger.info(
        f"Config: latent_dim={elsa_cfg['latent_dim']}, "
        f"lr={elsa_cfg['learning_rate']}, epochs={elsa_cfg['num_epochs']}"
    )

    training_start = time.time()

    for epoch in range(elsa_cfg["num_epochs"]):
        # Training
        model.train()
        train_loss = 0.0

        for x_batch in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)

        train_loss /= len(X_train)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_batch in val_loader:
                x_batch = x_batch.to(device)
                recon = model(x_batch)
                loss = criterion(recon, x_batch)
                val_loss += loss.item() * x_batch.size(0)

        val_loss /= len(X_val)

        metrics.record(epoch, train_loss=train_loss, val_loss=val_loss)

        logger.info(
            f"Epoch {epoch+1:3d}/{elsa_cfg['num_epochs']} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
        )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            epoch_started = epoch
            # Save with dataset metadata
            metadata = {
                "n_items": model.A.shape[0],
                "latent_dim": model.latent_dim,
            }
            checkpoint_mgr.save(
                model,
                epoch=epoch,
                metrics=metrics.to_dict(),
                name="elsa_best",
                metadata=metadata,
            )
        else:
            patience_counter += 1
            if patience_counter >= elsa_cfg["patience"]:
                early_stop_reason = f"patience threshold ({elsa_cfg['patience']} epochs without improvement)"
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    training_time = time.time() - training_start

    # Load best model
    model.load_state_dict(
        torch.load(checkpoint_mgr.checkpoint_dir / "elsa_best.pt", map_location=device)[
            "model_state_dict"
        ]
    )
    checkpoint_mgr.save_metrics(metrics.to_dict(), split="elsa_train")

    logger.info(f"ELSA training complete. Best val_loss={best_val_loss:.6f}")

    return (
        model,
        best_val_loss,
        {
            "best_epoch": int(epoch_started),
            "final_epoch": int(epoch),
            "training_time_sec": float(training_time),
            "early_stop_reason": early_stop_reason or "completed all epochs",
        },
    )


def train_sae(
    config: Config,
    elsa_model: ELSA,
    Z_train: torch.Tensor,
    Z_val: torch.Tensor,
    checkpoint_mgr: CheckpointManager,
) -> tuple[TopKSAE, float, dict]:
    """Train TopK SAE model.

    Parameters
    ----------
    config : Config
        Training configuration.
    elsa_model : ELSA
        Trained ELSA model (will be frozen).
    Z_train : torch.Tensor
        Training latent vectors from ELSA.
    Z_val : torch.Tensor
        Validation latent vectors from ELSA.
    checkpoint_mgr : CheckpointManager
        Checkpoint manager for saving.

    Returns
    -------
    tuple[TopKSAE, float, dict]
        Trained SAE model, best validation loss, and training statistics.
    """
    logger.info("=" * 60)
    logger.info("TRAINING TOPK SAE")
    logger.info("=" * 60)

    sae_cfg = config["sae"]
    elsa_cfg = config["elsa"]

    device = sae_cfg["device"]
    hidden_dim = sae_cfg["width_ratio"] * elsa_cfg["latent_dim"]

    sae = TopKSAE(
        input_dim=elsa_cfg["latent_dim"],
        hidden_dim=hidden_dim,
        k=sae_cfg["k"],
        l1_coef=sae_cfg["l1_coef"],
    ).to(device)

    optimizer = optim.Adam(
        sae.parameters(),
        lr=sae_cfg["learning_rate"],
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    train_loader = torch.utils.data.DataLoader(
        Z_train, batch_size=sae_cfg["batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        Z_val, batch_size=sae_cfg["batch_size"], shuffle=False
    )

    metrics = MetricsCollector()
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_reason = None
    epoch_started = 0

    logger.info(
        f"Config: width_ratio={sae_cfg['width_ratio']}, "
        f"hidden_dim={hidden_dim}, k={sae_cfg['k']}, "
        f"l1_coef={sae_cfg['l1_coef']}, epochs={sae_cfg['num_epochs']}"
    )

    training_start = time.time()

    for epoch in range(sae_cfg["num_epochs"]):
        # Training
        sae.train()
        train_recon_loss = 0.0
        train_l1_loss = 0.0
        train_active_neurons = []

        for z_batch in train_loader:
            z_batch = z_batch.to(device)
            optimizer.zero_grad()

            recon, h_sparse, _ = sae(z_batch)

            rec_loss = cosine_recon_loss(recon, z_batch)
            l1_loss = h_sparse.abs().mean()
            loss = rec_loss + sae_cfg["l1_coef"] * l1_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

            train_recon_loss += rec_loss.item() * z_batch.size(0)
            train_l1_loss += l1_loss.item() * z_batch.size(0)

            # Track sparsity
            active_per_sample = (h_sparse != 0).sum(dim=1).float()
            train_active_neurons.append(active_per_sample)

        train_recon_loss /= len(Z_train)
        train_l1_loss /= len(Z_train)
        avg_train_active = torch.cat(train_active_neurons).mean().item()

        # Validation
        sae.eval()
        val_recon_loss = 0.0
        cosine_sims = []
        val_active_neurons = []

        with torch.no_grad():
            for z_batch in val_loader:
                z_batch = z_batch.to(device)

                recon, h_sparse, _ = sae(z_batch)

                rec_loss = cosine_recon_loss(recon, z_batch)
                val_recon_loss += rec_loss.item() * z_batch.size(0)

                # Cosine similarity
                recon_norm = torch.nn.functional.normalize(recon, dim=-1)
                z_norm = torch.nn.functional.normalize(z_batch, dim=-1)
                cos_sim = torch.nn.functional.cosine_similarity(
                    recon_norm, z_norm, dim=-1
                ).mean()
                cosine_sims.append(cos_sim.item())

                # Track sparsity
                active_per_sample = (h_sparse != 0).sum(dim=1).float()
                val_active_neurons.append(active_per_sample)

        val_recon_loss /= len(Z_val)
        avg_cosine_sim = np.mean(cosine_sims)
        avg_val_active = torch.cat(val_active_neurons).mean().item()

        metrics.record(
            epoch,
            train_recon=train_recon_loss,
            train_l1=train_l1_loss,
            train_active=avg_train_active,
            val_recon=val_recon_loss,
            val_active=avg_val_active,
            cosine_sim=avg_cosine_sim,
        )

        logger.info(
            f"Epoch {epoch+1:3d}/{sae_cfg['num_epochs']} | "
            f"train_recon={train_recon_loss:.6f} train_l1={train_l1_loss:.6f} | "
            f"val_recon={val_recon_loss:.6f} cosine_sim={avg_cosine_sim:.4f} | "
            f"active={avg_val_active:.1f}"
        )

        # Early stopping
        if (best_val_loss - val_recon_loss) > sae_cfg["min_delta"]:
            best_val_loss = val_recon_loss
            patience_counter = 0
            epoch_started = epoch
            # Save with hyperparameter metadata
            metadata = {
                "k": sae_cfg["k"],
                "width_ratio": sae_cfg["width_ratio"],
                "latent_dim": elsa_model.latent_dim,
            }
            checkpoint_mgr.save(
                sae,
                epoch=epoch,
                metrics=metrics.to_dict(),
                name=f"sae_r{sae_cfg['width_ratio']}_k{sae_cfg['k']}_best",
                metadata=metadata,
            )
        else:
            patience_counter += 1
            if patience_counter >= sae_cfg["patience"]:
                early_stop_reason = f"patience threshold ({sae_cfg['patience']} epochs without improvement)"
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    training_time = time.time() - training_start

    # Load best model
    model_path = (
        checkpoint_mgr.checkpoint_dir
        / f"sae_r{sae_cfg['width_ratio']}_k{sae_cfg['k']}_best.pt"
    )
    sae.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
    )
    checkpoint_mgr.save_metrics(metrics.to_dict(), split="sae_train")

    logger.info(f"SAE training complete. Best val_recon={best_val_loss:.6f}")

    return (
        sae,
        best_val_loss,
        {
            "best_epoch": int(epoch_started),
            "final_epoch": int(epoch),
            "training_time_sec": float(training_time),
            "early_stop_reason": early_stop_reason or "completed all epochs",
        },
    )


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory with timestamp
    output_cfg = config["output"]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_cfg["base_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize run registry
    registry = RunRegistry()
    registry.register_run(
        run_id,
        "train",
        config={"skip_preprocessing": args.skip_preprocessing},
        status="pending",
    )

    # Set up logging
    setup_logger(
        __name__,
        log_dir=output_dir,
        level=getattr(logging, output_cfg["log_level"]),
    )
    logger.info(f"Output directory: {output_dir}")

    # Checkpoint manager
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # Device
    device = config["elsa"]["device"]
    logger.info(f"Using device: {device}")

    try:
        # Load data
        logger.info("=" * 60)
        logger.info("LOADING DATA")
        logger.info("=" * 60)

        db_path = config["data"]["db_path"]
        cache_dir = output_dir / "_preprocessed_cache"

        db_path_obj = Path(db_path)
        if not db_path_obj.exists():
            raise FileNotFoundError(f"DuckDB database not found: {db_path_obj}")

        # ⭐ CHECK IF WE CAN SKIP PREPROCESSING AND LOAD FROM CACHE
        if args.skip_preprocessing and cache_dir.exists():
            logger.info("⏭️  --skip-preprocessing: Loading from cache...")
            try:
                cached_dataset = load_dataset(cache_dir)
                X_csr = cached_dataset.csr
                item_map_before_kcore = cached_dataset.item_map
                logger.info(
                    f"✓ Loaded cached CSR: {X_csr.shape[0]} users × {X_csr.shape[1]} items"
                )

                # Still need universal mappings for output
                with open(cache_dir / "_universal_mappings.pkl", "rb") as f:
                    universal_user_map, universal_business_map = pickle.load(f)

                logger.info("✓ Loaded universal mappings from cache")
                # Skip to k-core filtering and beyond
                preprocessing_skipped = True
            except Exception as e:
                logger.warning(
                    f"Failed to load from cache: {e}. Rebuilding from scratch."
                )
                preprocessing_skipped = False
        else:
            preprocessing_skipped = False

        # ⭐ ONLY RUN PREPROCESSING IF NOT SKIPPED
        if not preprocessing_skipped:
            # Load with all applicable filters from config
            # ⭐ FIRST: Create UNIVERSAL mappings from ALL data BEFORE any filtering
            # This ensures we have complete mappings for all possible items
            logger.info("Creating universal item/business mappings...")
            all_reviews = load_reviews(
                db_path=db_path,
                pos_threshold=config["data"]["pos_threshold"],
                year_min=config["data"].get("year_min"),
                year_max=config["data"].get("year_max"),
            )

            # Build universal mappings from all data
            all_users = all_reviews["user_id"].unique()
            all_businesses = all_reviews["business_id"].unique()

            universal_user_map = {uid: idx for idx, uid in enumerate(all_users)}
            universal_business_map = {
                bid: idx for idx, bid in enumerate(all_businesses)
            }

            logger.info("Universal mappings created:")
            logger.info(f"  Total unique users: {len(universal_user_map)}")
            logger.info(f"  Total unique businesses: {len(universal_business_map)}")

            # ⭐ NOW apply filtering on top of universal data
            reviews = all_reviews.copy()

            # Filter by state (if specified)
            state_filter = config["data"].get("state_filter")
            if state_filter:
                businesses = load_businesses(
                    db_path=db_path,
                    state_filter=state_filter,
                    min_review_count=config["data"].get("min_review_count", 5),
                )
                business_ids = set(businesses["business_id"].values)
                logger.info(
                    f"Filtering by state {state_filter}: {len(business_ids)} businesses"
                )
                reviews = reviews[reviews["business_id"].isin(business_ids)]

            logger.info(f"Loaded {len(reviews)} reviews (after state filtering)")

            # Build CSR matrix FROM FILTERED DATA
            logger.info("Building CSR matrix from filtered data...")
            dataset = build_csr(reviews)
            X_csr = dataset.csr
            item_map_before_kcore = (
                dataset.item_map
            )  # business_id -> index (before k-core)
            logger.info(
                f"Built CSR: {X_csr.shape[0]} users × {X_csr.shape[1]} items, "
                f"{X_csr.nnz} interactions"
            )

            # Cache raw CSR + universal mappings for potential reuse
            cache_dir.mkdir(parents=True, exist_ok=True)
            save_dataset(dataset, cache_dir)
            with open(cache_dir / "_universal_mappings.pkl", "wb") as f:
                pickle.dump((universal_user_map, universal_business_map), f)
            logger.info(f"✓ Cached preprocessed data to {cache_dir}")

        # Apply k-core filtering
        logger.info("Applying k-core filtering (k=5)...")
        X_csr = apply_kcore_filtering(X_csr, k=5)
        logger.info(
            f"After k-core: {X_csr.shape[0]} users × {X_csr.shape[1]} items, "
            f"{X_csr.nnz} interactions"
        )

        # ⭐ Build final item mapping AFTER k-core filtering
        # The CSR matrix columns are renumbered after k-core, so we need to track
        # which original items survived and create a new mapping for them.
        # Create reverse mapping: index -> business_id (before k-core)
        index_to_business_before = {
            idx: bid for bid, idx in item_map_before_kcore.items()
        }

        # Get all items that are actually used (have non-zero entries) in the k-core filtered matrix
        item_indices_used = set(
            X_csr.nonzero()[1]
        )  # Column indices of non-zero entries

        # Build new mapping: business_id -> new_index (only items in k-core result)
        items_after_kcore = sorted(item_indices_used)
        item_map_after_kcore = {
            index_to_business_before[old_idx]: new_idx
            for new_idx, old_idx in enumerate(items_after_kcore)
        }

        logger.info(
            f"Item mapping: {len(item_map_before_kcore)} items → {len(item_map_after_kcore)} items (after k-core)"
        )

        # Save universal mappings for downstream use (e.g., labeling notebook)
        mappings_dir = output_dir / "mappings"
        mappings_dir.mkdir(parents=True, exist_ok=True)

        with open(mappings_dir / "user2index_universal.pkl", "wb") as f:
            pickle.dump(universal_user_map, f)
        with open(mappings_dir / "business2index_universal.pkl", "wb") as f:
            pickle.dump(universal_business_map, f)

        logger.info(f"Universal mappings saved to {mappings_dir}")

        # ⭐ IMPORTANT: Save FILTERED item mapping (after k-core)
        # This is what the model actually uses! Model indices are 0 to (n_items-1) from the k-core filtered matrix
        with open(mappings_dir / "item2index.pkl", "wb") as f:
            pickle.dump(item_map_after_kcore, f)

        logger.info(
            f"✓ Saved item2index_filtered (model space): {len(item_map_after_kcore)} items"
        )

        # Train/test split
        n_users = X_csr.shape[0]
        user_indices = np.arange(n_users)
        train_test_ratio = config["data"]["train_test_split"]
        val_ratio = config["data"]["val_split"]

        train_users, test_users = train_test_split(
            user_indices,
            test_size=1 - train_test_ratio,
            random_state=config["data"]["seed"],
        )

        n_test = len(test_users)
        n_train = len(train_users)
        logger.info(
            f"Train/test split: {n_train} users ({train_test_ratio*100:.0f}%) → train, "
            f"{n_test} users ({(1-train_test_ratio)*100:.0f}%) → test"
        )

        X_train_csr = X_csr[train_users]
        X_test_csr = X_csr[test_users]

        # Create datasets that handle sparse matrices efficiently
        X_train_dataset = SparseDataset(X_train_csr)
        X_test_dataset = SparseDataset(X_test_csr)

        # Train/val split on indices (not data)
        train_indices = np.arange(X_train_csr.shape[0])
        train_idx, val_idx = train_test_split(
            train_indices,
            test_size=val_ratio,
            random_state=config["data"]["seed"],
        )

        n_val = len(val_idx)
        n_train_split = len(train_idx)
        logger.info(
            f"Train/val split (from training data): {n_train_split} users ({(1-val_ratio)*100:.0f}%) → train, "
            f"{n_val} users ({val_ratio*100:.0f}%) → validation"
        )

        # Create subset datasets
        from torch.utils.data import Subset

        X_train_split = Subset(X_train_dataset, train_idx)
        X_val_split = Subset(X_train_dataset, val_idx)

        # Train ELSA
        elsa_model, elsa_best_loss, elsa_stats = train_elsa(
            config, X_train_split, X_val_split, X_train_csr.shape[1], checkpoint_mgr
        )

        # Encode all users with ELSA (frozen) using chunked encoding for large matrices
        logger.info("Encoding users with ELSA...")
        elsa_model.eval()
        with torch.no_grad():
            # Use chunked encoding for sparse matrices to avoid memory overflow
            Z_train = elsa_model.encode_csr_chunked(
                X_train_csr, chunk_size=4096, device=device
            )
            Z_val = elsa_model.encode_csr_chunked(
                X_train_csr[val_idx], chunk_size=4096, device=device
            )
            Z_test = elsa_model.encode_csr_chunked(
                X_test_csr, chunk_size=4096, device=device
            )

        logger.info(
            f"Encoded: z_train={Z_train.shape}, z_val={Z_val.shape}, z_test={Z_test.shape}"
        )

        # Train SAE
        sae_model, sae_best_loss, sae_stats = train_sae(
            config, elsa_model, Z_train, Z_val, checkpoint_mgr
        )

        # Final evaluation on test set
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 60)

        sae_model.eval()
        elsa_model.eval()
        with torch.no_grad():
            z_test_recon = sae_model.enc(Z_test)
            h_test = sae_model.encode(Z_test)
            z_recon = sae_model.decode(h_test)

            test_recon_loss = cosine_recon_loss(z_recon, Z_test).item()
            z_recon_norm = torch.nn.functional.normalize(z_recon, dim=-1)
            z_test_norm = torch.nn.functional.normalize(Z_test, dim=-1)
            test_cosine_sim = (
                torch.nn.functional.cosine_similarity(z_recon_norm, z_test_norm, dim=-1)
                .mean()
                .item()
            )

            # Sparsity analysis
            active_neurons = (h_test != 0).sum(dim=1).float().mean().item()

        # Compute ranking metrics (NDCG, Recall, MRR, etc.)
        logger.info("Computing ranking metrics...")
        evaluation_start = time.time()

        # ⭐ DIAGNOSTIC LOGGING (Phase 1)
        logger.info("\n📊 EVALUATION DIAGNOSTICS:")
        logger.info(
            f"Test set size: {X_test_csr.shape[0]} users × {X_test_csr.shape[1]} items"
        )
        logger.info(
            f"Test set sparsity: {(1.0 - X_test_csr.nnz / (X_test_csr.shape[0] * X_test_csr.shape[1])) * 100:.2f}%"
        )
        logger.info(f"Train users indices (first 5): {train_users[:5]}")
        logger.info(f"Test users indices (first 5): {test_users[:5]}")
        overlap_check = len(set(train_users) & set(test_users))
        logger.info(f"Train/test overlap: {overlap_check} users (should be 0)")
        if overlap_check > 0:
            logger.warning(
                f"⚠️  POTENTIAL DATA LEAKAGE: {overlap_check} users in both train and test!"
            )

        with torch.no_grad():
            # === ELSA ALONE EVALUATION ===
            logger.info("\nEvaluating ELSA alone on test set...")
            X_test_elsa_pred = elsa_model.decode(Z_test)  # (n_test_users, n_items)
            X_test_elsa_pred_np = X_test_elsa_pred.cpu().numpy()
            logger.info(
                f"ELSA predictions - min: {X_test_elsa_pred_np.min():.6f}, max: {X_test_elsa_pred_np.max():.6f}, mean: {X_test_elsa_pred_np.mean():.6f}"
            )

            # === SAE+ELSA EVALUATION ===
            logger.info("\nEvaluating SAE+ELSA on test set...")
            # Z_test is already encoded by ELSA. Now encode through SAE.
            h_test_sparse = sae_model.encode(Z_test)  # SAE encoder with sparsity
            z_test_sae_decoded = sae_model.decode(h_test_sparse)  # SAE decoder
            # z_test_sae_decoded is in latent space. Decode with ELSA to get items.
            X_test_sae_pred = elsa_model.decode(z_test_sae_decoded)
            X_test_sae_pred_np = X_test_sae_pred.cpu().numpy()
            logger.info(
                f"SAE+ELSA predictions - min: {X_test_sae_pred_np.min():.6f}, max: {X_test_sae_pred_np.max():.6f}, mean: {X_test_sae_pred_np.mean():.6f}"
            )

            # Compare ELSA vs SAE predictions
            diff = np.abs(X_test_elsa_pred_np - X_test_sae_pred_np)
            logger.info(
                f"Prediction difference - min: {diff.min():.6f}, max: {diff.max():.6f}, mean: {diff.mean():.6f}"
            )
            pct_changed = (
                (diff / (np.abs(X_test_elsa_pred_np) + 1e-6) > 0.5).sum()
                / diff.size
                * 100
            )
            logger.info(f"Percentage of predictions changed >50%: {pct_changed:.1f}%")

        # Evaluate both models
        logger.info("Computing metrics for ELSA alone...")
        ranking_metrics_elsa = evaluate_recommendations(
            X_test_csr, X_test_elsa_pred_np, ks=[5, 10, 20]
        )

        logger.info("Computing metrics for SAE+ELSA...")
        ranking_metrics_sae = evaluate_recommendations(
            X_test_csr, X_test_sae_pred_np, ks=[5, 10, 20]
        )

        # Generate recommendations for diversity metrics (using SAE+ELSA predictions)
        from src.utils.evaluation import (
            generate_recommendations,
            compute_coverage,
            compute_entropy,
            benchmark_inference,
            compare_model_performance,
        )

        all_recommendations = generate_recommendations(
            X_test_csr, X_test_sae_pred_np, k=20
        )

        # Compute diversity and coverage metrics
        coverage = compute_coverage(all_recommendations, n_items=X_test_csr.shape[1])
        entropy = compute_entropy(all_recommendations, n_items=X_test_csr.shape[1])

        # Benchmark inference latency
        latency_metrics = benchmark_inference(
            X_test_sae_pred_np, n_samples=min(100, X_test_csr.shape[0])
        )

        # Add these to SAE+ELSA ranking metrics
        ranking_metrics_sae["coverage"] = coverage
        ranking_metrics_sae["entropy"] = entropy
        ranking_metrics_sae["latency"] = latency_metrics

        # ⭐ Print comparison table
        comparison_report = compare_model_performance(
            ranking_metrics_elsa, ranking_metrics_sae
        )
        logger.info("\n" + comparison_report)

        # Print detailed reports
        logger.info("\n" + print_evaluation_report(ranking_metrics_elsa))
        logger.info("\n" + print_evaluation_report(ranking_metrics_sae))

        # Use SAE+ELSA metrics as primary ranking_metrics for saving
        ranking_metrics = ranking_metrics_sae

        evaluation_time = time.time() - evaluation_start
        logger.info(f"Ranking evaluation completed in {evaluation_time:.2f}s")

        logger.info(f"Test reconstruction loss: {test_recon_loss:.6f}")
        logger.info(f"Test cosine similarity: {test_cosine_sim:.4f}")
        logger.info(
            f"Average active neurons: {active_neurons:.1f}/{config['sae']['width_ratio'] * config['elsa']['latent_dim']}"
        )

        # Compute model sizes
        model_sizes = compute_model_sizes(elsa_model, sae_model, output_dir)

        # Compute sparsity stats on test set
        test_sparsity = compute_sparsity_stats(
            sae_model, Z_test, device, batch_size=256
        )

        # Save comprehensive summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "data": {
                "n_users": int(n_users),
                "n_items": int(X_csr.shape[1]),
                "n_items_before_kcore": int(len(item_map_before_kcore)),
                "n_interactions": int(X_csr.nnz),
                "n_train_users": int(len(train_users)),
                "n_test_users": int(len(test_users)),
                "sparsity_percent": float(
                    100.0 * (1.0 - X_csr.nnz / (X_csr.shape[0] * X_csr.shape[1]))
                ),
            },
            "elsa": {
                "best_val_loss": float(elsa_best_loss),
                "best_epoch": elsa_stats["best_epoch"],
                "final_epoch": elsa_stats["final_epoch"],
                "training_time_sec": elsa_stats["training_time_sec"],
                "early_stop_reason": elsa_stats["early_stop_reason"],
            },
            "sae": {
                "best_val_loss": float(sae_best_loss),
                "best_epoch": sae_stats["best_epoch"],
                "final_epoch": sae_stats["final_epoch"],
                "training_time_sec": sae_stats["training_time_sec"],
                "early_stop_reason": sae_stats["early_stop_reason"],
                "test_recon_loss": float(test_recon_loss),
                "test_cosine_sim": float(test_cosine_sim),
                "test_avg_active_neurons": float(active_neurons),
                "test_avg_active_neurons_detailed": test_sparsity["avg_active_neurons"],
                "test_max_active_neurons": test_sparsity["max_active_neurons"],
                "test_total_neurons": test_sparsity["total_neurons"],
                "test_sparsity_ratio": test_sparsity["sparsity_ratio"],
            },
            "ranking_metrics_elsa": ranking_metrics_elsa,  # ⭐ ELSA alone
            "ranking_metrics_sae": ranking_metrics_sae,  # ⭐ SAE+ELSA
            "ranking_metrics": ranking_metrics,  # Primary (same as ranking_metrics_sae)
            "model_sizes": model_sizes,
            "training": {
                "total_time_sec": elsa_stats["training_time_sec"]
                + sae_stats["training_time_sec"],
                "elsa_time_sec": elsa_stats["training_time_sec"],
                "sae_time_sec": sae_stats["training_time_sec"],
                "evaluation_time_sec": evaluation_time,
            },
            "output_dir": str(output_dir),
        }

        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        # � SAVE DATA FOR NEURON LABELING REPRODUCIBILITY
        logger.info("\nSaving data files for neuron labeling...")
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)

        try:
            # Save filtered reviews DataFrame
            with open(data_dir / "reviews_df.pkl", "wb") as f:
                pickle.dump(reviews, f)
            logger.info(
                f"✅ Saved reviews_df ({len(reviews)} rows) to {data_dir / 'reviews_df.pkl'}"
            )

            # Save item mapping (k-core filtered)
            with open(data_dir / "item_map_after_kcore.pkl", "wb") as f:
                pickle.dump(item_map_after_kcore, f)
            logger.info(
                f"✅ Saved item_map_after_kcore ({len(item_map_after_kcore)} items) to {data_dir / 'item_map_after_kcore.pkl'}"
            )

            # Save business metadata (if available)
            if "business_metadata" in locals():
                with open(data_dir / "business_metadata.pkl", "wb") as f:
                    pickle.dump(business_metadata, f)
                logger.info(
                    f"✅ Saved business_metadata to {data_dir / 'business_metadata.pkl'}"
                )
        except Exception as e:
            logger.error(f"Failed to save data for neuron labeling: {e}")
            logger.warning("Continuing without saved data (neuron labeling may fail)")

        # 🔄 PRECOMPUTE USER CSR MATRICES (post-training step)
        # This enables fast user history lookup in the app without querying DB each time
        logger.info("\nPRECOMPUTATION PHASE:")
        try:
            precompute_user_csr_matrices(
                reviews,
                item_map_after_kcore,
                output_dir,
                upload_to_cloud=True,
                top_n_users=50,
            )
        except Exception as e:
            logger.error(f"User matrix precomputation failed: {e}")
            logger.warning(
                "Continuing without precomputed matrices (app will work but slower)"
            )

        # Upload results to cloud if configured
        timestamp = output_dir.name  # YYYYMMDD_HHMMSS format
        upload_results_to_cloud(output_dir, timestamp)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Total training time: {summary['training']['total_time_sec']:.1f}s"
        )
        logger.info(f"  ELSA: {summary['training']['elsa_time_sec']:.1f}s")
        logger.info(f"  SAE: {summary['training']['sae_time_sec']:.1f}s")
        logger.info(f"  Evaluation: {summary['training']['evaluation_time_sec']:.1f}s")
        logger.info(
            f"Model sizes: Total {model_sizes['total_mb']:.2f}MB "
            f"(ELSA {model_sizes['elsa_mb']:.2f}MB, SAE {model_sizes['sae_mb']:.2f}MB)"
        )
        logger.info(f"\nRanking Metrics Summary:")
        for metric_name in ["ndcg", "recall", "precision", "mrr"]:
            if metric_name in ranking_metrics:
                values = ranking_metrics[metric_name]
                logger.info(
                    f"  {metric_name.upper()}: "
                    + ", ".join(f"{k}={v:.4f}" for k, v in sorted(values.items()))
                )
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Summary saved to: {summary_path}")

        # Register run as completed
        final_metrics = {
            "elsa_epochs": config["elsa"]["num_epochs"],
            "sae_epochs": config["sae"]["num_epochs"],
            "final_output_dir": str(output_dir),
        }
        registry.update_run_status(run_id, "train", "completed", final_metrics)
        logger.info(f"✓ Run {run_id} registered as completed")

    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        # Register run as failed
        registry.update_run_status(run_id, "train", "failed", {"error": str(e)})
        raise


if __name__ == "__main__":
    main()
