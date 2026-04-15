"""Training entry point for ELSA + TopK SAE POI recommender.

Pipeline:
  1. Load Yelp review/business data from Parquet via DuckDB
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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.data.preprocessing import apply_kcore_filtering, build_csr
from src.data.yelp_loader import load_businesses, load_reviews
from src.models.collaborative_filtering import ELSA, NMSELoss
from src.models.sparse_autoencoder import TopKSAE
from src.utils import CheckpointManager, Config, load_config, setup_logger

logger = logging.getLogger(__name__)


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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ELSA + TopK SAE POI recommender",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
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


def train_elsa(
    config: Config,
    X_train,
    X_val,
    n_items: int,
    checkpoint_mgr: CheckpointManager,
) -> tuple[ELSA, float]:
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
    tuple[ELSA, float]
        Trained ELSA model and best validation loss.
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

    logger.info(
        f"Config: latent_dim={elsa_cfg['latent_dim']}, "
        f"lr={elsa_cfg['learning_rate']}, epochs={elsa_cfg['num_epochs']}"
    )

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
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

    # Load best model
    model.load_state_dict(
        torch.load(checkpoint_mgr.checkpoint_dir / "elsa_best.pt", map_location=device)[
            "model_state_dict"
        ]
    )
    checkpoint_mgr.save_metrics(metrics.to_dict(), split="elsa_train")

    logger.info(f"ELSA training complete. Best val_loss={best_val_loss:.6f}")
    return model, best_val_loss


def train_sae(
    config: Config,
    elsa_model: ELSA,
    Z_train: torch.Tensor,
    Z_val: torch.Tensor,
    checkpoint_mgr: CheckpointManager,
) -> tuple[TopKSAE, float]:
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
    tuple[TopKSAE, float]
        Trained SAE model and best validation loss.
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

    logger.info(
        f"Config: width_ratio={sae_cfg['width_ratio']}, "
        f"hidden_dim={hidden_dim}, k={sae_cfg['k']}, "
        f"l1_coef={sae_cfg['l1_coef']}, epochs={sae_cfg['num_epochs']}"
    )

    for epoch in range(sae_cfg["num_epochs"]):
        # Training
        sae.train()
        train_recon_loss = 0.0
        train_l1_loss = 0.0

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

        train_recon_loss /= len(Z_train)
        train_l1_loss /= len(Z_train)

        # Validation
        sae.eval()
        val_recon_loss = 0.0
        cosine_sims = []

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

        val_recon_loss /= len(Z_val)
        avg_cosine_sim = np.mean(cosine_sims)

        metrics.record(
            epoch,
            train_recon=train_recon_loss,
            train_l1=train_l1_loss,
            val_recon=val_recon_loss,
            cosine_sim=avg_cosine_sim,
        )

        logger.info(
            f"Epoch {epoch+1:3d}/{sae_cfg['num_epochs']} | "
            f"train_recon={train_recon_loss:.6f} train_l1={train_l1_loss:.6f} | "
            f"val_recon={val_recon_loss:.6f} cosine_sim={avg_cosine_sim:.4f}"
        )

        # Early stopping
        if (best_val_loss - val_recon_loss) > sae_cfg["min_delta"]:
            best_val_loss = val_recon_loss
            patience_counter = 0
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
                logger.info(f"Early stopping after {epoch + 1} epochs")
                break

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
    return sae, best_val_loss


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory with timestamp
    output_cfg = config["output"]
    output_dir = Path(output_cfg["base_dir"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

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

        parquet_dir = Path(config["data"]["parquet_dir"])
        db_path = config["data"]["db_path"]

        if not parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {parquet_dir}")

        # Load with all applicable filters from config
        # ⭐ FIRST: Create UNIVERSAL mappings from ALL data BEFORE any filtering
        # This ensures we have complete mappings for all possible items
        logger.info("Creating universal item/business mappings...")
        all_reviews = load_reviews(
            parquet_dir,
            db_path=db_path,
            pos_threshold=config["data"]["pos_threshold"],
            year_min=config["data"].get("year_min"),
            year_max=config["data"].get("year_max"),
        )

        # Build universal mappings from all data
        all_users = all_reviews["user_id"].unique()
        all_businesses = all_reviews["business_id"].unique()

        universal_user_map = {uid: idx for idx, uid in enumerate(all_users)}
        universal_business_map = {bid: idx for idx, bid in enumerate(all_businesses)}

        logger.info("Universal mappings created:")
        logger.info(f"  Total unique users: {len(universal_user_map)}")
        logger.info(f"  Total unique businesses: {len(universal_business_map)}")

        # Save universal mappings for downstream use (e.g., labeling notebook)
        import pickle

        mappings_dir = output_dir / "mappings"
        mappings_dir.mkdir(parents=True, exist_ok=True)

        with open(mappings_dir / "user2index_universal.pkl", "wb") as f:
            pickle.dump(universal_user_map, f)
        with open(mappings_dir / "business2index_universal.pkl", "wb") as f:
            pickle.dump(universal_business_map, f)

        logger.info(f"Universal mappings saved to {mappings_dir}")

        # ⭐ NOW apply filtering on top of universal data
        reviews = all_reviews.copy()

        # Filter by state (if specified)
        state_filter = config["data"].get("state_filter")
        if state_filter:
            businesses = load_businesses(
                parquet_dir,
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
        logger.info(
            f"Built CSR: {X_csr.shape[0]} users × {X_csr.shape[1]} items, "
            f"{X_csr.nnz} interactions"
        )

        # Apply k-core filtering
        logger.info("Applying k-core filtering (k=5)...")
        X_csr = apply_kcore_filtering(X_csr, k=5)
        logger.info(
            f"After k-core: {X_csr.shape[0]} users × {X_csr.shape[1]} items, "
            f"{X_csr.nnz} interactions"
        )

        # Train/test split
        n_users = X_csr.shape[0]
        user_indices = np.arange(n_users)
        train_users, test_users = train_test_split(
            user_indices,
            test_size=1 - config["data"]["train_test_split"],
            random_state=config["data"]["seed"],
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
            test_size=config["data"]["val_split"],
            random_state=config["data"]["seed"],
        )

        # Create subset datasets
        from torch.utils.data import Subset

        X_train_split = Subset(X_train_dataset, train_idx)
        X_val_split = Subset(X_train_dataset, val_idx)

        # Train ELSA
        elsa_model, elsa_best_loss = train_elsa(
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
        sae_model, sae_best_loss = train_sae(
            config, elsa_model, Z_train, Z_val, checkpoint_mgr
        )

        # Final evaluation on test set
        logger.info("=" * 60)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 60)

        sae_model.eval()
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

        logger.info(f"Test reconstruction loss: {test_recon_loss:.6f}")
        logger.info(f"Test cosine similarity: {test_cosine_sim:.4f}")
        logger.info(
            f"Average active neurons: {active_neurons:.1f}/{config['sae']['width_ratio'] * config['elsa']['latent_dim']}"
        )

        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": config.to_dict(),
            "data": {
                "n_users": n_users,
                "n_items": X_csr.shape[1],
                "n_interactions": X_csr.nnz,
            },
            "elsa": {
                "best_val_loss": float(elsa_best_loss),
            },
            "sae": {
                "test_recon_loss": float(test_recon_loss),
                "test_cosine_sim": float(test_cosine_sim),
                "avg_active_neurons": float(active_neurons),
            },
            "output_dir": str(output_dir),
        }

        summary_path = output_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Summary saved to: {summary_path}")

    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
