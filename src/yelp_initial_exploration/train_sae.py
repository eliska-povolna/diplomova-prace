import os
import pickle
import numpy as np
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from train_elsa import ELSA, latent_dim

width_ratio = 4
hidden_dim = latent_dim * width_ratio
k = 32

lr = 3e-4
l1_coef = 3e-4
batch_size = 1024
n_epochs = 50
patience = 10
min_delta = 1e-4
val_split = 0.1
seed = 42
device = "cuda" if torch.cuda.is_available() else "cpu"

# I/O paths
DATA_PKL = "data/processed_train.pkl"
ELSA_CKPT = "models/elsa_model_best.pt"
SAE_CKPT = f"models/sae_model_r{width_ratio}_k{k}.pt"
SAE_META = f"models/sae_meta_r{width_ratio}_k{k}.pt"
SPARSE_EMB = f"models/sparse_embeddings_r{width_ratio}_k{k}.pt"


def set_seed(s: int = 42):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_recon(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity loss as mentioned in paper"""
    an = torch.nn.functional.normalize(a, dim=-1)
    bn = torch.nn.functional.normalize(b, dim=-1)
    return 1.0 - torch.nn.functional.cosine_similarity(an, bn, dim=-1).mean()


def topk_mask(x: torch.Tensor, k_: int) -> torch.Tensor:
    vals, idx = torch.topk(x, k_, dim=1)
    return torch.zeros_like(x).scatter(1, idx, 1.0)


def encode_in_chunks(X_csr, A: torch.Tensor, bs: int = 4096) -> torch.Tensor:
    """Encode sparse CSR matrix X_csr to Z = normalize(XA) in chunks to avoid OOM."""
    Z_parts = []
    n = X_csr.shape[0]
    for s in range(0, n, bs):
        xb = torch.tensor(
            X_csr[s : s + bs].toarray(), dtype=torch.float32, device=device
        )
        zb = xb @ A
        zb = torch.nn.functional.normalize(zb, dim=-1)
        Z_parts.append(zb)
        del xb, zb
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return torch.cat(Z_parts, dim=0)


class TopKSAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        super().__init__()
        self.k = k
        self.enc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.dec = nn.Linear(hidden_dim, input_dim, bias=True)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_pre = self.enc(x)

        # Signed Top-K: select by magnitude, keep the sign
        m = topk_mask(h_pre.abs(), self.k)
        h_sparse = h_pre * m

        out = self.dec(h_sparse)
        return out, h_sparse, h_pre


def main():
    set_seed(seed)
    os.makedirs("models", exist_ok=True)

    print(f"Training TopK SAE with paper specifications:")
    print(f"  ELSA latent_dim: {latent_dim}")
    print(f"  Width ratio: {width_ratio} → hidden_dim: {hidden_dim}")
    print(f"  Target sparsity: k={k} ({100.0 * k / hidden_dim:.1f}%)")
    print(f"  L1 coefficient: {l1_coef}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {n_epochs}")
    print(f"  Patience: {patience}")

    # Load interactions (CSR, keep sparse)
    with open(DATA_PKL, "rb") as f:
        X = pickle.load(f)
    n_users, n_items = X.shape

    # ELSA → user latent z (raw), then L2-normalize (encoded in chunks to save RAM)
    elsa = ELSA(n_items, latent_dim).to(device)
    elsa.load_state_dict(torch.load(ELSA_CKPT, map_location=device))
    elsa.eval()
    with torch.no_grad():
        Z = encode_in_chunks(X, elsa.A)

    # Split
    idx = np.arange(Z.shape[0])
    tr_idx, va_idx = train_test_split(idx, test_size=val_split, random_state=seed)
    Z_tr = Z[tr_idx]
    Z_va = Z[va_idx]

    # Model
    sae = TopKSAE(input_dim=latent_dim, hidden_dim=hidden_dim, k=k).to(device)
    optimizer = optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Z_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val = float("inf")
    no_improve = 0

    for epoch in range(n_epochs):
        sae.train()
        tr_rec, tr_l1 = 0.0, 0.0

        for (xbatch,) in loader:
            optimizer.zero_grad()
            recon, h_sparse, h_pre = sae(xbatch)

            # Simplified loss: reconstruction + small L1 penalty
            rec_loss = cosine_recon(recon, xbatch)
            l1_loss = h_sparse.abs().mean()
            loss = rec_loss + l1_coef * l1_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()

            bs = xbatch.size(0)
            tr_rec += rec_loss.item() * bs
            tr_l1 += l1_loss.item() * bs

        tr_rec /= len(Z_tr)
        tr_l1 /= len(Z_tr)

        # Validation
        sae.eval()
        with torch.no_grad():
            r_va, h_va, h_pre_va = sae(Z_va)
            va_rec = cosine_recon(r_va, Z_va).item()

            # Compute reconstruction cosine similarity
            r_va_norm = torch.nn.functional.normalize(r_va, dim=-1)
            Z_va_norm = torch.nn.functional.normalize(Z_va, dim=-1)
            cosine_sim = (
                torch.nn.functional.cosine_similarity(r_va_norm, Z_va_norm, dim=-1)
                .mean()
                .item()
            )

            avg_active = float(k)
            sparsity_pct = 100.0 * avg_active / hidden_dim
            actual_active = (h_va != 0).sum(dim=1).float().mean().item()

        print(
            f"Epoch {epoch+1:03d} | k={k:02d} | "
            f"train_rec={tr_rec:.6f} l1={tr_l1:.6f} | "
            f"val_rec={va_rec:.6f} | "
            f"cosine_sim={cosine_sim:.3f} | "
            f"sparsity={avg_active:.1f}/{hidden_dim} ({sparsity_pct:.1f}%) | "
            f"actual_active={actual_active:.1f}"
        )

        # Early stopping with min_delta
        if (best_val - va_rec) > min_delta:
            best_val = va_rec
            no_improve = 0
            sd = sae.state_dict()
            if any(k.startswith("module.") for k in sd.keys()):
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
            torch.save(sd, SAE_CKPT)
            torch.save(
                {
                    "hidden_dim": hidden_dim,
                    "k": k,
                    "width_ratio": width_ratio,
                    "latent_dim": latent_dim,
                    "l1_coef": l1_coef,
                },
                SAE_META,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    # Final evaluation and export
    sae.load_state_dict(torch.load(SAE_CKPT, map_location=device))
    sae.eval()

    # Compute final metrics
    with torch.no_grad():
        r_final, h_final, _ = sae(Z_va)
        r_norm = torch.nn.functional.normalize(r_final, dim=-1)
        z_norm = torch.nn.functional.normalize(Z_va, dim=-1)
        final_cosine_sim = (
            torch.nn.functional.cosine_similarity(r_norm, z_norm, dim=-1).mean().item()
        )

    # Streaming selection frequency
    counts = torch.zeros(hidden_dim, dtype=torch.float64, device=device)
    B = 4096
    with torch.no_grad():
        for s in range(0, Z.shape[0], B):
            hb = sae.enc(Z[s : s + B])
            mb = topk_mask(hb.abs(), k).sum(dim=0)
            counts += mb.to(dtype=torch.float64)
            del hb, mb
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
    sel_freq = (counts / Z.shape[0]).cpu().numpy()

    active_any = int((sel_freq > 0).sum())
    overly_general = int((sel_freq > 0.50).sum())
    highly_specific = int((sel_freq < (10 / Z.shape[0])).sum())

    print(f"\nFinal Results:")
    print(f"  Width ratio: {width_ratio} (hidden_dim: {hidden_dim})")
    print(
        f"  Target sparsity: {k}/{hidden_dim} neurons active ({100.0 * k / hidden_dim:.1f}%)"
    )
    print(f"  L1 coefficient: {l1_coef}")
    print(f"  Training epochs: {epoch + 1}")
    print(f"  Best validation loss: {best_val:.6f}")
    print(f"  Final reconstruction cosine similarity: {final_cosine_sim:.3f}")
    print(f"  Active neurons (selected at least once): {active_any}/{hidden_dim}")
    print(f"  Overly general neurons (>50% users select): {overly_general}")
    print(f"  Highly specific neurons (<10 users select): {highly_specific}")

    # Compact export: per-user top-k indices and signed values (saves memory)
    topk_idx_parts, topk_val_parts = [], []
    with torch.no_grad():
        for s in range(0, Z.shape[0], B):
            hb = sae.enc(Z[s : s + B])
            val, idx = torch.topk(hb.abs(), k, dim=1)
            signed = hb.gather(1, idx).cpu()
            topk_idx_parts.append(idx.cpu().to(torch.int32))
            topk_val_parts.append(signed)
            del hb, val, idx, signed
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    export = {
        "idx": torch.cat(topk_idx_parts, dim=0),
        "val": torch.cat(topk_val_parts, dim=0),
        "k": k,
        "hidden_dim": hidden_dim,
        "width_ratio": width_ratio,
        "latent_dim": latent_dim,
    }
    torch.save(export, SPARSE_EMB)

    # Save training results
    results = {
        "final_cosine_similarity": final_cosine_sim,
        "val_loss": best_val,
        "width_ratio": width_ratio,
        "hidden_dim": hidden_dim,
        "k": k,
        "l1_coef": l1_coef,
        "epochs": epoch + 1,
        "sparsity_percent": 100.0 * k / hidden_dim,
        "active_neurons": active_any,
        "overly_general": overly_general,
        "highly_specific": highly_specific,
    }

    import json

    with open(f"models/sae_results_r{width_ratio}_k{k}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:")
    print(f"  {SAE_CKPT}")
    print(f"  {SAE_META}")
    print(f"  {SPARSE_EMB}")
    print(f"  models/sae_results_r{width_ratio}_k{k}.json")


if __name__ == "__main__":
    main()
