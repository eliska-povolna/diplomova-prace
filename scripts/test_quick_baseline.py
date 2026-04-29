"""Quick sanity test for baseline pipeline on a small subset of data.

Runs ALS on first N users and M items (subsampled) using per-user holdout
and prints NDCG@20 to ensure metrics are > 0.
"""
from pathlib import Path
import pickle
import numpy as np
from scipy.sparse import load_npz, csr_matrix

import importlib.util
from pathlib import Path

# Import train_baseline as a module by file path (scripts is not a package)
tb_path = Path(__file__).parent / "train_baseline.py"
spec = importlib.util.spec_from_file_location("train_baseline", str(tb_path))
tb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tb)
ALSBaseline = tb.ALSBaseline
compute_ranking_metrics_batched = tb.compute_ranking_metrics_batched

DATA_DIR = Path(__file__).parent.parent / "data" / "preprocessed_yelp"

R = load_npz(DATA_DIR / "R_PA_compact.npz")
with open(DATA_DIR / "user2index.pkl","rb") as f:
    user2index = pickle.load(f)
with open(DATA_DIR / "item2index.pkl","rb") as f:
    item2index = pickle.load(f)

# Subsample
N_USERS = 500
N_ITEMS = 2000
R = R[:N_USERS, :N_ITEMS].tocsr()
print("Subsample shape:", R.shape, "nnz=", R.nnz)

# Per-user holdout
rng = np.random.default_rng(42)
train = R.tolil(copy=True)
test = csr_matrix(R.shape, dtype=R.dtype).tolil()
for u in range(R.shape[0]):
    items = R[u].nonzero()[1]
    if len(items) <= 1:
        continue
    k = max(1, int(len(items) * 0.2))
    held = rng.choice(items, size=k, replace=False)
    for it in held:
        train[u, it] = 0
        test[u, it] = R[u, it]
train = train.tocsr()
test = test.tocsr()

print("Train nnz:", train.nnz, "Test nnz:", test.nnz)

# Train quick ALS
model = ALSBaseline(n_items=train.shape[1], factors=16, regularization=0.1, iterations=5, use_gpu=False)
model.fit(train)
# Batched evaluation with train masking
metrics = compute_ranking_metrics_batched(model, test, train_csr=train, batch_size=64, k_values=[20])
print("Quick test metrics:", metrics)
