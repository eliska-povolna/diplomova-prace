# Training Pipeline - Quick Start Guide

## Overview

The training pipeline implements ELSA + TopK SAE for interpretable POI recommendation. Run a single command to:
1. Load Yelp review data from Parquet
2. Build user-item interaction CSR matrix
3. Train ELSA model with validation metrics
4. Train TopK SAE on ELSA latent representations
5. Evaluate on test set with Recall@K, NDCG@K, HR@K metrics

---

## Setup

### 1. Install Dependencies

```bash
pip install -r src/requirements.txt
```

### 2. Prepare Data

Place Yelp Parquet data in:
```
data/Yelp-JSON/
├── yelp_parquet/
│   ├── business/state=XX/*.parquet
│   └── review/year=YYYY/*.parquet
└── yelp.duckdb
```

Or edit `configs/default.yaml` to point to your data location.

---

## Running the Pipeline

### Option 1: Quick Start (Recommended)

```bash
python run_training.py
```

This runs training with default configuration and outputs results to `outputs/YYYYMMDD_HHMMSS/`.

### Option 2: Custom Configuration

```bash
python src/train.py --config configs/default.yaml
```

Or create a custom config:
```bash
cp configs/default.yaml configs/custom.yaml
# Edit configs/custom.yaml
python src/train.py --config configs/custom.yaml
```

---

## Output Structure

After training completes, you'll have:

```
outputs/20240316_120000/
├── checkpoints/
│   ├── elsa_best.pt                    # Best ELSA model
│   ├── sae_r4_k32_best.pt              # Best SAE model
│   ├── metrics_elsa_train.json          # ELSA training metrics
│   ├── metrics_sae_train.json           # SAE training metrics
│   └── evaluation_test.json             # Test set evaluation
├── logs/
│   ├── src_train.jsonl                  # Structured training logs
│   └── src_evaluate.jsonl               # Structured evaluation logs
└── summary.json                         # Training summary + config
```

---

## Reading the Output

### 1. Training Logs

Console output shows real-time progress:
```
TRAINING ELSA
Epoch   1/25 | train_loss=0.234567 | val_loss=0.223456
Epoch   2/25 | train_loss=0.212345 | val_loss=0.201234
...
ELSA training complete. Best val_loss=0.195432

TRAINING TOPK SAE
Epoch   1/50 | train_recon=0.123456 train_l1=0.045678 | val_recon=0.118765 cosine_sim=0.8234
Epoch   2/50 | train_recon=0.112345 train_l1=0.034567 | val_recon=0.107654 cosine_sim=0.8456
...
SAE training complete. Best val_recon=0.095432

FINAL EVALUATION ON TEST SET
Test reconstruction loss: 0.098765
Test cosine similarity: 0.8567
Average active neurons: 32.0/2048
```

### 2. Summary File (`summary.json`)

Key metrics for your thesis:
```json
{
  "data": {
    "n_users": 12345,
    "n_items": 54321,
    "n_interactions": 1234567
  },
  "elsa": {
    "best_val_loss": 0.195432
  },
  "sae": {
    "test_recon_loss": 0.098765,
    "test_cosine_sim": 0.8567,
    "avg_active_neurons": 32.0
  }
}
```

### 3. Evaluation Results (`evaluation_test.json`)

Ranking metrics:
```json
{
  "metrics": {
    "Recall@10": 0.2345,
    "Recall@20": 0.3456,
    "Recall@50": 0.4567,
    "NDCG@10": 0.3210,
    "NDCG@20": 0.3876,
    "NDCG@50": 0.4123,
    "HR@10": 0.5678,
    "HR@20": 0.6789,
    "HR@50": 0.7890
  }
}
```

---

## Evaluating a Trained Model

After training, evaluate on validation or test set:

```bash
python src/evaluate.py \
    --checkpoint outputs/20240316_120000/checkpoints \
    --split test \
    --k-values 10 20 50
```

---

## Configuration

Edit `configs/default.yaml` to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.parquet_dir` | `data/Yelp-JSON/yelp_parquet` | Location of Parquet data |
| `data.pos_threshold` | `4.0` | Stars >= this = positive interaction |
| `elsa.latent_dim` | `512` | ELSA latent dimension |
| `elsa.num_epochs` | `25` | ELSA training epochs |
| `sae.width_ratio` | `4` | SAE width = 4 × latent_dim |
| `sae.k` | `32` | SAE sparsity: keep top-k features |
| `elsa.device` | `cuda` | `cuda` or `cpu` |

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch sizes in config:
```yaml
elsa:
  batch_size: 512  # default 1024
sae:
  batch_size: 512  # default 1024
```

### Slow Training

1. Use `cuda` device (check with `torch.cuda.is_available()`)
2. Increase batch size (if memory allows)
3. Reduce `num_epochs` for testing

### Data Not Found

Check `configs/default.yaml`:
```yaml
data:
  parquet_dir: "path/to/your/yelp_parquet"
  db_path: "path/to/yelp.duckdb"
```

---

## Next Steps from Here

Once you have trained models, you can proceed to:
1. **Neuron Interpretation** - Map SAE features to business categories
2. **Manual Labeling** - Label neurons based on top-k features
3. **Query Boosting** - Use labeled features for interactive recommendations

See `src/yelp_initial_exploration/` for reference implementations of these features.
