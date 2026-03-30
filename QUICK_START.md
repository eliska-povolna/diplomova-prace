# Quick Start: Train & Evaluate in 3 Steps

## Step 1: Ensure Data is in Place

```bash
ls data/Yelp-JSON/yelp_parquet/
# Should show: business/  review/
```

If not there, edit path in `configs/default.yaml`:
```yaml
data:
  parquet_dir: "your/path/to/yelp_parquet"
```

## Step 2: Install Requirements (if not already done)

```bash
pip install -r src/requirements.txt
```

## Step 3: Run Training

### Option A: One-liner (Recommended)
```bash
python run_training.py
```

### Option B: With custom config
```bash
python src/train.py --config configs/default.yaml
```

## What to Expect

### During Training (~5-30 minutes depending on hardware)

You'll see:
```
TRAINING ELSA
Epoch   1/25 | train_loss=0.2345 | val_loss=0.2234
Epoch   2/25 | train_loss=0.1234 | val_loss=0.1167
...
ELSA training complete. Best val_loss=0.0954

TRAINING TOPK SAE  
Epoch   1/50 | train_recon=0.1234 train_l1=0.0456 | val_recon=0.1187 cosine_sim=0.8234
...
SAE training complete. Best val_recon=0.0954

FINAL EVALUATION ON TEST SET
Test reconstruction loss: 0.0987
Test cosine similarity: 0.8567
Average active neurons: 32.0/2048
```

### After Training (~30 seconds for evaluation)

Find output in `outputs/YYYYMMDD_HHMMSS/`:

```bash
# View training summary
cat outputs/20240316_120000/summary.json

# View evaluation results  
cat outputs/20240316_120000/evaluation_test.json

# Check training logs
tail outputs/20240316_120000/logs/*.jsonl
```

## Key Metrics to Look For

| Metric | What It Means | Good Value |
|--------|---------------|-----------|
| **ELSA val_loss** | How well ELSA reconstructs user interactions | < 0.1 |
| **SAE cosine_sim** | How well SAE preserves information | > 0.80 |
| **Recall@20** | % relevant items ranked in top-20 | > 0.20 |
| **NDCG@20** | Quality of ranking (best items first) | > 0.25 |
| **Active neurons** | SAE sparsity (k=32 target) | ≈ 32 |

## If Training Fails

### Out of Memory
Edit `configs/default.yaml`:
```yaml
elsa:
  batch_size: 512  # was 1024
sae:
  batch_size: 512
```

### Data not found
```bash
python -c "from pathlib import Path; Path('data/Yelp-JSON/yelp_parquet').exists()"
# Should print: True
```

### GPU not available (falls back to CPU, slow)
```python
import torch
print(torch.cuda.is_available())  # Should be True for fast training
```

## After Training Works

See: [TRAINING_README.md](TRAINING_README.md) for full documentation

Next step: Neuron interpretation & manual labeling

---

**TL;DR:**
```bash
python run_training.py && ls -la outputs/*/summary.json
```

That's it! 🚀
