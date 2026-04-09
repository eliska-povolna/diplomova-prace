#!/usr/bin/env python
"""Test service initialization to find errors."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

print("=" * 60)
print("TEST: Service Initialization")
print("=" * 60)

# Load config
config_path = Path("configs/default.yaml")
print(f"\n1. Loading config from {config_path}...")
try:
    with open(config_path) as f:
        raw_config = yaml.safe_load(f)
    print(f"   [OK] Config loaded. Keys: {list(raw_config.keys())}")
except Exception as e:
    print(f"   [ERROR] Config error: {e}")
    sys.exit(1)

# Flatten config (as cache.py does)
config = {}
if "data" in raw_config:
    config["duckdb_path"] = raw_config["data"].get("db_path", "")
    config["parquet_dir"] = raw_config["data"].get("parquet_dir", "")

if "elsa" in raw_config:
    config["latent_dim"] = raw_config["elsa"].get("latent_dim", 512)
    config["device"] = raw_config["elsa"].get("device", "cpu")

if "sae" in raw_config:
    config["k"] = raw_config["sae"].get("k", 32)
    config["width_ratio"] = raw_config["sae"].get("width_ratio", 4)

config["steering_alpha"] = 0.3
config["model_checkpoint_dir"] = raw_config.get("output", {}).get("base_dir", "outputs")
config["neuron_labels_path"] = "outputs/neuron_labels.json"
config["n_items"] = 50000  # Placeholder

print(f"   Config keys: {list(config.keys())}")

# Try to import services
print("\n2. Importing services...")
try:
    from src.ui.services import (
        InferenceService,
        DataService,
        LabelingService,
        ModelLoader,
    )
    print("   [OK] Services imported")
except Exception as e:
    print(f"   [ERROR] Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Try to find checkpoint
print("\n3. Finding model checkpoint...")
try:
    outputs_dir = Path(config["model_checkpoint_dir"]).parent
    print(f"   Looking in: {outputs_dir}")
    checkpoint_dir = ModelLoader.find_latest_checkpoint(outputs_dir)
    if checkpoint_dir:
        print(f"   [OK] Found checkpoint: {checkpoint_dir}")
    else:
        print(f"   [WARN] No checkpoint found")
except Exception as e:
    print(f"   [ERROR] Checkpoint error: {e}")
    import traceback
    traceback.print_exc()

# Try to initialize data service
print("\n4. Initializing DataService...")
try:
    data_service = DataService(
        duckdb_path=Path(config["duckdb_path"]),
        parquet_dir=Path(config["parquet_dir"]),
        config=config,
    )
    print(f"   [OK] DataService initialized: {data_service.num_pois} POIs")
except Exception as e:
    print(f"   [ERROR] DataService error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
