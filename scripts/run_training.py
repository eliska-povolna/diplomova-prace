"""Quick training script for the ELSA + SAE pipeline.

This is a convenience wrapper that runs the full training with sensible defaults.
For more control, use: python src/train.py --config configs/default.yaml

Automatically creates per-state output directories based on config state_filter.
"""

import subprocess
import sys
import os
import yaml
from pathlib import Path
from datetime import datetime

# Get the repo root
repo_root = Path(__file__).parent.parent

# Load config to detect state
config_path = repo_root / "configs" / "default.yaml"

if not config_path.exists():
    print(f"ERROR: Config file not found: {config_path}")
    sys.exit(1)

with open(config_path) as f:
    config = yaml.safe_load(f)

state = config.get("data", {}).get("state_filter", "FULL")
if state is None:
    state = "FULL"

# Create timestamp-based output directory with state prefix
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
state_prefix = f"{state}_" if state else ""
output_dir = repo_root / "outputs" / f"{state_prefix}{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"RUNNING TRAINING PIPELINE (State: {state})")
print(f"Output directory: {output_dir}")
print("=" * 70)

# Set PYTHONPATH to include repo root for module imports
env = os.environ.copy()
env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

result = subprocess.run(
    [sys.executable, "src/train.py", "--config", str(config_path), "--output_dir", str(output_dir)],
    cwd=repo_root,
    env=env,
)

sys.exit(result.returncode)
