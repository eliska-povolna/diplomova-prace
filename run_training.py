"""Quick training script for the ELSA + SAE pipeline.

This is a convenience wrapper that runs the full training with sensible defaults.
For more control, use: python src/train.py --config configs/default.yaml
"""

import subprocess
import sys
import os
from pathlib import Path

# Get the repo root
repo_root = Path(__file__).parent

# Run training
print("=" * 70)
print("RUNNING TRAINING PIPELINE")
print("=" * 70)

config_path = repo_root / "configs" / "default.yaml"

if not config_path.exists():
    print(f"ERROR: Config file not found: {config_path}")
    sys.exit(1)

# Set PYTHONPATH to include repo root for module imports
env = os.environ.copy()
env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")

result = subprocess.run(
    [sys.executable, "src/train.py", "--config", str(config_path)],
    cwd=repo_root,
    env=env,
)

sys.exit(result.returncode)
