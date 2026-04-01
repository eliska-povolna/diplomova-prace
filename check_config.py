#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from pathlib import Path
from src.ui.cache import load_config

config = load_config(Path('configs/default.yaml'))
print(f"state_filter: {config.get('state_filter')}")
print(f"parquet_dir: {config.get('parquet_dir')}")
print(f"n_items: {config.get('n_items')}")
