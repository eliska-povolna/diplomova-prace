"""Helpers for reproducible training and evaluation runs."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def set_global_reproducibility(seed: int, *, deterministic: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy, and PyTorch state for reproducible runs."""
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn_deterministic = False
    cudnn_benchmark = False
    deterministic_algorithms = False

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            cudnn_deterministic = True
            cudnn_benchmark = bool(torch.backends.cudnn.benchmark)

        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_algorithms = bool(
                getattr(torch, "are_deterministic_algorithms_enabled", lambda: False)()
            )

    if hasattr(torch, "set_num_threads"):
        existing_threads = torch.get_num_threads()
    else:
        existing_threads = None

    return {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cuda_available": cuda_available,
        "deterministic_requested": deterministic,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
        "deterministic_algorithms_enabled": deterministic_algorithms,
        "torch_version": torch.__version__,
        "torch_num_threads": existing_threads,
    }


def build_dataloader_generator(seed: int) -> torch.Generator:
    """Build a seeded generator for DataLoader shuffling."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
