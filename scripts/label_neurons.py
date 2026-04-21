#!/usr/bin/env python3
"""CLI wrapper for neuron labeling.

This is a convenience script that calls the main labeling logic from src.label.

For the actual implementation, see: src/label.py

Usage:
    python scripts/label_neurons.py [args]
    OR
    python -m src.label [args]

Features:
    - Full labeling + embeddings + superfeatures + coactivations
    - Skip coactivation data generation (--skip-coactivation)
    - Generate only coactivation data (--coactivation-only)
    - Auto-detects latest training run

Examples:
    # Full pipeline
    python scripts/label_neurons.py

    # With custom training directory
    python scripts/label_neurons.py --training-dir outputs/20260420_170147

    # Skip coactivation
    python scripts/label_neurons.py --skip-coactivation

    # Only coactivation data
    python scripts/label_neurons.py --coactivation-only
"""

import sys
from src.label import main


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
