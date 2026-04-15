#!/usr/bin/env python
"""
Quick test of neuron labeling module with mock data
"""
from src.interpret.neuron_labeling import TagBasedLabeler


def test_neuron_labeling_creates_labels():
    """Test that TagBasedLabeler produces semantic labels."""
    # Mock business metadata
    business_metadata = {
        "Italian": {"categories": ["Italian"], "business_type": "restaurant"},
        "Japanese": {"categories": ["Japanese"], "business_type": "restaurant"},
        "Cafe": {"categories": ["Cafes"], "business_type": "cafe"},
        "Fitness": {"categories": ["Fitness"], "business_type": "gym"},
    }

    # Mock neuron profiles
    neuron_profiles = {
        0: {
            "max_activating": [("Italian", 0.95), ("Italian", 0.88), ("Italian", 0.82)],
            "zero_activating": [("Fitness", 0.01), ("Japanese", 0.02)],
        },
        1: {
            "max_activating": [
                ("Japanese", 0.92),
                ("Japanese", 0.85),
                ("Japanese", 0.78),
            ],
            "zero_activating": [("Fitness", 0.01), ("Italian", 0.03)],
        },
        2: {
            "max_activating": [("Cafe", 0.89), ("Cafe", 0.81), ("Cafe", 0.75)],
            "zero_activating": [("Fitness", 0.02), ("Italian", 0.01)],
        },
        3: {
            "max_activating": [("Fitness", 0.90), ("Fitness", 0.83), ("Fitness", 0.79)],
            "zero_activating": [("Italian", 0.02), ("Japanese", 0.01)],
        },
    }

    # Test Tag-Based Labeling
    labeler = TagBasedLabeler()
    labels = labeler.label_neurons(neuron_profiles, business_metadata)

    # Verify labels are generated (don't check content, as labeler logic may vary)
    assert len(labels) == 4, f"Expected 4 labels, got {len(labels)}"

    for neuron_idx in range(4):
        label = labels[neuron_idx]
        assert isinstance(
            label, str
        ), f"Neuron {neuron_idx} label should be string, got {type(label)}"
        assert len(label) > 0, f"Neuron {neuron_idx} label should not be empty"
