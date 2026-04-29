"""Feature formatting utilities for consistent display across the UI.

Provides helper functions to format neuron indices and labels in a consistent format:
- Single feature: "#42: Casual Dining [0.73]"
- Feature list: "#42: Casual Dining | #15: Italian Food"
"""

from typing import Dict, List, Optional


def format_feature_id(
    neuron_idx: int, label: str, activation: Optional[float] = None
) -> str:
    """
    Format a single feature as "#42: Casual Dining" or "#42: Casual Dining [0.73]".

    Args:
        neuron_idx: Neuron/feature index
        label: Human-readable label for the feature
        activation: Optional activation value to include in brackets

    Returns:
        Formatted string like "#42: Casual Dining [0.73]"

    Example:
        >>> format_feature_id(42, "Casual Dining", 0.73)
        "#42: Casual Dining [0.73]"
        >>> format_feature_id(42, "Casual Dining")
        "#42: Casual Dining"
    """
    base = f"#{neuron_idx}: {label}"
    if activation is not None:
        return f"{base} [{activation:.2f}]"
    return base


def format_features_list(
    neuron_ids: List[int],
    labels: Dict[int, str],
    activations: Optional[Dict[int, float]] = None,
    separator: str = " | ",
) -> str:
    """
    Format multiple features as a list.

    Args:
        neuron_ids: List of neuron indices to format
        labels: Dict mapping neuron_idx to label string
        activations: Optional dict mapping neuron_idx to activation value
        separator: String to join features (default: " | ")

    Returns:
        Formatted string like "#42: Casual Dining | #15: Italian Food [0.65]"

    Example:
        >>> labels = {42: "Casual Dining", 15: "Italian Food"}
        >>> activations = {42: 0.73, 15: 0.65}
        >>> format_features_list([42, 15], labels, activations)
        "#42: Casual Dining [0.73] | #15: Italian Food [0.65]"
    """
    features = []
    for nid in neuron_ids:
        label = labels.get(nid, "Unknown")
        act = activations.get(nid) if activations else None
        features.append(format_feature_id(nid, label, act))
    return separator.join(features)


def format_feature_explanation(
    neuron_idx: int,
    label: str,
    activation: Optional[float] = None,
    context: str = "This feature",
) -> str:
    """
    Format a feature explanation sentence.

    Args:
        neuron_idx: Neuron index
        label: Feature label
        activation: Optional activation value
        context: Context text (default: "This feature")

    Returns:
        Formatted explanation like "This feature #42: Casual Dining [0.73] influenced the recommendation"

    Example:
        >>> format_feature_explanation(42, "Casual Dining", 0.73, context="The top reason")
        "The top reason #42: Casual Dining [0.73] influenced this recommendation"
    """
    feature_str = format_feature_id(neuron_idx, label, activation)
    return f"{context} {feature_str} influenced this recommendation"


__all__ = ["format_feature_id", "format_features_list", "format_feature_explanation"]
