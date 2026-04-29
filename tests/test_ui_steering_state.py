from src.ui.components.concept_steering_panel import _merge_neuron_values
from src.ui.steering_state import (
    build_steering_config,
    get_steering_config,
    set_steering_config,
    steering_config_hash,
)


def test_merge_neuron_values_patch_semantics() -> None:
    existing = {1: 0.4, 5: -0.2}
    patch = {5: 0.0, 8: 1.3}

    merged = _merge_neuron_values(existing, patch)

    assert merged == {1: 0.4, 8: 1.3}


def test_steering_hash_is_content_based() -> None:
    cfg_a = build_steering_config(
        neuron_values={4: 0.5, 9: 0.0},
        alpha=0.3,
        source="concept",
    )
    cfg_b = build_steering_config(
        neuron_values={4: 0.5},
        alpha=0.3,
        source="concept",
    )

    assert steering_config_hash(cfg_a) == steering_config_hash(cfg_b)



def test_concept_then_neuron_edit_keeps_merged_targets() -> None:
    state = {}

    set_steering_config(
        state,
        "user_1",
        neuron_values={2: 0.7, 6: 1.1},
        alpha=0.3,
        source="concept",
        provenance={"entity": "japanese food"},
    )

    concept_cfg = get_steering_config(state, "user_1")
    merged = dict(concept_cfg["neuron_values"])
    merged[6] = 1.4
    merged[11] = -0.3

    set_steering_config(
        state,
        "user_1",
        neuron_values=merged,
        alpha=concept_cfg["alpha"],
        source="neuron",
        provenance={"edited_in": "neuron_panel"},
    )

    final_cfg = get_steering_config(state, "user_1")
    assert final_cfg is not None
    assert final_cfg["neuron_values"][2] == 0.7
    assert final_cfg["neuron_values"][6] == 1.4
    assert final_cfg["neuron_values"][11] == -0.3
