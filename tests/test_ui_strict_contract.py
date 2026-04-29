import numpy as np
import pytest
from pathlib import Path

from src.ui.cache import _validate_item2index_mapping, _validate_precomputed_matrices_payload


def test_strict_precomputed_payload_shape_mismatch_raises() -> None:
    payload = {
        "run_id": "20260423_123055",
        "n_items": 17388,
        "matrices": {
            "user_a": np.zeros((1, 12793), dtype=np.float32),
        },
    }

    with pytest.raises(RuntimeError) as exc:
        _validate_precomputed_matrices_payload(
            payload,
            run_id="20260423_123055",
            expected_n_items=17388,
            source_path=Path("outputs/20260423_123055/precomputed/user_csr_matrices.pkl"),
        )

    text = str(exc.value)
    assert "User matrix shape mismatch" in text
    assert "expected_second_dim=17388" in text
    assert "actual_shape=(1, 12793)" in text


def test_strict_item2index_range_sanity_raises() -> None:
    bad_item2index = {
        "biz_1": 1,
        "biz_2": 2,
        "biz_3": 3,
    }

    with pytest.raises(RuntimeError) as exc:
        _validate_item2index_mapping(
            bad_item2index,
            source_path=Path("outputs/20260423_123055/mappings/item2index.pkl"),
            expected_n_items=3,
        )

    text = str(exc.value)
    assert "item2index index range mismatch" in text
    assert "expected_min=0" in text
