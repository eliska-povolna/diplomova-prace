"""Smoke tests for Streamlit page renderers."""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


class _FakeSAE:
    k = 64


class _FakeInference:
    def __init__(self) -> None:
        self.sae = _FakeSAE()
        self.n_items = 2
        self.user_latents = {}

    def encode_user(self, user_id, user_interactions_csr) -> None:
        self.user_latents[user_id] = [1.0, 0.5]

    def get_top_activations(self, user_z, k=10):
        return [
            {"neuron_idx": i, "label": f"Feature {i}", "activation": 1.0 / (i + 1)}
            for i in range(min(k, 8))
        ]

    def steer_and_recommend(self, user_id, steering_updates, top_k=20):
        return {
            "recommendations": [
                {
                    "poi_idx": 0,
                    "score": 0.9,
                    "contributing_neurons": [{"idx": 1, "label": "Italian"}],
                },
                {
                    "poi_idx": 1,
                    "score": 0.8,
                    "contributing_neurons": [{"idx": 2, "label": "Cafe"}],
                },
            ],
            "latency_ms": 120.0,
        }

    def get_user_history(self, user_id):
        return [0]


class _FakeData:
    num_pois = 2

    def get_test_users(self, limit=50):
        return [{"id": "user_1", "interactions": 2}]

    def get_user_interactions(self, user_id):
        return [0, 1]

    def get_poi_details(self, poi_idx):
        return {
            "poi_idx": poi_idx,
            "name": f"POI {poi_idx}",
            "category": "Cafe",
            "lat": 37.7 + poi_idx * 0.001,
            "lon": -122.4 - poi_idx * 0.001,
            "rating": 4.2,
            "review_count": 42,
            "url": "https://www.yelp.com",
            "photos": [],
            "primary_photo": None,
            "photo_count": 0,
            "business_id": f"biz_{poi_idx}",
        }


class _FakeLabels:
    def get_label(self, neuron_idx):
        return f"Label {neuron_idx}"

    def get_pois_for_neuron(self, neuron_idx, top_k=10):
        return [{"name": "POI 0", "category": "Cafe", "activation": 0.9, "rating": 4.2}]


def _build_app_test(app_fn):
    at = AppTest.from_function(app_fn)
    at.session_state["inference"] = _FakeInference()
    at.session_state["data"] = _FakeData()
    at.session_state["labels"] = _FakeLabels()
    at.run(timeout=60)
    return at


def test_home_page_renders_without_exceptions():
    def _home_app():
        from src.ui.pages import home

        home.show()

    at = _build_app_test(_home_app)
    assert len(at.exception) == 0
    assert len(at.error) == 0


def test_results_page_renders_without_exceptions():
    def _results_app():
        from src.ui.pages import results

        results.show()

    at = _build_app_test(_results_app)
    assert len(at.exception) == 0
    assert len(at.error) == 0


def test_interpretability_page_renders_without_exceptions():
    def _interpretability_app():
        from src.ui.pages import interpretability

        interpretability.show()

    at = _build_app_test(_interpretability_app)
    assert len(at.exception) == 0
    assert len(at.error) == 0


def test_live_demo_page_renders_without_exceptions():
    def _live_demo_app():
        from src.ui.pages import live_demo

        live_demo.show()

    at = _build_app_test(_live_demo_app)
    assert len(at.exception) == 0
    assert len(at.error) == 0
