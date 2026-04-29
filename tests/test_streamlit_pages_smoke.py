"""Smoke tests for Streamlit page renderers."""

from __future__ import annotations

import tempfile
from pathlib import Path

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
    state_filter = "PA"

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

    def get_dataset_summary(self, scope="training", state=None, city=None, pos_threshold=None):
        return {
            "n_businesses": 10,
            "n_users": 20,
            "n_items": 10,
            "n_interactions": 100,
            "density_pct": 1.0,
            "min_year": 2018,
            "max_year": 2022,
        }

    def get_state_distribution(self, limit=60):
        import pandas as pd

        return pd.DataFrame(
            [{"state": "PA", "n_businesses": 10, "avg_rating": 3.8}]
        )

    def get_top_cities(self, scope="training", state=None, limit=20):
        import pandas as pd

        return pd.DataFrame(
            [{"city": "Philadelphia", "state": "PA", "n_businesses": 10, "avg_rating": 3.9}]
        )

    def get_rating_distribution(self, scope="training", state=None, city=None):
        import pandas as pd

        return pd.DataFrame([{"stars": 4, "review_count": 50}, {"stars": 5, "review_count": 30}])

    def get_review_volume_by_year(
        self,
        scope="training",
        state=None,
        city=None,
        rating_min=None,
        rating_max=None,
    ):
        import pandas as pd

        return pd.DataFrame(
            [{"year": 2020, "review_count": 40}, {"year": 2021, "review_count": 60}]
        )

    def get_top_categories(self, scope="training", state=None, city=None, limit=20):
        import pandas as pd

        return pd.DataFrame(
            [{"categories": "Restaurants, Pizza", "n_businesses": 5, "avg_rating": 4.0}]
        )

    def get_activity_distributions(self, scope="training", state=None, city=None, pos_threshold=None):
        import pandas as pd

        return (
            pd.DataFrame([{"cnt": 1, "num_users": 10}, {"cnt": 2, "num_users": 5}]),
            pd.DataFrame([{"cnt": 1, "num_items": 7}, {"cnt": 2, "num_items": 3}]),
        )

    def get_sample_business_rows(self, scope="training", state=None, city=None, limit=10):
        import pandas as pd

        return pd.DataFrame(
            [{"business_id": "b1", "name": "Test Biz", "city": "Philadelphia", "state": "PA"}]
        )

    def get_sample_review_rows(
        self,
        scope="training",
        state=None,
        city=None,
        year_min=None,
        year_max=None,
        rating_min=None,
        rating_max=None,
        limit=10,
    ):
        import pandas as pd

        return pd.DataFrame(
            [{"review_id": "r1", "text": "Great place", "business_name": "Test Biz"}]
        )


class _FakeLabels:
    def get_label(self, neuron_idx):
        return f"Label {neuron_idx}"

    def get_pois_for_neuron(self, neuron_idx, top_k=10):
        return [{"name": "POI 0", "category": "Cafe", "activation": 0.9, "rating": 4.2}]


def _build_app_test(app_fn):
    tempfile.tempdir = str(Path.cwd())
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


def test_dataset_statistics_page_renders_without_exceptions():
    def _dataset_app():
        from src.ui.pages import dataset_statistics

        dataset_statistics.show()

    at = _build_app_test(_dataset_app)
    assert len(at.exception) == 0
    assert len(at.error) == 0
