"""Tests for dataset statistics helpers in DataService."""

from __future__ import annotations

import uuid
from pathlib import Path

import duckdb
import pytest

from src.ui.services.data_service import DataService


def _seed_stats_db(db_path: Path) -> None:
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """
            CREATE TABLE yelp_business (
                business_id TEXT,
                name TEXT,
                city TEXT,
                state TEXT,
                stars DOUBLE,
                review_count INTEGER,
                categories TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE yelp_review (
                review_id TEXT,
                user_id TEXT,
                business_id TEXT,
                stars DOUBLE,
                text TEXT,
                date TIMESTAMP
            )
            """
        )

        con.executemany(
            """
            INSERT INTO yelp_business (business_id, name, city, state, stars, review_count, categories)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("b1", "Biz PA 1", "Philadelphia", "PA", 4.5, 10, "Restaurants, Pizza"),
                ("b2", "Biz PA 2", "Pittsburgh", "PA", 4.0, 8, "Food, Coffee & Tea"),
                ("b3", "Biz CA 1", "Los Angeles", "CA", 3.5, 6, "Restaurants, Mexican"),
            ],
        )
        con.executemany(
            """
            INSERT INTO yelp_review (review_id, user_id, business_id, stars, text, date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                ("r1", "u1", "b1", 5.0, "Excellent", "2020-01-10"),
                ("r2", "u2", "b1", 4.0, "Great", "2021-02-01"),
                ("r3", "u3", "b2", 3.0, "Okay", "2021-03-02"),
                ("r4", "u4", "b3", 5.0, "Amazing", "2022-04-05"),
                ("r5", "u1", "b3", 2.0, "Bad", "2022-06-09"),
            ],
        )
    finally:
        con.close()


@pytest.fixture()
def local_db_path() -> Path:
    # Avoid temp subdirectories in this environment; file in repo root is reliable.
    return Path.cwd() / f"ds_stats_{uuid.uuid4().hex}.duckdb"


def test_dataset_summary_global_vs_training_scope(local_db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.ui.services.secrets_helper.get_cloudsql_config",
        lambda: {"instance": None, "database": None, "user": None, "password": None},
    )
    monkeypatch.setattr(
        "src.ui.services.secrets_helper.get_cloud_storage_bucket",
        lambda: None,
        raising=False,
    )
    db_path = local_db_path
    _seed_stats_db(db_path)

    service = DataService(
        duckdb_path=db_path,
        config={"state_filter": "PA", "pos_threshold": 4.0},
        local_photos_dir=Path.cwd() / "missing_photos_dir",
    )
    try:
        global_summary = service.get_dataset_summary(scope="global")
        training_summary = service.get_dataset_summary(scope="training")

        assert int(global_summary["n_businesses"]) == 3
        assert int(training_summary["n_businesses"]) == 2
        assert int(global_summary["n_interactions"]) > int(training_summary["n_interactions"])
        assert float(training_summary["density_pct"]) >= 0.0
    finally:
        service.close()
        if db_path.exists():
            db_path.unlink()


def test_dataset_filters_and_samples(local_db_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "src.ui.services.secrets_helper.get_cloudsql_config",
        lambda: {"instance": None, "database": None, "user": None, "password": None},
    )
    monkeypatch.setattr(
        "src.ui.services.secrets_helper.get_cloud_storage_bucket",
        lambda: None,
        raising=False,
    )
    db_path = local_db_path
    _seed_stats_db(db_path)

    service = DataService(
        duckdb_path=db_path,
        config={"state_filter": "PA", "pos_threshold": 4.0},
        local_photos_dir=Path.cwd() / "missing_photos_dir",
    )
    try:
        yearly = service.get_review_volume_by_year(scope="global", state="PA")
        assert not yearly.empty
        assert set(yearly["year"].astype(int).tolist()) == {2020, 2021}

        ratings = service.get_rating_distribution(scope="global", state="CA")
        assert not ratings.empty
        assert ratings["review_count"].sum() == 2

        businesses = service.get_sample_business_rows(scope="training", limit=5)
        reviews = service.get_sample_review_rows(scope="training", limit=5)
        assert not businesses.empty
        assert not reviews.empty
        assert {"business_id", "name", "city", "state"}.issubset(set(businesses.columns))
        assert {"review_id", "user_id", "business_id", "stars", "text"}.issubset(
            set(reviews.columns)
        )
    finally:
        service.close()
        if db_path.exists():
            db_path.unlink()
