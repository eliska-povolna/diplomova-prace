"""Regression tests for Streamlit DataService POI/photo resolution."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from src.ui.services.data_service import DataService


def _write_business_parquet(parquet_root: Path) -> None:
    business_dir = parquet_root / "business" / "state=CA"
    business_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately keep row order different from item2index mapping order.
    df = pd.DataFrame(
        [
            {
                "business_id": "biz_row0",
                "name": "Row Zero Cafe",
                "categories": "Cafe",
                "latitude": 1.0,
                "longitude": 2.0,
                "stars": 4.5,
                "review_count": 10,
                "state": "CA",
                "photos": json.dumps(["https://example.com/row0.jpg"]),
            },
            {
                "business_id": "biz_row1",
                "name": "Row One Pizza",
                "categories": "Pizza",
                "latitude": 3.0,
                "longitude": 4.0,
                "stars": 4.0,
                "review_count": 20,
                "state": "CA",
                "photos": json.dumps(["https://example.com/row1.jpg"]),
            },
        ]
    )
    df.to_parquet(business_dir / "part-0.parquet", index=False)


def _write_item2index(path: Path) -> None:
    # Model index 0 points to biz_row1 (not row0).
    mapping = {"biz_row1": 0, "biz_row0": 1}
    with open(path, "wb") as f:
        pickle.dump(mapping, f)


def _write_local_photos(photo_root: Path) -> None:
    photos_dir = photo_root / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    photo_id = "photo_for_biz_row1"
    (photos_dir / f"{photo_id}.jpg").write_bytes(b"fake-jpg")

    with open(photo_root / "photos.json", "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "photo_id": photo_id,
                    "business_id": "biz_row1",
                    "caption": "",
                    "label": "inside",
                }
            )
            + "\n"
        )


def test_get_poi_details_uses_model_index_mapping_and_local_photos(tmp_path: Path) -> None:
    parquet_root = tmp_path / "parquet"
    _write_business_parquet(parquet_root)

    item2index_path = tmp_path / "item2index.pkl"
    _write_item2index(item2index_path)

    photo_root = tmp_path / "yelp_photos"
    _write_local_photos(photo_root)

    db_path = tmp_path / "test.duckdb"
    service = DataService(
        duckdb_path=db_path,
        parquet_dir=parquet_root,
        config={"state_filter": "CA"},
        item2index_path=item2index_path,
        local_photos_dir=photo_root,
    )
    try:
        poi = service.get_poi_details(0)
        assert poi["business_id"] == "biz_row1"
        assert poi["name"] == "Row One Pizza"
        assert poi["photo_count"] == 1
        assert "photo_for_biz_row1.jpg" in poi["primary_photo"]
    finally:
        service.close()


def test_get_poi_details_falls_back_to_dataset_photo_urls(tmp_path: Path) -> None:
    parquet_root = tmp_path / "parquet"
    _write_business_parquet(parquet_root)

    item2index_path = tmp_path / "item2index.pkl"
    _write_item2index(item2index_path)

    db_path = tmp_path / "test.duckdb"
    service = DataService(
        duckdb_path=db_path,
        parquet_dir=parquet_root,
        config={"state_filter": "CA"},
        item2index_path=item2index_path,
        local_photos_dir=tmp_path / "missing_photos",
    )
    try:
        poi = service.get_poi_details(1)
        assert poi["business_id"] == "biz_row0"
        assert poi["photo_count"] == 1
        assert poi["primary_photo"] == "https://example.com/row0.jpg"
    finally:
        service.close()


def test_photos_sorted_by_label_priority(tmp_path: Path) -> None:
    """Verify photos are prioritized by label (outside > inside > food > drink > menu)."""
    parquet_root = tmp_path / "parquet"
    _write_business_parquet(parquet_root)

    item2index_path = tmp_path / "item2index.pkl"
    _write_item2index(item2index_path)

    photo_root = tmp_path / "yelp_photos"
    photos_dir = photo_root / "photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    # Create photos with different labels (intentionally out of priority order)
    photos_data = [
        {"photo_id": "photo_food", "business_id": "biz_row1", "label": "food", "caption": ""},
        {"photo_id": "photo_outside", "business_id": "biz_row1", "label": "outside", "caption": "storefront"},
        {"photo_id": "photo_menu", "business_id": "biz_row1", "label": "menu", "caption": ""},
        {"photo_id": "photo_inside", "business_id": "biz_row1", "label": "inside", "caption": "interior"},
    ]

    # Write photo files and JSON metadata
    with open(photo_root / "photos.json", "w", encoding="utf-8") as f:
        for photo in photos_data:
            (photos_dir / f"{photo['photo_id']}.jpg").write_bytes(b"fake-jpg")
            f.write(json.dumps(photo) + "\n")

    db_path = tmp_path / "test.duckdb"
    service = DataService(
        duckdb_path=db_path,
        parquet_dir=parquet_root,
        config={"state_filter": "CA"},
        item2index_path=item2index_path,
        local_photos_dir=photo_root,
    )
    try:
        poi = service.get_poi_details(0)
        
        # Verify we have all 4 photos
        assert poi["photo_count"] == 4, f"Expected 4 photos, got {poi['photo_count']}"
        
        # Verify they're sorted by priority: outside > inside > food > menu
        photos = poi["photos"]
        assert "photo_outside.jpg" in photos[0], f"First photo should be 'outside', got {photos[0]}"
        assert "photo_inside.jpg" in photos[1], f"Second photo should be 'inside', got {photos[1]}"
        assert "photo_food.jpg" in photos[2], f"Third photo should be 'food', got {photos[2]}"
        assert "photo_menu.jpg" in photos[3], f"Fourth photo should be 'menu', got {photos[3]}"
        
        # Verify primary photo is the highest priority
        assert "photo_outside.jpg" in poi["primary_photo"]
        
    finally:
        service.close()
