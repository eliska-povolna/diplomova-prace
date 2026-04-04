"""Data service for loading and serving POI metadata."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import pickle

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DataService:
    """
    Load POI metadata from DuckDB + Parquet files.

    Provides:
    - POI details with photos
    - Test user filtering
    - Batch lookups

    NOTE: Uses item2index mapping from training to ensure POI indices match
    the model's coordinate space (0 to n_items).
    """

    def __init__(
        self,
        duckdb_path: Path,
        parquet_dir: Path,
        config: Optional[Dict] = None,
        item2index_path: Optional[Path] = None,
        local_photos_dir: Optional[Path] = None,
    ):
        """
        Initialize DataService.

        Args:
            duckdb_path: Path to yelp.duckdb
            parquet_dir: Path to parquet data directory
            config: Optional config dict
            item2index_path: Path to item2index.pkl mapping (business_id -> model index)
            local_photos_dir: Path to local photos directory
        """
        self.duckdb_path = Path(duckdb_path)
        self.parquet_dir = Path(parquet_dir)
        self.config = config or {}
        self.state_filter = self.config.get("state_filter")
        self.local_photos_dir = Path(local_photos_dir) if local_photos_dir else None

        # Load item2index mapping from training.
        if item2index_path and Path(item2index_path).exists():
            with open(item2index_path, "rb") as f:
                self.item2index = pickle.load(f)
            logger.info(
                f"Loaded item2index mapping with {len(self.item2index)} items"
            )
        else:
            self.item2index = None
            logger.warning("item2index mapping not found; index alignment may be incorrect")

        # Reverse lookup: model idx -> business_id.
        self.index2item = (
            {idx: business_id for business_id, idx in self.item2index.items()}
            if self.item2index
            else {}
        )

        logger.info("Loading POI data...")
        self.conn = duckdb.connect(str(self.duckdb_path))
        self.pois_df = self._load_pois_dataframe()
        self.business_to_row_idx = self._build_business_index()
        self.local_photo_index = self._build_local_photo_index()

        logger.info(
            f"Loaded {len(self.pois_df)} POIs (state_filter={self.state_filter})"
        )

    def _load_pois_dataframe(self) -> pd.DataFrame:
        """Load POI metadata from Parquet into memory."""
        parquet_pattern = str(self.parquet_dir / "business" / "state=*" / "*.parquet")
        parquet_pattern = parquet_pattern.replace("\\", "/")

        try:
            if self.state_filter:
                query = f"""
                    SELECT * FROM read_parquet('{parquet_pattern}')
                    WHERE state = '{self.state_filter}'
                """
                logger.debug(f"Loading POIs with filter: state = '{self.state_filter}'")
            else:
                query = f"SELECT * FROM read_parquet('{parquet_pattern}')"
                logger.debug("Loading all POIs (no state filter)")

            df = self.conn.execute(query).df()
            df = df.reset_index(drop=True)
            logger.info(f"Loaded {len(df)} POIs from parquet")
            return df
        except Exception as e:
            logger.error(
                f"Failed to load Parquet data with pattern '{parquet_pattern}': {e}"
            )
            return pd.DataFrame()

    def _build_business_index(self) -> Dict[str, int]:
        """Create fast lookup map from business_id to row index in pois_df."""
        if self.pois_df.empty or "business_id" not in self.pois_df.columns:
            return {}

        return {
            str(business_id): int(row_idx)
            for row_idx, business_id in enumerate(self.pois_df["business_id"])
            if pd.notna(business_id)
        }

    def _build_local_photo_index(self) -> Dict[str, List[str]]:
        """
        Build business_id -> local photo paths index from Yelp photo dataset layout.

        Expected layout:
        - <local_photos_dir>/photos.json (JSONL with photo_id + business_id)
        - <local_photos_dir>/photos/*.jpg
        """
        if not self.local_photos_dir or not self.local_photos_dir.exists():
            return {}

        photos_json_path = self.local_photos_dir / "photos.json"
        photos_dir = self.local_photos_dir / "photos"

        if not photos_json_path.exists() or not photos_dir.exists():
            return {}

        photo_index: Dict[str, List[str]] = {}
        target_business_ids = set(self.business_to_row_idx.keys())

        try:
            with open(photos_json_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        photo_record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    business_id = str(photo_record.get("business_id", ""))
                    photo_id = str(photo_record.get("photo_id", ""))

                    if not business_id or not photo_id:
                        continue
                    if business_id not in target_business_ids:
                        continue

                    photo_path = photos_dir / f"{photo_id}.jpg"
                    if not photo_path.exists():
                        continue

                    photo_index.setdefault(business_id, []).append(str(photo_path))
        except OSError as e:
            logger.warning(
                f"Failed to build local photo index from {photos_json_path}: {e}"
            )
            return {}

        if photo_index:
            logger.info(
                f"Indexed local photos for {len(photo_index)} businesses from {photos_json_path}"
            )
        return photo_index

    def _resolve_business_and_row(
        self, poi_idx: int
    ) -> Tuple[Optional[str], Optional[pd.Series]]:
        """Resolve model-space POI index to business_id and POI row."""
        if poi_idx < 0:
            return None, None

        if self.index2item:
            business_id = self.index2item.get(poi_idx)
            if business_id is None:
                return None, None

            row_idx = self.business_to_row_idx.get(business_id)
            if row_idx is None:
                return business_id, None

            return business_id, self.pois_df.iloc[row_idx]

        if poi_idx >= len(self.pois_df):
            return None, None

        row = self.pois_df.iloc[poi_idx]
        return str(row.get("business_id", "")), row

    def _get_local_photos_for_business(self, business_id: str) -> List[str]:
        """Return local photo paths for a business if available."""
        if not business_id:
            return []

        indexed_paths = self.local_photo_index.get(business_id, [])
        if indexed_paths:
            return indexed_paths

        # Backward-compatible fallback for legacy <business_id>.<ext> naming.
        if not self.local_photos_dir:
            return []

        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            legacy_path = self.local_photos_dir / f"{business_id}{ext}"
            if legacy_path.exists():
                return [str(legacy_path)]

        return []

    def _parse_dataset_photos(self, photos_field) -> List[str]:
        """Parse photos field from dataset row into a list of URLs."""
        if not photos_field:
            return []

        if isinstance(photos_field, list):
            return [str(photo) for photo in photos_field if photo]

        if isinstance(photos_field, str):
            photos_text = photos_field.strip()
            if not photos_text:
                return []

            if photos_text.startswith("["):
                try:
                    parsed = json.loads(photos_text)
                except json.JSONDecodeError:
                    return []
                if isinstance(parsed, list):
                    return [str(photo) for photo in parsed if photo]

            if photos_text.startswith("http://") or photos_text.startswith("https://"):
                return [photos_text]

        return []

    def get_poi_details(self, poi_idx: int) -> Dict:
        """
        Get complete POI information by model-space index.

        Includes local photos (if available) or Yelp URL photos from dataset.
        """
        business_id, row = self._resolve_business_and_row(poi_idx)
        if row is None:
            logger.warning(f"POI index {poi_idx} could not be resolved to POI metadata")
            return {}

        photos = self._get_local_photos_for_business(business_id or "")
        if not photos:
            photos = self._parse_dataset_photos(row.get("photos", ""))

        resolved_business_id = str(row.get("business_id", business_id or ""))

        return {
            "poi_idx": poi_idx,
            "business_id": resolved_business_id,
            "name": str(row.get("name", "Unnamed")),
            "category": str(row.get("categories", "")),
            "lat": float(row.get("latitude", 0)),
            "lon": float(row.get("longitude", 0)),
            "rating": float(row.get("stars", 0)),
            "review_count": int(row.get("review_count", 0)),
            "url": f"https://www.yelp.com/biz/{resolved_business_id}",
            "photos": photos,
            "primary_photo": photos[0] if photos else None,
            "photo_count": len(photos),
        }

    def get_pois_batch(self, poi_indices: List[int]) -> List[Dict]:
        """Bulk lookup for multiple POIs."""
        return [self.get_poi_details(idx) for idx in poi_indices]

    def get_test_users(self, limit: int = 50) -> List[Dict]:
        """
        Get top N test users for dropdown selector.

        Filters to users with interactions in the current state_filter.
        """
        try:
            review_pattern = str(self.parquet_dir / "review" / "year=*" / "*.parquet")
            business_pattern = str(
                self.parquet_dir / "business" / "state=*" / "*.parquet"
            )
            review_pattern = review_pattern.replace("\\", "/")
            business_pattern = business_pattern.replace("\\", "/")

            if self.state_filter:
                query = f"""
                    SELECT
                        reviews.user_id,
                        COUNT(*) as interactions
                    FROM read_parquet('{review_pattern}') AS reviews
                    INNER JOIN read_parquet('{business_pattern}') AS business
                        ON reviews.business_id = business.business_id
                    WHERE reviews.stars >= 4.0 AND business.state = '{self.state_filter}'
                    GROUP BY reviews.user_id
                    ORDER BY interactions DESC
                    LIMIT {limit}
                """
            else:
                query = f"""
                    SELECT
                        user_id,
                        COUNT(*) as interactions
                    FROM read_parquet('{review_pattern}')
                    WHERE stars >= 4.0
                    GROUP BY user_id
                    ORDER BY interactions DESC
                    LIMIT {limit}
                """

            users_df = self.conn.execute(query).df()
            result = [
                {"id": row["user_id"], "interactions": int(row["interactions"])}
                for _, row in users_df.iterrows()
            ]

            logger.info(
                f"Found {len(result)} test users with state_filter={self.state_filter}"
            )
            return result

        except Exception as e:
            logger.error(f"Failed to load test users: {e}")
            return []

    def get_user_interactions(self, user_id: str, min_stars: float = 4.0) -> List[int]:
        """
        Get list of POI indices the user has interacted with.

        Uses item2index mapping to ensure indices match model coordinate space.
        """
        try:
            review_pattern = str(self.parquet_dir / "review" / "year=*" / "*.parquet")
            review_pattern = review_pattern.replace("\\", "/")

            business_ids = self.conn.execute(
                f"""
                SELECT DISTINCT business_id
                FROM read_parquet('{review_pattern}')
                WHERE user_id = '{user_id}' AND stars >= {min_stars}
                """
            ).df()

            poi_indices: List[int] = []
            if self.item2index:
                for bid in business_ids["business_id"]:
                    if bid in self.item2index:
                        poi_indices.append(self.item2index[bid])
            else:
                for bid in business_ids["business_id"]:
                    matches = self.pois_df[
                        self.pois_df["business_id"] == bid
                    ].index.tolist()
                    poi_indices.extend(matches)

            if poi_indices:
                max_idx = max(poi_indices)
                logger.debug(
                    f"User {user_id}: {len(poi_indices)} interactions, max_idx={max_idx}"
                )

            return poi_indices

        except Exception as e:
            logger.debug(f"Failed to get interactions for user {user_id}: {e}")
            return []

    @property
    def num_pois(self) -> int:
        """Total number of POIs."""
        return len(self.pois_df)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
