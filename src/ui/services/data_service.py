"""Data service for loading and serving POI metadata."""

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

import pandas as pd
import duckdb

logger = logging.getLogger(__name__)


class DataService:
    """
    Load POI metadata from DuckDB + Parquet files.

    Provides:
    - POI details with photos
    - Test user filtering
    - Batch lookups
    """

    def __init__(
        self, duckdb_path: Path, parquet_dir: Path, config: Optional[Dict] = None
    ):
        """
        Initialize DataService.

        Args:
            duckdb_path: Path to yelp.duckdb
            parquet_dir: Path to parquet data directory
            config: Optional config dict
        """
        self.duckdb_path = Path(duckdb_path)
        self.parquet_dir = Path(parquet_dir)
        self.config = config or {}

        logger.info("Loading POI data...")
        self.conn = duckdb.connect(str(self.duckdb_path))
        self.pois_df = self._load_pois_dataframe()

        logger.info(f"✅ Loaded {len(self.pois_df)} POIs")

    def _load_pois_dataframe(self) -> pd.DataFrame:
        """Load POI metadata from Parquet into memory."""
        parquet_pattern = str(self.parquet_dir / "business" / "**" / "*.parquet")

        try:
            df = self.conn.execute(
                f"SELECT * FROM read_parquet('{parquet_pattern}')"
            ).df()
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet data: {e}")
            return pd.DataFrame()

    def get_poi_details(self, poi_idx: int) -> Dict:
        """
        Get complete POI information by index.

        Includes real Yelp photos from dataset.

        Returns:
            Dict with keys:
                - poi_idx, business_id, name, category
                - lat, lon, rating, review_count, url
                - photos (list of URLs), primary_photo, photo_count
        """
        if poi_idx >= len(self.pois_df):
            logger.warning(f"POI index {poi_idx} out of range")
            return {}

        row = self.pois_df.iloc[poi_idx]

        # Parse photos field (JSON list from Yelp dataset)
        photos = []
        photos_field = row.get("photos", "")
        if photos_field:
            try:
                if isinstance(photos_field, str):
                    photos = json.loads(photos_field)
                elif isinstance(photos_field, list):
                    photos = photos_field
            except (json.JSONDecodeError, TypeError) as e:
                logger.debug(f"Could not parse photos for POI {poi_idx}: {e}")

        return {
            "poi_idx": poi_idx,
            "business_id": str(row.get("business_id", "")),
            "name": str(row.get("name", "Unnamed")),
            "category": str(row.get("categories", "")),
            "lat": float(row.get("latitude", 0)),
            "lon": float(row.get("longitude", 0)),
            "rating": float(row.get("stars", 0)),
            "review_count": int(row.get("review_count", 0)),
            "url": f"https://www.yelp.com/biz/{row.get('business_id')}",
            # Photo information
            "photos": photos,
            "primary_photo": photos[0] if photos else None,
            "photo_count": len(photos),
        }

    def get_pois_batch(self, poi_indices: List[int]) -> List[Dict]:
        """
        Bulk lookup for multiple POIs.

        More efficient than repeated get_poi_details() calls.
        """
        return [self.get_poi_details(idx) for idx in poi_indices]

    def get_test_users(self, limit: int = 50) -> List[Dict]:
        """
        Get top N test users for dropdown selector.

        Returns:
            List of dicts: [{'id': user_id, 'interactions': count}, ...]
        """
        try:
            # Query review Parquet to find top users by interaction count
            review_pattern = str(self.parquet_dir / "review" / "**" / "*.parquet")

            users_df = self.conn.execute(
                f"""
                SELECT 
                    user_id,
                    COUNT(*) as interactions
                FROM read_parquet('{review_pattern}')
                WHERE stars >= 4.0
                GROUP BY user_id
                ORDER BY interactions DESC
                LIMIT {limit}
            """
            ).df()

            result = [
                {"id": row["user_id"], "interactions": int(row["interactions"])}
                for _, row in users_df.iterrows()
            ]

            logger.info(f"Found {len(result)} test users")
            return result

        except Exception as e:
            logger.error(f"Failed to load test users: {e}")
            return []

    def get_user_interactions(self, user_id: str, min_stars: float = 4.0) -> List[int]:
        """
        Get list of POI indices the user has interacted with.

        Args:
            user_id: Yelp user ID
            min_stars: Minimum star rating threshold

        Returns:
            List of POI indices
        """
        try:
            review_pattern = str(self.parquet_dir / "review" / "**" / "*.parquet")

            # Get business IDs for this user
            business_ids = self.conn.execute(
                f"""
                SELECT DISTINCT business_id
                FROM read_parquet('{review_pattern}')
                WHERE user_id = '{user_id}' AND stars >= {min_stars}
            """
            ).df()

            # Map business IDs to POI indices
            poi_indices = []
            for bid in business_ids["business_id"]:
                # Find this business in pois_df
                matches = self.pois_df[
                    self.pois_df["business_id"] == bid
                ].index.tolist()
                poi_indices.extend(matches)

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
