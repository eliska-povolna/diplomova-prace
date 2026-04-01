"""Data service for loading and serving POI metadata."""

from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
import pickle

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
    
    NOTE: Uses item2index mapping from training to ensure POI indices match
    the model's coordinate space (0 to n_items).
    """

    def __init__(
        self, duckdb_path: Path, parquet_dir: Path, config: Optional[Dict] = None,
        item2index_path: Optional[Path] = None,
        local_photos_dir: Optional[Path] = None
    ):
        """
        Initialize DataService.

        Args:
            duckdb_path: Path to yelp.duckdb
            parquet_dir: Path to parquet data directory
            config: Optional config dict
            item2index_path: Path to item2index.pkl mapping (business_id -> model index)
            local_photos_dir: Path to local photos directory (business_id.jpg format)
        """
        self.duckdb_path = Path(duckdb_path)
        self.parquet_dir = Path(parquet_dir)
        self.config = config or {}
        self.state_filter = self.config.get("state_filter")
        self.local_photos_dir = Path(local_photos_dir) if local_photos_dir else None

        # Load item2index mapping from training
        # This ensures POI indices stay in model coordinate space
        if item2index_path and Path(item2index_path).exists():
            with open(item2index_path, 'rb') as f:
                self.item2index = pickle.load(f)
            logger.info(f"✅ Loaded item2index mapping with {len(self.item2index)} items")
        else:
            self.item2index = None
            logger.warning("item2index mapping not found - POI indices may not match model")

        logger.info("Loading POI data...")
        self.conn = duckdb.connect(str(self.duckdb_path))
        self.pois_df = self._load_pois_dataframe()

        logger.info(
            f"✅ Loaded {len(self.pois_df)} POIs (state_filter={self.state_filter})"
        )

    def _load_pois_dataframe(self) -> pd.DataFrame:
        """Load POI metadata from Parquet into memory."""
        # Use same glob pattern as training: business/state=*/...
        # This ensures state is available as a column for filtering
        parquet_pattern = str(self.parquet_dir / "business" / "state=*" / "*.parquet")
        # DuckDB needs forward slashes (POSIX paths)
        parquet_pattern = parquet_pattern.replace("\\", "/")

        try:
            # Build query with state filter if provided
            if self.state_filter:
                # Use parameterized query to match training approach
                query = f"""
                    SELECT * FROM read_parquet('{parquet_pattern}')
                    WHERE state = '{self.state_filter}'
                """
                logger.debug(f"Loading POIs with filter: state = '{self.state_filter}'")
            else:
                query = f"SELECT * FROM read_parquet('{parquet_pattern}')"
                logger.debug("Loading all POIs (no state filter)")

            df = self.conn.execute(query).df()
            df = df.reset_index(drop=True)  # Ensure clean 0-based indexing
            logger.info(f"Loaded {len(df)} POIs from parquet")
            return df
        except Exception as e:
            logger.error(f"Failed to load Parquet data with pattern '{parquet_pattern}': {e}")
            return pd.DataFrame()

    def get_poi_details(self, poi_idx: int) -> Dict:
        """
        Get complete POI information by index.

        Includes real Yelp photos from dataset or local directory.

        Returns:
            Dict with keys:
                - poi_idx, business_id, name, category
                - lat, lon, rating, review_count, url
                - photos (list of URLs or local paths), primary_photo, photo_count
        """
        if poi_idx >= len(self.pois_df):
            logger.warning(f"POI index {poi_idx} out of range")
            return {}

        row = self.pois_df.iloc[poi_idx]
        business_id = str(row.get("business_id", ""))

        # Try to load local photo first
        photos = []
        if self.local_photos_dir:
            local_photo_path = self.local_photos_dir / f"{business_id}.jpg"
            if local_photo_path.exists():
                photos = [str(local_photo_path)]
                logger.debug(f"Loaded local photo for {business_id}")
        
        # If no local photo, parse from Yelp dataset
        if not photos:
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

        Filters to users with interactions in the current state_filter.

        Returns:
            List of dicts: [{'id': user_id, 'interactions': count}, ...]
        """
        try:
            review_pattern = str(self.parquet_dir / "review" / "year=*" / "*.parquet")
            business_pattern = str(self.parquet_dir / "business" / "state=*" / "*.parquet")
            # DuckDB needs forward slashes (POSIX paths)
            review_pattern = review_pattern.replace("\\", "/")
            business_pattern = business_pattern.replace("\\", "/")

            # Build query to count interactions with businesses matching state filter
            if self.state_filter:
                # Count interactions with businesses in this state only
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
                # No state filter - use all interactions
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

        Args:
            user_id: Yelp user ID
            min_stars: Minimum star rating threshold

        Returns:
            List of POI indices (in model coordinate space 0 to n_items)
        """
        try:
            review_pattern = str(self.parquet_dir / "review" / "year=*" / "*.parquet")
            # DuckDB needs forward slashes (POSIX paths)
            review_pattern = review_pattern.replace("\\", "/")

            # Get business IDs for this user with minimum rating
            business_ids = self.conn.execute(
                f"""
                SELECT DISTINCT business_id
                FROM read_parquet('{review_pattern}')
                WHERE user_id = '{user_id}' AND stars >= {min_stars}
            """
            ).df()

            # Map business IDs to POI indices using item2index mapping
            poi_indices = []
            if self.item2index:
                # Use trained model's item2index mapping
                for bid in business_ids["business_id"]:
                    if bid in self.item2index:
                        poi_indices.append(self.item2index[bid])
            else:
                # Fallback: map to pois_df index (will have wrong coordinate space)
                for bid in business_ids["business_id"]:
                    matches = self.pois_df[
                        self.pois_df["business_id"] == bid
                    ].index.tolist()
                    poi_indices.extend(matches)

            if poi_indices:
                max_idx = max(poi_indices)
                logger.debug(
                    f"User {user_id}: {len(poi_indices)} interactions, "
                    f"max_idx={max_idx}"
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
