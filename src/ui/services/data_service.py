"""Data service for loading and serving POI metadata."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import pickle
from dataclasses import dataclass

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PhotoMetadata:
    """Photo metadata extracted from photos.json."""
    photo_id: str
    business_id: str
    path: str
    label: str = "other"  # e.g., 'outside', 'inside', 'food', 'drink', 'menu'
    caption: str = ""

    @staticmethod
    def label_priority(label: str) -> int:
        """Return priority score for photo label (higher = better)."""
        priorities = {
            "outside": 5,
            "inside": 4,
            "food": 3,
            "drink": 2,
            "menu": 1,
            "other": 0,
        }
        return priorities.get(label.lower(), 0)


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

        logger.info("Initializing data service (lazy loading)...")
        self.conn = duckdb.connect(str(self.duckdb_path))
        
        # DO NOT load all POIs into memory - use lazy loading instead
        self.pois_df = None  # Load on-demand only
        self.business_to_row_idx = {}  # Will build from queries
        self.local_photo_index = self._build_local_photo_index()
        
        # Cache business IDs from item2index for quick validation
        self._valid_business_ids = set(self.item2index.keys()) if self.item2index else None

        logger.info(f"✅ Data service ready (lazy loading enabled, state_filter={self.state_filter})")

    def _load_pois_dataframe(self) -> pd.DataFrame:
        """DEPRECATED: Use lazy loading instead. This is kept for compatibility only."""
        logger.warning("_load_pois_dataframe is deprecated - use lazy loading instead")
        return pd.DataFrame()
    
    def _get_poi_from_duckdb(self, business_id: str) -> Dict | None:
        """Query a single POI from DuckDB by business_id (lazy loading)."""
        try:
            parquet_pattern = str(self.parquet_dir / "business" / "state=*" / "*.parquet")
            parquet_pattern = parquet_pattern.replace("\\", "/")
            
            where_clause = f"WHERE business_id = '{business_id}'"
            if self.state_filter:
                where_clause += f" AND state = '{self.state_filter}'"
            
            query = f"SELECT * FROM read_parquet('{parquet_pattern}') {where_clause} LIMIT 1"
            
            result = self.conn.execute(query).df()
            if len(result) == 0:
                return None
            
            return result.iloc[0].to_dict()
        except Exception as e:
            logger.debug(f"Failed to load POI {business_id} from DuckDB: {e}")
            return None

    def _build_business_index(self) -> Dict[str, int]:
        """DEPRECATED: No longer needed with lazy loading."""
        return {}

    def _build_local_photo_index(self) -> Dict[str, List[PhotoMetadata]]:
        """
        Build business_id -> sorted photo metadata index from Yelp photo dataset layout.

        Expected layout:
        - <local_photos_dir>/photos.json (JSONL with photo_id, business_id, label, caption)
        - <local_photos_dir>/photos/*.jpg

        Photos are sorted by label priority (outside > inside > food > drink > menu > other).
        NOTE: In lazy loading mode, we index ALL photos. Filtering by state happens when photos are requested.
        NOTE: We trust that all entries in photos.json have corresponding .jpg files (skip slow filesystem check).
        """
        if not self.local_photos_dir or not self.local_photos_dir.exists():
            return {}

        photos_json_path = self.local_photos_dir / "photos.json"
        photos_dir = self.local_photos_dir / "photos"

        if not photos_json_path.exists():
            return {}

        photo_index: Dict[str, List[PhotoMetadata]] = {}

        try:
            logger.info(f"Building photo index from {photos_json_path}...")
            loaded_count = 0
            
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

                    # Trust that photos exist; skip slow filesystem check
                    photo_path = photos_dir / f"{photo_id}.jpg"

                    # Extract label and caption (new fields)
                    label = str(photo_record.get("label", "other"))
                    caption = str(photo_record.get("caption", ""))

                    photo_meta = PhotoMetadata(
                        photo_id=photo_id,
                        business_id=business_id,
                        path=str(photo_path),
                        label=label,
                        caption=caption,
                    )

                    photo_index.setdefault(business_id, []).append(photo_meta)
                    loaded_count += 1
            
            logger.info(f"Indexed {loaded_count} photos for {len(photo_index)} businesses")
        except OSError as e:
            logger.warning(
                f"Failed to build local photo index from {photos_json_path}: {e}"
            )
            return {}

        # Sort photos by label priority (descending) for each business
        for business_id in photo_index:
            photo_index[business_id].sort(
                key=lambda p: PhotoMetadata.label_priority(p.label), reverse=True
            )

        if photo_index:
            total_photos = sum(len(photos) for photos in photo_index.values())
            logger.info(
                f"Indexed {total_photos} local photos for {len(photo_index)} businesses from {photos_json_path}"
            )
            # Log label distribution for validation
            label_dist = {}
            for photos in photo_index.values():
                for photo in photos:
                    label_dist[photo.label] = label_dist.get(photo.label, 0) + 1
            logger.info(f"Photo label distribution: {label_dist}")

        return photo_index

    def _resolve_business_and_row(
        self, poi_idx: int
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Resolve model-space POI index to business_id and POI data (lazy loaded from DuckDB)."""
        if poi_idx < 0:
            return None, None

        if self.index2item:
            business_id = self.index2item.get(poi_idx)
            if business_id is None:
                logger.debug(f"POI index {poi_idx} not in item2index mapping")
                return None, None

            # Lazy load POI data from DuckDB (not from memory)
            row_data = self._get_poi_from_duckdb(business_id)
            if row_data is None:
                logger.debug(f"POI index {poi_idx} (business_id={business_id}) not found in filtered POI data")
                return None, None

            return business_id, row_data
        
        return None, None

    def _get_local_photos_for_business(self, business_id: str) -> List[str]:
        """Return local photo paths for a business if available, sorted by label priority."""
        if not business_id:
            return []

        indexed_photos = self.local_photo_index.get(business_id, [])
        if indexed_photos:
            # Extract paths from sorted PhotoMetadata objects
            return [photo.path for photo in indexed_photos]

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
        
        Returns empty dict if POI is not found or invalid (can happen with state filtering).
        Validates all required fields to skip corrupted/incomplete POI data.
        """
        business_id, row = self._resolve_business_and_row(poi_idx)
        if row is None:
            # Expected when POI is outside filtered region or doesn't exist in map
            logger.debug(f"POI index {poi_idx} not available (outside filtered data)")
            return {}

        # Validate required fields to skip invalid POI data
        try:
            name = str(row.get("name", "")).strip()
            if not name or name.lower() == "unnamed":
                logger.debug(f"POI index {poi_idx} has no valid name, skipping")
                return {}
                
            # Validate coordinates are reasonable
            lat = float(row.get("latitude", 0))
            lon = float(row.get("longitude", 0))
            if lat == 0 and lon == 0:
                logger.debug(f"POI index {poi_idx} ({name}) has invalid coordinates (0,0), skipping")
                return {}
            
            # Validate basic numeric fields
            rating = float(row.get("stars", 0))
            if rating < 0 or rating > 5:
                logger.debug(f"POI index {poi_idx} ({name}) has invalid rating {rating}, skipping")
                return {}
                
        except (ValueError, TypeError) as e:
            logger.debug(f"POI index {poi_idx} has invalid data: {e}, skipping")
            return {}

        # Resolve business_id early for use in logging
        resolved_business_id = str(row.get("business_id", business_id or ""))
        if not resolved_business_id:
            logger.debug(f"POI index {poi_idx} has no business_id, skipping")
            return {}

        photos = self._get_local_photos_for_business(business_id or "")
        if not photos:
            photos = self._parse_dataset_photos(row.get("photos", ""))
            if photos:
                logger.debug(f"POI {resolved_business_id}: Loaded {len(photos)} dataset photos")
            else:
                logger.debug(f"POI {resolved_business_id}: No photos found (photos field: {row.get('photos', 'N/A')[:100]})")
        else:
            logger.debug(f"POI {resolved_business_id}: Loaded {len(photos)} local photos")

        return {
            "poi_idx": poi_idx,
            "business_id": resolved_business_id,
            "name": name,
            "category": str(row.get("categories", "")),
            "lat": lat,
            "lon": lon,
            "rating": rating,
            "review_count": int(row.get("review_count", 0)),
            "url": f"https://www.yelp.com/biz/{resolved_business_id}",
            "photos": photos,
            "primary_photo": photos[0] if photos else None,
            "photo_count": len(photos),
        }

    def get_pois_batch(self, poi_indices: List[int]) -> List[Dict]:
        """Bulk lookup for multiple POIs. Skips POIs that can't be resolved."""
        pois = [self.get_poi_details(idx) for idx in poi_indices]
        # Filter out empty dicts (POIs that couldn't be resolved)
        return [p for p in pois if p]

    def get_test_users(self, limit: int = 50) -> List[Dict]:
        """
        Get top N test users for dropdown selector.

        Filters to users with interactions in the current state_filter.
        Loads from precomputed cache if available for fast startup.
        """
        # Try loading from precomputed cache first
        project_root = Path(__file__).parent.parent.parent
        cache_dir = project_root / "data" / "ui_cache"
        
        if cache_dir.exists():
            state_suffix = f"_{self.state_filter}" if self.state_filter else "_all"
            cache_file = cache_dir / f"test_users{state_suffix}.pkl"
            
            if cache_file.exists():
                try:
                    with open(cache_file, "rb") as f:
                        users_list = pickle.load(f)
                    logger.info(
                        f"✅ Loaded {len(users_list)} test users from cache: {cache_file}"
                    )
                    return users_list[:limit]  # Respect limit parameter
                except Exception as e:
                    logger.warning(f"Failed to load user cache from {cache_file}: {e}")
                    # Fall through to compute dynamically
        
        # Fallback: compute dynamically (slower, but works without precomputation)
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
        Filters to interactions in the same state/region as the loaded POI data.
        """
        try:
            review_pattern = str(self.parquet_dir / "review" / "year=*" / "*.parquet")
            business_pattern = str(self.parquet_dir / "business" / "state=*" / "*.parquet")
            review_pattern = review_pattern.replace("\\", "/")
            business_pattern = business_pattern.replace("\\", "/")

            # Build WHERE clause to match the same data filtering as POI loading
            where_clause = f"WHERE reviews.user_id = '{user_id}' AND reviews.stars >= {min_stars}"
            if self.state_filter:
                where_clause += f" AND business.state = '{self.state_filter}'"
            
            query = f"""
                SELECT DISTINCT business.business_id
                FROM read_parquet('{review_pattern}') reviews
                JOIN read_parquet('{business_pattern}') business
                ON reviews.business_id = business.business_id
                {where_clause}
            """
            business_ids = self.conn.execute(query).df()

            logger.debug(
                f"Query for user {user_id}: found {len(business_ids)} businesses with stars >= {min_stars}"
            )

            poi_indices: List[int] = []
            if self.item2index:
                unmapped = 0
                for bid in business_ids["business_id"]:
                    if bid in self.item2index:
                        poi_indices.append(self.item2index[bid])
                    else:
                        unmapped += 1
                
                if unmapped > 0:
                    logger.debug(
                        f"User {user_id}: {unmapped} businesses not in item2index mapping (total mapping size: {len(self.item2index)})"
                    )
            else:
                logger.warning("item2index is None, cannot map user interactions to model indices")

            if poi_indices:
                max_idx = max(poi_indices)
                logger.info(
                    f"User {user_id}: {len(poi_indices)} valid interactions, max_idx={max_idx}"
                )
            else:
                logger.warning(
                    f"User {user_id}: No valid interactions found after mapping"
                )

            return poi_indices

        except Exception as e:
            logger.error(f"Failed to get interactions for user {user_id}: {e}", exc_info=True)
            return []

    @property
    def num_pois(self) -> int:
        """Total number of POIs (queried from DuckDB, not loaded in memory)."""
        try:
            parquet_pattern = str(self.parquet_dir / "business" / "state=*" / "*.parquet")
            parquet_pattern = parquet_pattern.replace("\\", "/")
            
            if self.state_filter:
                query = f"SELECT COUNT(*) as cnt FROM read_parquet('{parquet_pattern}') WHERE state = '{self.state_filter}'"
            else:
                query = f"SELECT COUNT(*) as cnt FROM read_parquet('{parquet_pattern}')"
            
            result = self.conn.execute(query).df()
            return int(result["cnt"][0]) if len(result) > 0 else 0
        except Exception as e:
            logger.warning(f"Failed to count POIs: {e}")
            return len(self.item2index) if self.item2index else 0

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
