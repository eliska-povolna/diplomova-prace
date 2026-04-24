"""Data service for loading and serving POI metadata."""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd
from src.data.run_artifacts import load_shared_preprocessing_payload_for_run

# Conditional Streamlit import for caching
try:
    from streamlit import cache_data

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

    # Dummy decorator for non-Streamlit contexts
    def cache_data(func):
        return func


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
    Load POI metadata from DuckDB or CloudSQL.

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
        config: Optional[Dict] = None,
        item2index_path: Optional[Path] = None,
        local_photos_dir: Optional[Path] = None,
        active_run_dir: Optional[Path] = None,
    ):
        """
        Initialize DataService.

        Tries Cloud SQL first (if configured), falls back to local DuckDB.

        Args:
            duckdb_path: Path to yelp.duckdb
            config: Optional config dict
            item2index_path: Path to item2index.pkl mapping (business_id -> model index)
            local_photos_dir: Path to local photos directory
        """
        import os

        self.duckdb_path = Path(duckdb_path)
        self.config = config or {}
        self.state_filter = self.config.get("state_filter")
        self.local_photos_dir = Path(local_photos_dir) if local_photos_dir else None
        self.active_run_dir = Path(active_run_dir) if active_run_dir else None
        self.backend_type = "unknown"
        self.engine = None  # For Cloud SQL

        # Load item2index mapping from training.
        # This maps business_id -> model index for coordinate space alignment
        # It's optional; without it, the app still works but may show indices differently
        self.item2index = None
        if item2index_path and Path(item2index_path).exists():
            try:
                with open(item2index_path, "rb") as f:
                    self.item2index = pickle.load(f)
                logger.info(
                    f"✓ Loaded item2index mapping with {len(self.item2index)} items"
                )
            except Exception as e:
                logger.debug(f"Could not load item2index from {item2index_path}: {e}")
        else:
            logger.debug(
                "item2index mapping not found (optional); "
                "app will function but without strict index alignment. "
                "This is expected on Streamlit Cloud."
            )

        # Reverse lookup: model idx -> business_id.
        self.index2item = (
            {idx: business_id for business_id, idx in self.item2index.items()}
            if self.item2index
            else {}
        )

        # Try Cloud SQL first, fall back to local DuckDB
        logger.info("Initializing data service...")
        self.conn = None

        # Check for Cloud SQL credentials
        from .secrets_helper import get_cloudsql_config

        config = get_cloudsql_config()
        cloudsql_instance = config["instance"]
        cloudsql_db = config["database"]
        cloudsql_user = config["user"]
        cloudsql_password = config["password"]

        logger.debug(f"Cloud SQL config check:")
        logger.debug(f"  instance: {cloudsql_instance}")
        logger.debug(f"  database: {cloudsql_db}")
        logger.debug(f"  user: {cloudsql_user}")
        logger.debug(
            f"  password: {'*' * len(cloudsql_password) if cloudsql_password else 'NOT SET'}"
        )
        logger.debug(f"  credentials_path: {config.get('credentials_path')}")

        if all([cloudsql_instance, cloudsql_db, cloudsql_user, cloudsql_password]):
            try:
                logger.info(
                    f"🔗 Cloud SQL credentials found: instance={cloudsql_instance}"
                )
                logger.info("Attempting Cloud SQL connection...")
                from .cloud_sql_helper import CloudSQLHelper

                # Use CloudSQLHelper for connection management (no code duplication)
                sql_helper = CloudSQLHelper(
                    instance_connection_name=cloudsql_instance,
                    database=cloudsql_db,
                    user=cloudsql_user,
                    password=cloudsql_password,
                )
                self.engine = sql_helper.engine
                self.backend_type = "cloudsql"
                logger.info("✅ Using Cloud SQL as backend")

            except Exception as e:
                logger.error(f"❌ Cloud SQL connection FAILED: {type(e).__name__}: {e}")
                logger.error("Falling back to local DuckDB...")
                self.engine = None
                self.conn = duckdb.connect(str(self.duckdb_path))
                self.backend_type = "duckdb"
        else:
            # Use local DuckDB
            missing = []
            if not cloudsql_instance:
                missing.append("CLOUDSQL_INSTANCE")
            if not cloudsql_db:
                missing.append("CLOUDSQL_DATABASE")
            if not cloudsql_user:
                missing.append("CLOUDSQL_USER")
            if not cloudsql_password:
                missing.append("CLOUDSQL_PASSWORD")
            logger.warning(
                f"⚠️ Missing Cloud SQL config ({', '.join(missing)}), using local DuckDB"
            )
            self.conn = duckdb.connect(str(self.duckdb_path))
            self.backend_type = "duckdb"

        logger.info(
            f"✓ Data service ready (backend: {self.backend_type}, state_filter={self.state_filter})"
        )

        # DO NOT load all POIs into memory - use lazy loading instead
        self.pois_df = None  # Load on-demand only
        self.business_to_row_idx = {}  # Will build from queries

        # Initialize Cloud Storage helper for primary photo source
        self.cloud_storage_helper = None
        try:
            from .secrets_helper import get_cloud_storage_bucket

            bucket_name = get_cloud_storage_bucket()
            if bucket_name:
                try:
                    from .cloud_storage_helper import CloudStorageHelper

                    self.cloud_storage_helper = CloudStorageHelper(bucket_name)
                    logger.info(
                        f"✓ Cloud Storage helper initialized (bucket: {bucket_name})"
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not initialize Cloud Storage helper: {e} "
                        "(photos from Cloud Storage will not be available)"
                    )
        except Exception as e:
            logger.debug(f"Cloud Storage config not available: {e}")

        # Load photo index (try cloud precomputed → local pickle → build from scratch)
        self._cloud_photo_index = None
        self.local_photo_index = {}

        if self.cloud_storage_helper:
            logger.info("📷 Checking for precomputed photo index in Cloud Storage...")
            self._cloud_photo_index = self._load_precomputed_cloud_photo_index()
            if self._cloud_photo_index:
                logger.info(
                    f"✅ Loaded precomputed photo index from Cloud Storage ({len(self._cloud_photo_index)} businesses)"
                )
            else:
                logger.debug("No precomputed photo index in Cloud Storage")
                # Try local pickle fallback
                logger.debug("Checking for local pickle precompute...")
                self.local_photo_index = self._load_precomputed_local_photo_index()
                if not self.local_photo_index:
                    logger.debug("No local pickle, building from photos.json...")
                    self.local_photo_index = self._build_local_photo_index()
        else:
            logger.info(
                "Cloud Storage not available, checking local precomputed index..."
            )
            self.local_photo_index = self._load_precomputed_local_photo_index()
            if not self.local_photo_index:
                logger.debug("No local precompute, building from photos.json...")
                self.local_photo_index = self._build_local_photo_index()

        # Cache business IDs from item2index for quick validation
        self._valid_business_ids = (
            set(self.item2index.keys()) if self.item2index else None
        )

    def _load_pois_dataframe(self) -> pd.DataFrame:
        """DEPRECATED: Use lazy loading instead. This is kept for compatibility only."""
        logger.warning("_load_pois_dataframe is deprecated - use lazy loading instead")
        return pd.DataFrame()

    def _get_poi_from_duckdb(self, business_id: str) -> Dict | None:
        """Query a single POI by business_id (supports Cloud SQL or DuckDB backend)."""
        from sqlalchemy import text

        try:
            if self.backend_type == "cloudsql":
                # Query from Cloud SQL - using parameterized queries
                # No state filter (item2index comes from full training data)
                query = (
                    "SELECT * FROM businesses WHERE business_id = :business_id LIMIT 1"
                )
                params = {"business_id": business_id}

                with self.engine.connect() as conn:
                    result = pd.read_sql(text(query), conn, params=params)
                    if len(result) > 0:
                        return result.iloc[0].to_dict()
                    return None

            else:
                # Query from DuckDB (local) - use yelp_business table
                query = "SELECT * FROM yelp_business WHERE business_id = ? LIMIT 1"
                result = self.conn.execute(query, [business_id]).df()

                if len(result) > 0:
                    return result.iloc[0].to_dict()
                return None

        except Exception as e:
            logger.warning(f"Failed to load POI {business_id}: {e}")
            return None

    def _build_business_index(self) -> Dict[str, int]:
        """DEPRECATED: No longer needed with lazy loading."""
        return {}

    def _load_precomputed_local_photo_index(self) -> Dict[str, List[PhotoMetadata]]:
        """
        Load precomputed photo index from local pickle file.

        Expected location: data/photo_index.pkl (generated by scripts/precompute_photo_index.py)
        Returns empty dict if file not found.
        """
        import pickle
        from pathlib import Path

        pickle_path = (
            Path(__file__).parent.parent.parent.parent / "data" / "photo_index.pkl"
        )

        if not pickle_path.exists():
            logger.debug(f"Precomputed photo index not found at {pickle_path}")
            return {}

        try:
            logger.info(f"Loading precomputed photo index from {pickle_path}...")
            with open(pickle_path, "rb") as f:
                photo_index = pickle.load(f)

            total_photos = sum(len(photos) for photos in photo_index.values())
            logger.info(
                f"✅ Loaded precomputed photo index ({total_photos} photos for {len(photo_index)} businesses)"
            )
            return photo_index

        except Exception as e:
            logger.warning(f"Failed to load precomputed photo index: {e}")
            return {}

    def _load_precomputed_cloud_photo_index(self) -> Dict[str, List[PhotoMetadata]]:
        """
        Load precomputed photo index from Cloud Storage.

        Tries in order:
        1. metadata/photo_index.pkl (binary, fastest)
        2. metadata/photo_index.json (JSON, portable but larger)

        Returns empty dict if not found or error occurs.
        """
        if not self.cloud_storage_helper:
            return {}

        # Try 1: Load pickle (fast, direct unpickling)
        try:
            logger.debug("Checking Cloud Storage for precomputed photo_index.pkl...")
            photo_index = self.cloud_storage_helper.read_pickle(
                "metadata/photo_index.pkl"
            )
            if photo_index:
                total_photos = sum(len(photos) for photos in photo_index.values())
                logger.info(
                    f"✅ Loaded precomputed photo index from cloud pickle ({total_photos} photos for {len(photo_index)} businesses)"
                )
                return photo_index
        except Exception as e:
            logger.debug(f"Could not load pickle from cloud: {e}")

        # Try 2: Load JSON (fallback to JSON version if pickle not available)
        try:
            logger.debug("Checking Cloud Storage for precomputed photo_index.json...")
            photos_data = self.cloud_storage_helper.read_json(
                "metadata/photo_index.json"
            )
            if not photos_data:
                return {}

            # Reconstruct PhotoMetadata objects from JSON
            photo_index: Dict[str, List[PhotoMetadata]] = {}
            for business_id, photos_list in photos_data.items():
                photo_index[business_id] = [
                    PhotoMetadata(
                        photo_id=p["photo_id"],
                        business_id=p["business_id"],
                        path=f"gs://{self.cloud_storage_helper.bucket_name}/photos/{p['photo_id']}.jpg",
                        label=p["label"],
                        caption=p["caption"],
                    )
                    for p in photos_list
                ]

            total_photos = sum(len(photos) for photos in photo_index.values())
            logger.info(
                f"✅ Loaded precomputed photo index from cloud JSON ({total_photos} photos for {len(photo_index)} businesses)"
            )
            return photo_index

        except Exception as e:
            logger.debug(f"Could not load JSON from cloud: {e}")
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

            logger.info(
                f"Indexed {loaded_count} photos for {len(photo_index)} businesses"
            )
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

    def _load_cloud_photo_index(self) -> Dict[str, List[PhotoMetadata]]:
        """
        Load photo metadata from Cloud Storage (photos.json).

        Expected format: JSONL file with photo records
        (same format as local photos.json)

        Returns empty dict if Cloud Storage not available or file not found.
        """
        if not self.cloud_storage_helper:
            return {}

        photo_index: Dict[str, List[PhotoMetadata]] = {}

        try:
            logger.info("Loading photo metadata from Cloud Storage...")
            photos_json_bytes = self.cloud_storage_helper.download_json("photos.json")

            # Parse JSONL
            if isinstance(photos_json_bytes, bytes):
                content = photos_json_bytes.decode("utf-8")
            else:
                # If it's already a dict (from download_json), convert back
                logger.debug("Got photos data from Cloud Storage")
                return {}

            loaded_count = 0
            for line in content.split("\n"):
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

                label = str(photo_record.get("label", "other"))
                caption = str(photo_record.get("caption", ""))

                photo_meta = PhotoMetadata(
                    photo_id=photo_id,
                    business_id=business_id,
                    path=f"gs://{self.cloud_storage_helper.bucket_name}/photos/{photo_id}.jpg",
                    label=label,
                    caption=caption,
                )

                photo_index.setdefault(business_id, []).append(photo_meta)
                loaded_count += 1

            logger.info(
                f"Loaded {loaded_count} photos for {len(photo_index)} businesses from Cloud Storage"
            )

            # Sort by label priority
            for business_id in photo_index:
                photo_index[business_id].sort(
                    key=lambda p: PhotoMetadata.label_priority(p.label), reverse=True
                )

        except Exception as e:
            logger.debug(f"Could not load photo index from Cloud Storage: {e}")
            return {}

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
                logger.debug(
                    f"POI index {poi_idx} (business_id={business_id}) not found in filtered POI data"
                )
                return None, None

            return business_id, row_data

        return None, None

    def _get_local_photos_for_business(self, business_id: str) -> List[str]:
        """
        Return photo URLs/paths for a business.

        Strategy (Cloud-first):
        1. First try Cloud Storage (primary - data already there)
        2. Then try local indexed photos (local fallback)
        3. Then try legacy local files (legacy fallback)

        Returns list of URLs (GCS signed URLs or local file paths).
        """
        if not business_id:
            return []

        # Try 1: Cloud Storage first (primary source)
        if self.cloud_storage_helper:
            if self._cloud_photo_index is None:
                self._cloud_photo_index = self._load_cloud_photo_index()

            # Defensive: ensure _cloud_photo_index is not None
            if self._cloud_photo_index is None:
                self._cloud_photo_index = {}

            indexed_photos = self._cloud_photo_index.get(business_id, [])
            if indexed_photos:
                # Get signed URLs for photos
                photo_urls = []
                for photo in indexed_photos:
                    try:
                        # Extract GCS path (photo_id.jpg) from PhotoMetadata
                        gcs_path = f"photos/{photo.photo_id}.jpg"
                        url = self.cloud_storage_helper.get_photo_url(gcs_path)
                        if url:
                            photo_urls.append(url)
                    except Exception as e:
                        logger.debug(
                            f"Failed to generate URL for {photo.photo_id}: {e}"
                        )
                if photo_urls:
                    logger.debug(
                        f"Using {len(photo_urls)} cloud photos for business {business_id}"
                    )
                    return photo_urls

        # Try 2: Local indexed photos (fallback)
        indexed_photos = self.local_photo_index.get(business_id, [])
        if indexed_photos:
            logger.debug(
                f"Cloud unavailable, using {len(indexed_photos)} local indexed photos for business {business_id}"
            )
            return [photo.path for photo in indexed_photos]

        # Try 3: Legacy local files (legacy fallback)
        if self.local_photos_dir:
            for ext in (".jpg", ".jpeg", ".png", ".webp"):
                legacy_path = self.local_photos_dir / f"{business_id}{ext}"
                if legacy_path.exists():
                    logger.debug(f"Using legacy local photo for business {business_id}")
                    return [str(legacy_path)]

        return []

    def _parse_dataset_photos(self, photos_field: Optional[any]) -> List[str]:
        """
        Parse photos field from dataset row into a list of URLs.

        Handles multiple formats: JSON list, comma-separated strings, single URL, etc.

        Args:
            photos_field: Raw photos field value (list, string, or None)

        Returns:
            List of photo URLs/paths
        """
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
        """Get complete POI information and photos by model-space index.

        Retrieves full details for a POI (restaurant, shop, etc) including coordinates,
        basic info (name, category, rating), and photos.

        **Validation**:
        Returns empty dict {} if POI fails validation. This is NORMAL and expected,
        especially with state filtering. Invalid POIs are automatically skipped.

        **Photo Resolution**:
        - Tries local_photos_dir first if configured
        - Falls back to Yelp dataset photos from DuckDB
        - Returns photo URL or None if unavailable

        Args:
            poi_idx: Integer index in model space (0 to n_items-1)

        Returns:
            Dict with: name, business_id, category, lat, lon, rating, num_reviews, photo_url
            or empty dict {} if validation fails

        Raises:
            No exceptions. Returns {} for any error. Logs validation failures at DEBUG level.

        Performance:
            - Database lookup: ~1-5ms per POI
            - Typical batch of 20: ~50-150ms total

        See Also:
            - get_poi_details_batch(): Optimized for multiple POI lookups
        """
        business_id, row = self._resolve_business_and_row(poi_idx)
        if row is None:
            # Expected when POI is outside filtered region or doesn't exist in map
            logger.warning(f"  🚫 POI {poi_idx}: NOT AVAILABLE (outside filtered data)")
            return {}

        # Validate required fields to skip invalid POI data
        try:
            name = str(row.get("name", "")).strip()
            if not name or name.lower() == "unnamed":
                logger.warning(f"  🚫 POI {poi_idx}: NO VALID NAME (got '{name}')")
                return {}

            # Validate coordinates are reasonable
            lat = float(row.get("latitude", 0))
            lon = float(row.get("longitude", 0))
            if lat == 0 and lon == 0:
                logger.warning(f"  🚫 POI {poi_idx} ({name}): INVALID COORDINATES (0,0)")
                return {}

            # Validate basic numeric fields
            rating = float(row.get("stars", 0))
            if rating < 0 or rating > 5:
                logger.warning(f"  🚫 POI {poi_idx} ({name}): INVALID RATING {rating}")
                return {}

        except (ValueError, TypeError) as e:
            logger.warning(f"  🚫 POI {poi_idx}: INVALID DATA - {e}")
            return {}

        # Resolve business_id early for use in logging
        resolved_business_id = str(row.get("business_id", business_id or ""))
        if not resolved_business_id:
            logger.warning(f"  🚫 POI {poi_idx}: NO BUSINESS_ID")
            return {}

        photos = self._get_local_photos_for_business(business_id or "")
        if not photos:
            photos = self._parse_dataset_photos(row.get("photos", ""))
            if photos:
                logger.debug(
                    f"POI {resolved_business_id}: Loaded {len(photos)} dataset photos"
                )
            else:
                logger.debug(
                    f"POI {resolved_business_id}: No photos found (photos field: {row.get('photos', 'N/A')[:100]})"
                )
        else:
            logger.debug(
                f"POI {resolved_business_id}: Loaded {len(photos)} local photos"
            )

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

    @cache_data
    def get_poi_details_batch(_self, poi_indices: List[int]) -> Dict[int, Dict]:
        """Batch retrieve POI details for multiple indices (optimized).

        Retrieves details for multiple POIs in a single operation with optimizations:
        - Reuses database connections (no per-POI reconnects)
        - Faster than calling get_poi_details() in a loop

        **Optimization**:
        - Single database query for all POIs vs N queries for N POIs
        - Connection pooling reused across batch
        - Typical: 20 POIs in ~50ms vs 200ms with individual calls

        Args:
            poi_indices: List of integer POI indices (0 to n_items-1)

        Returns:
            Dict mapping poi_idx to POI details dict.
            Only includes POIs that pass validation.
            Missing keys indicate failed validation.

        Performance:
            - 20 POIs: ~50-100ms
            - 100 POIs: ~200-300ms

        Example:
            ```python
            poi_indices = [0, 5, 10, 15, 20]
            details = data.get_poi_details_batch(poi_indices)

            for poi_idx in poi_indices:
                if poi_idx in details:  # Skip failed validation
                    print(details[poi_idx]['name'])
            ```

        See Also:
            - get_poi_details(): Single POI lookup
        Args:
            poi_indices: List of integer POI indices (0 to n_items-1)

        Returns:
            Dict mapping poi_idx to POI details dict.
            Only includes POIs that pass validation.
            Missing keys indicate failed validation.

        Performance:
            - 20 POIs: ~50-100ms
            - 100 POIs: ~200-300ms

        Example:
            ```python
            poi_indices = [0, 5, 10, 15, 20]
            details = data.get_poi_details_batch(poi_indices)

            for poi_idx in poi_indices:
                if poi_idx in details:  # Skip failed validation
                    print(details[poi_idx]['name'])
            ```

        See Also:
            - get_poi_details(): Single POI lookup
        """
        result = {}
        for poi_idx in poi_indices:
            details = _self.get_poi_details(poi_idx)
            if details:  # Only include valid POIs
                result[poi_idx] = details
        return result

    def get_valid_item_ids(self) -> set:
        """
        Get set of valid item indices for current state filter.

        Returns a set of POI indices that exist in the filtered dataset.
        Used by inference service to filter recommendations to current state.

        Returns:
            Set of valid POI indices
        """
        if not self.state_filter:
            # No filtering - all items are valid
            return set(self.index2item.keys())

        # Build set of valid items by checking index2item mapping
        # (index2item only contains items that exist in filtered state)
        valid_ids = set(self.index2item.keys())

        logger.info(
            f"State filter active ({self.state_filter}): {len(valid_ids)} valid items available for recommendations"
        )

        return valid_ids

    def get_pois_batch(self, poi_indices: List[int]) -> List[Dict]:
        """Bulk lookup for multiple POIs. Skips POIs that can't be resolved."""
        pois = [self.get_poi_details(idx) for idx in poi_indices]
        # Filter out empty dicts (POIs that couldn't be resolved)
        return [p for p in pois if p]

    def get_business_name(self, business_id: str) -> Optional[str]:
        """Get the business name for a given business_id or item index.

        Args:
            business_id: The Yelp business ID or item index (e.g., 'item_30167')

        Returns:
            Business name string, or None if not found
        """
        actual_business_id = business_id

        # Handle item indices like 'item_30167' by converting to real business_id
        if str(business_id).startswith("item_") and self.item2index:
            try:
                # Create reverse mapping cache if needed
                if not hasattr(self, "_index2item"):
                    self._index2item = {v: k for k, v in self.item2index.items()}

                # Extract index from 'item_30167' -> 30167
                index = int(str(business_id).replace("item_", ""))
                actual_business_id = self._index2item.get(index, business_id)
            except (ValueError, KeyError):
                pass

        # Look up POI in database
        poi = self._get_poi_from_duckdb(actual_business_id)
        if poi:
            return str(poi.get("name", "")).strip() or None

        return None

    @cache_data
    def get_test_users(_self, limit: int = 50) -> List[Dict]:
        """
        Get top N test users for dropdown selector.

        Filters to users with interactions in the current state_filter.
        Prefers the active run's saved test-user artifacts. Falls back to legacy
        broad caches/database queries only for older runs that do not have them.
        """
        run_users = _self._load_run_test_users(limit=limit)
        if run_users:
            logger.info(
                "✅ Loaded %d real test users from active run artifacts",
                len(run_users),
            )
            return run_users

        # Try loading from precomputed cache first
        project_root = Path(__file__).parent.parent.parent
        cache_dir = project_root / "data" / "ui_cache"

        if cache_dir.exists():
            state_suffix = f"_{_self.state_filter}" if _self.state_filter else "_all"
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
            from sqlalchemy import text

            if _self.backend_type == "cloudsql":
                from sqlalchemy import text

                # Query from Cloud SQL
                if _self.state_filter:
                    query = text(
                        """
                        SELECT
                            reviews.user_id,
                            COUNT(*) as interactions
                        FROM reviews
                        INNER JOIN businesses
                            ON reviews.business_id = businesses.business_id
                        WHERE reviews.stars >= 4.0 AND businesses.state = :state_filter
                        GROUP BY reviews.user_id
                        ORDER BY interactions DESC
                        LIMIT :limit
                    """
                    )
                    params = {"state_filter": _self.state_filter, "limit": limit}
                else:
                    query = text(
                        """
                        SELECT
                            user_id,
                            COUNT(*) as interactions
                        FROM reviews
                        WHERE stars >= 4.0
                        GROUP BY user_id
                        ORDER BY interactions DESC
                        LIMIT :limit
                    """
                    )
                    params = {"limit": limit}

                with _self.engine.connect() as conn:
                    users_df = pd.read_sql(query, conn, params=params)

            else:
                # Query from DuckDB - use yelp_review and yelp_business tables
                if _self.state_filter:
                    query = f"""
                        SELECT
                            reviews.user_id,
                            COUNT(*) as interactions
                        FROM yelp_review AS reviews
                        INNER JOIN yelp_business AS business
                            ON reviews.business_id = business.business_id
                        WHERE reviews.stars >= 4.0 AND business.state = '{_self.state_filter}'
                        GROUP BY reviews.user_id
                        ORDER BY interactions DESC
                        LIMIT {limit}
                    """
                else:
                    query = f"""
                        SELECT
                            user_id,
                            COUNT(*) as interactions
                        FROM yelp_review
                        WHERE stars >= 4.0
                        GROUP BY user_id
                        ORDER BY interactions DESC
                        LIMIT {limit}
                    """

                users_df = _self.conn.execute(query).df()
            result = [
                {"id": row["user_id"], "interactions": int(row["interactions"])}
                for _, row in users_df.iterrows()
            ]

            logger.info(
                f"✓ Found {len(result)} test users (backend: {_self.backend_type}, state_filter={_self.state_filter})"
            )
            return result

        except Exception as e:
            logger.debug(
                f"get_test_users failed (expected if using Streamlit Cloud without local data): {e}"
            )
            return []

    def _load_run_test_users(self, limit: int = 50) -> List[Dict]:
        """Load real held-out test users for the active run."""
        if not self.active_run_dir:
            return []

        if getattr(self, "cloud_storage_helper", None) and len(self.active_run_dir.name) == 15:
            gcs_path = f"models/{self.active_run_dir.name}/data/test_users_top50.json"
            try:
                users = self.cloud_storage_helper.read_json(gcs_path)
                if isinstance(users, list):
                    logger.info(
                        "Loaded real test users from GCS artifact: %s", gcs_path
                    )
                    return users[:limit]
            except Exception as e:
                logger.debug("Could not load run-scoped test users from cloud: %s", e)

        top_users_path = self.active_run_dir / "data" / "test_users_top50.json"
        if top_users_path.exists():
            try:
                with open(top_users_path, "r", encoding="utf-8") as f:
                    users = json.load(f)
                if isinstance(users, list):
                    return users[:limit]
            except Exception as e:
                logger.warning("Failed to load %s: %s", top_users_path, e)

        test_ids_path = self.active_run_dir / "data" / "test_user_ids.json"
        reviews_path = self.active_run_dir / "data" / "reviews_df.pkl"
        if test_ids_path.exists():
            try:
                with open(test_ids_path, "r", encoding="utf-8") as f:
                    test_user_ids = json.load(f)
                if reviews_path.exists():
                    with open(reviews_path, "rb") as f:
                        reviews_df = pickle.load(f)
                else:
                    shared_payload = load_shared_preprocessing_payload_for_run(
                        self.active_run_dir / "data"
                    )
                    if not shared_payload:
                        raise FileNotFoundError("shared preprocessing payload unavailable")
                    reviews_df = shared_payload["reviews"]

                counts = (
                    reviews_df[reviews_df["user_id"].isin(test_user_ids)]
                    .groupby("user_id")
                    .size()
                    .sort_values(ascending=False)
                )
                return [
                    {"id": str(user_id), "interactions": int(count)}
                    for user_id, count in counts.head(limit).items()
                ]
            except Exception as e:
                logger.warning(
                    "Failed to derive run-scoped test users from %s: %s",
                    self.active_run_dir,
                    e,
                )

        return []

    @cache_data
    def get_user_interactions(_self, user_id: str, min_stars: float = 4.0) -> List[int]:
        """
        Get list of POI indices the user has interacted with.

        Uses item2index mapping to ensure indices match model coordinate space.
        Filters to interactions in the same state/region as the loaded POI data.
        Supports both Cloud SQL and DuckDB backends.
        """
        try:
            from sqlalchemy import text

            if _self.backend_type == "cloudsql":
                # Query from Cloud SQL - using parameterized queries
                base_query = """
                    SELECT DISTINCT businesses.business_id
                    FROM reviews
                    JOIN businesses ON reviews.business_id = businesses.business_id
                    WHERE reviews.user_id = :user_id AND reviews.stars >= :min_stars
                """

                # Convert min_stars to int to match PostgreSQL integer column type
                params = {"user_id": user_id, "min_stars": int(min_stars)}

                if _self.state_filter:
                    base_query += " AND businesses.state = :state_filter"
                    params["state_filter"] = _self.state_filter

                with _self.engine.connect() as conn:
                    business_ids = pd.read_sql(text(base_query), conn, params=params)

            else:
                # Query from DuckDB - use yelp_review and yelp_business tables
                # Build WHERE clause using parameterized queries to prevent SQL injection
                where_clause = "WHERE reviews.user_id = ? AND reviews.stars >= ?"
                params = [user_id, min_stars]
                if _self.state_filter:
                    where_clause += " AND business.state = ?"
                    params.append(_self.state_filter)

                query = f"""
                    SELECT DISTINCT business.business_id
                    FROM yelp_review reviews
                    JOIN yelp_business business
                    ON reviews.business_id = business.business_id
                    {where_clause}
                """
                business_ids = _self.conn.execute(query, params).df()

            logger.debug(
                f"Query for user {user_id}: found {len(business_ids)} businesses with stars >= {min_stars}"
            )

            poi_indices: List[int] = []
            if _self.item2index:
                unmapped = 0
                for bid in business_ids["business_id"]:
                    if bid in _self.item2index:
                        poi_indices.append(_self.item2index[bid])
                    else:
                        unmapped += 1

                if unmapped > 0:
                    logger.debug(
                        f"User {user_id}: {unmapped} businesses not in item2index mapping (total mapping size: {len(_self.item2index)})"
                    )
                logger.debug(
                    f"✅ Successfully mapped {len(poi_indices)} interactions for user {user_id}"
                )
            else:
                logger.error(
                    f"❌ item2index mapping is empty/None - cannot map user interactions. "
                    f"item2index should be loaded from the configured item2index mapping "
                    f"(for example, item2index.pkl) during DataService initialization. "
                    f"Database returned {len(business_ids)} businesses, but index alignment is unavailable. "
                    f"Ensure the item2index mapping is present and configured for this environment."
                )

            if poi_indices:
                max_idx = max(poi_indices)
                logger.info(
                    f"User {user_id}: {len(poi_indices)} valid interactions, max_idx={max_idx}"
                )

                # Log warning if indices might be out of bounds (this is checked later by the caller)
                logger.debug(
                    f"Caller is responsible for validating indices are < n_items. "
                    f"Current max_idx={max_idx}"
                )
            else:
                logger.warning(
                    f"User {user_id}: No valid interactions found after mapping"
                )

            return poi_indices

        except Exception as e:
            logger.error(
                f"Failed to get interactions for user {user_id}: {e}", exc_info=True
            )
            return []

    @property
    def num_pois(self) -> int:
        """Total number of POIs (supports Cloud SQL or DuckDB backend)."""
        try:
            from sqlalchemy import text

            if self.backend_type == "cloudsql":
                # Query from Cloud SQL
                base_query = "SELECT COUNT(*) as cnt FROM businesses"
                params = {}

                if self.state_filter:
                    base_query += " WHERE state = :state_filter"
                    params["state_filter"] = self.state_filter

                with self.engine.connect() as conn:
                    result = pd.read_sql(text(base_query), conn, params=params)
                    return int(result["cnt"][0]) if len(result) > 0 else 0

            else:
                # Query from DuckDB - use yelp_business table
                if self.state_filter:
                    query = f"SELECT COUNT(*) as cnt FROM yelp_business WHERE state = '{self.state_filter}'"
                else:
                    query = "SELECT COUNT(*) as cnt FROM yelp_business"

                result = self.conn.execute(query).df()
                return int(result["cnt"][0]) if len(result) > 0 else 0

        except Exception as e:
            logger.warning(f"Failed to count POIs: {e}")
            # Fallback to item2index mapping size
            return len(self.item2index) if self.item2index else 0

    def close(self) -> None:
        """
        Close database connection (supports Cloud SQL or DuckDB backend).

        Properly disposes connection pool for Cloud SQL or closes DuckDB connection.
        """
        if self.backend_type == "cloudsql" and self.engine:
            self.engine.dispose()
            logger.info("Cloud SQL connection pool closed")
        elif self.backend_type == "duckdb" and self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")

    def get_all_users(self, min_reviews: int = 1) -> List[str]:
        """
        Get all unique user IDs from the database.

        Optionally filter to users with minimum number of reviews.
        """
        try:
            from sqlalchemy import text

            if self.backend_type == "cloudsql":
                # Query from Cloud SQL
                query = "SELECT DISTINCT user_id FROM reviews"
                if min_reviews > 1:
                    query += f" GROUP BY user_id HAVING COUNT(*) >= {min_reviews}"

                with self.engine.connect() as conn:
                    result = pd.read_sql(text(query), conn)
                    return result["user_id"].tolist() if len(result) > 0 else []

            else:
                # Query from DuckDB - use yelp_review table
                if min_reviews > 1:
                    query = f"""
                        SELECT DISTINCT user_id FROM yelp_review
                        GROUP BY user_id HAVING COUNT(*) >= {min_reviews}
                    """
                else:
                    query = "SELECT DISTINCT user_id FROM yelp_review"

                result = self.conn.execute(query).df()
                return result["user_id"].tolist() if len(result) > 0 else []

        except Exception as e:
            logger.error(f"Failed to get all users: {e}", exc_info=True)
            return []

    def _load_precomputed_user_matrices(self) -> Dict[str, any]:
        """
        Load precomputed CSR matrices for all users.

        Load priority:
        1. Active run local: outputs/<run_id>/precomputed/user_csr_matrices.pkl
        2. Active run cloud: models/<run_id>/precomputed/user_csr_matrices.pkl
        3. Legacy Cloud Storage: metadata/user_csr_matrices.pkl
        4. Legacy local: data/user_csr_matrices.pkl
        5. Return empty dict if none available

        Returns:
            Dict[user_id, csr_matrix] or empty dict if not found
        """
        if self.active_run_dir:
            if getattr(self, "cloud_storage_helper", None) and len(self.active_run_dir.name) == 15:
                cloud_run_path = (
                    f"models/{self.active_run_dir.name}/precomputed/user_csr_matrices.pkl"
                )
                try:
                    matrices = self.cloud_storage_helper.read_pickle(cloud_run_path)
                    if matrices:
                        logger.info(
                            f"✅ Loaded precomputed user matrices from active run cloud ({len(matrices)} users)"
                        )
                        return matrices
                except Exception as e:
                    logger.debug(f"Could not load active-run matrices from cloud: {e}")

            run_local_path = self.active_run_dir / "precomputed" / "user_csr_matrices.pkl"
            if run_local_path.exists():
                try:
                    with open(run_local_path, "rb") as f:
                        matrices = pickle.load(f)
                    logger.info(
                        f"✅ Loaded precomputed user matrices from active run ({len(matrices)} users)"
                    )
                    return matrices
                except Exception as e:
                    logger.debug(f"Could not load active-run matrices from local: {e}")

        # Try legacy cloud path
        cloud_path = "metadata/user_csr_matrices.pkl"
        try:
            if getattr(self, "cloud_storage_helper", None):
                matrices = self.cloud_storage_helper.read_pickle(cloud_path)
                if matrices:
                    logger.info(
                        f"✅ Loaded precomputed user matrices from cloud ({len(matrices)} users)"
                    )
                    return matrices
        except Exception as e:
            logger.debug(f"Could not load from cloud: {e}")

        # Try local
        local_path = Path("data") / "user_csr_matrices.pkl"
        if local_path.exists():
            try:
                with open(local_path, "rb") as f:
                    matrices = pickle.load(f)
                logger.info(
                    f"✅ Loaded precomputed user matrices from local ({len(matrices)} users)"
                )
                return matrices
            except Exception as e:
                logger.debug(f"Could not load from local: {e}")

        logger.debug("Precomputed user matrices not available (will build on-demand)")
        return {}

    def get_precomputed_user_matrix(self, user_id: str):
        """
        Get precomputed CSR matrix for a user.

        Returns:
            csr_matrix or None if not found
        """
        if not hasattr(self, "_user_matrices_cache"):
            self._user_matrices_cache = self._load_precomputed_user_matrices()

        return self._user_matrices_cache.get(user_id)

    def upload_to_cloud(self, local_path: str, cloud_path: str) -> bool:
        """
        Upload a file to Cloud Storage.

        Args:
            local_path: Local file path
            cloud_path: Cloud path (e.g., 'metadata/photo_index.pkl')

        Returns:
            True if successful, False otherwise
        """
        try:
            from google.cloud import storage

            bucket_name = self.config.get("GCS_BUCKET_NAME")
            if not bucket_name:
                logger.warning("GCS_BUCKET_NAME not configured, skipping cloud upload")
                return False

            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(cloud_path)

            blob.upload_from_filename(local_path)
            logger.info(f"✅ Uploaded to gs://{bucket_name}/{cloud_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to upload to cloud: {e}")
            return False
