"""Data service for loading and serving POI metadata."""

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import pandas as pd

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
        expected_n_items: Optional[int] = None,
        strict_run_artifacts: bool = False,
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
        self.expected_n_items = int(expected_n_items) if expected_n_items else None
        self.strict_run_artifacts = bool(strict_run_artifacts)
        self._missing_cloud_paths_warned: set[str] = set()
        self._business_columns_cache: Optional[set[str]] = None
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

        if self.strict_run_artifacts and not self.item2index:
            expected_path = (
                str(item2index_path)
                if item2index_path
                else "<run>/mappings/item2index.pkl"
            )
            raise RuntimeError(
                "Strict best-run mode requires run-scoped item2index mapping. "
                f"Missing or unreadable mapping at: {expected_path}"
            )

        if (
            self.item2index
            and self.expected_n_items is not None
            and len(self.item2index) != self.expected_n_items
        ):
            raise RuntimeError(
                "Run-scoped item2index mapping size mismatch: "
                f"expected {self.expected_n_items} items, got {len(self.item2index)}."
            )

        if self.strict_run_artifacts and self.item2index:
            normalized_indices: List[int] = []
            non_int_examples: List[str] = []
            for business_id, raw_idx in self.item2index.items():
                try:
                    normalized_indices.append(int(raw_idx))
                except (TypeError, ValueError):
                    non_int_examples.append(f"{business_id}:{raw_idx}")
                    if len(non_int_examples) >= 5:
                        break

            if non_int_examples:
                raise RuntimeError(
                    "Strict best-run mode requires integer item2index values. "
                    f"Found invalid entries: {non_int_examples}"
                )

            if len(set(normalized_indices)) != len(normalized_indices):
                raise RuntimeError(
                    "Strict best-run mode detected duplicate item2index values. "
                    "Mapping must be one-to-one."
                )

            if self.expected_n_items is not None and normalized_indices:
                min_idx = min(normalized_indices)
                max_idx = max(normalized_indices)
                expected_max = int(self.expected_n_items) - 1
                if min_idx != 0 or max_idx != expected_max:
                    raise RuntimeError(
                        "Strict best-run item2index range mismatch: "
                        f"expected [0, {expected_max}], got [{min_idx}, {max_idx}]."
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
        # In-memory cache for dataset statistics queries (key -> DataFrame/dict payload).
        self._stats_cache: Dict[Tuple[Any, ...], Any] = {}
        self._stats_mv_name = "ui_dataset_stats_by_state"
        self._stats_mv_ready = False

        # Precompute state-level summary aggregates once (fixed dataset assumption).
        # CloudSQL: materialized view; DuckDB: persisted table.
        self._ensure_stats_materialization()

        # In strict mode, load and validate precomputed matrices during startup.
        self._user_matrices_cache = (
            self._load_precomputed_user_matrices()
            if self.strict_run_artifacts
            else None
        )

    def _get_business_columns(self) -> set[str]:
        """Return available columns from businesses/yelp_business table."""
        if self._business_columns_cache is not None:
            return self._business_columns_cache

        default_cols = {
            "business_id",
            "name",
            "categories",
            "latitude",
            "longitude",
            "stars",
            "review_count",
        }

        try:
            from sqlalchemy import text

            if self.backend_type == "cloudsql" and self.engine is not None:
                query = text(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'businesses'
                    """
                )
                with self.engine.connect() as conn:
                    result = pd.read_sql(query, conn)
                cols = {str(v) for v in result["column_name"].tolist()}
                self._business_columns_cache = cols if cols else default_cols
                return self._business_columns_cache

            if self.backend_type == "duckdb" and self.conn is not None:
                result = self.conn.execute("PRAGMA table_info('yelp_business')").df()
                cols = {str(v) for v in result["name"].tolist()}
                self._business_columns_cache = (
                    cols if cols else (default_cols | {"photos"})
                )
                return self._business_columns_cache
        except Exception as e:
            logger.debug("Could not inspect business columns; using defaults: %s", e)

        self._business_columns_cache = default_cols
        return self._business_columns_cache

    def _get_batch_projection_columns(self) -> List[str]:
        """Return batch POI lookup projection, including optional photos when available."""
        required = [
            "business_id",
            "name",
            "categories",
            "latitude",
            "longitude",
            "stars",
            "review_count",
        ]
        available = self._get_business_columns()
        columns = [col for col in required if col in available]
        if "photos" in available:
            columns.append("photos")
        return columns

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

    def _build_poi_details_from_row(
        self,
        poi_idx: int,
        row: Dict,
        *,
        fallback_business_id: Optional[str] = None,
    ) -> Dict:
        """Validate one business row and normalize it for UI consumers.

        Used by both single-item and batched retrieval paths so cards/maps/history
        all share one normalization contract (schema, validation, and photo logic).

        Returns:
            Canonical POI dict, or {} when row is invalid/missing required fields.
        """
        try:
            name = str(row.get("name", "")).strip()
            if not name or name.lower() == "unnamed":
                logger.warning(f"  🚫 POI {poi_idx}: NO VALID NAME (got '{name}')")
                return {}

            lat = float(row.get("latitude", 0))
            lon = float(row.get("longitude", 0))
            if lat == 0 and lon == 0:
                logger.warning(f"  🚫 POI {poi_idx} ({name}): INVALID COORDINATES (0,0)")
                return {}

            rating = float(row.get("stars", 0))
            if rating < 0 or rating > 5:
                logger.warning(f"  🚫 POI {poi_idx} ({name}): INVALID RATING {rating}")
                return {}
        except (ValueError, TypeError) as e:
            logger.warning(f"  🚫 POI {poi_idx}: INVALID DATA - {e}")
            return {}

        resolved_business_id = str(
            row.get("business_id", fallback_business_id or "")
        ).strip()
        if not resolved_business_id:
            logger.warning(f"  🚫 POI {poi_idx}: NO BUSINESS_ID")
            return {}

        photos = self._get_local_photos_for_business(resolved_business_id)
        if not photos:
            photos = self._parse_dataset_photos(row.get("photos", ""))
            if photos:
                logger.debug(
                    "POI %s: Loaded %d dataset photos",
                    resolved_business_id,
                    len(photos),
                )
            else:
                logger.debug(
                    "POI %s: No photos found (photos field: %s)",
                    resolved_business_id,
                    str(row.get("photos", "N/A"))[:100],
                )
        else:
            logger.debug(
                "POI %s: Loaded %d local photos", resolved_business_id, len(photos)
            )

        return {
            "poi_idx": int(poi_idx),
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

        return self._build_poi_details_from_row(
            poi_idx,
            row,
            fallback_business_id=business_id,
        )

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
        if not poi_indices:
            return {}

        ordered_unique_indices: List[int] = []
        seen_indices = set()
        for raw_idx in poi_indices:
            try:
                poi_idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if poi_idx in seen_indices:
                continue
            seen_indices.add(poi_idx)
            ordered_unique_indices.append(poi_idx)

        if not ordered_unique_indices:
            return {}

        business_ids: Dict[int, str] = {}
        for poi_idx in ordered_unique_indices:
            business_id = _self.index2item.get(poi_idx) if _self.index2item else None
            if business_id:
                business_ids[poi_idx] = business_id

        if not business_ids:
            logger.debug("Batch POI lookup skipped: no indices resolved in item2index")
            return {}

        rows_by_business_id: Dict[str, Dict] = {}
        business_id_list = list(dict.fromkeys(business_ids.values()))

        try:
            from sqlalchemy import bindparam, text

            projection_cols = _self._get_batch_projection_columns()
            projection_sql = ", ".join(projection_cols)

            if _self.backend_type == "cloudsql":
                query = text(
                    """
                        SELECT {projection_sql}
                        FROM businesses
                        WHERE business_id IN :business_ids
                        """.format(
                        projection_sql=projection_sql
                    )
                ).bindparams(bindparam("business_ids", expanding=True))
                with _self.engine.connect() as conn:
                    rows = pd.read_sql(
                        query,
                        conn,
                        params={"business_ids": business_id_list},
                    )
                    if len(rows) > 0:
                        rows_by_business_id = {
                            str(row["business_id"]): row.to_dict()
                            for _, row in rows.iterrows()
                        }
            else:
                placeholders = ",".join("?" for _ in business_id_list)
                query = (
                    f"SELECT {projection_sql} "
                    f"FROM yelp_business WHERE business_id IN ({placeholders})"
                )
                rows = _self.conn.execute(query, business_id_list).df()
                if len(rows) > 0:
                    rows_by_business_id = {
                        str(row["business_id"]): row.to_dict()
                        for _, row in rows.iterrows()
                    }
        except Exception as e:
            logger.warning(
                "Batch POI lookup failed, falling back to single lookups: %s", e
            )
            return {
                idx: details
                for idx in ordered_unique_indices
                if (details := _self.get_poi_details(idx))
            }

        result: Dict[int, Dict] = {}
        for poi_idx in ordered_unique_indices:
            business_id = business_ids.get(poi_idx)
            row = rows_by_business_id.get(business_id or "")
            if not row:
                continue
            details = _self._build_poi_details_from_row(
                poi_idx,
                row,
                fallback_business_id=business_id,
            )
            if details:
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

        if _self.strict_run_artifacts:
            run_name = (
                _self.active_run_dir.name if _self.active_run_dir else "<unknown>"
            )
            raise RuntimeError(
                "Strict best-run mode requires run-scoped `test_users_top50.json`. "
                f"Missing artifact for run {run_name} at "
                f"{_self.active_run_dir / 'data' / 'test_users_top50.json' if _self.active_run_dir else '<missing run dir>'}."
            )

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

    def _warn_missing_cloud_path_once(self, gcs_path: str) -> None:
        """Emit a single warning per missing cloud artifact path per process."""
        if gcs_path in self._missing_cloud_paths_warned:
            return
        self._missing_cloud_paths_warned.add(gcs_path)
        logger.warning(
            "Cloud artifact not found, using local strict-run fallback when available: %s",
            gcs_path,
        )

    def _load_run_test_users(self, limit: int = 50) -> List[Dict]:
        """Load real held-out test users for the active run."""
        if not self.active_run_dir:
            return []

        if (
            getattr(self, "cloud_storage_helper", None)
            and len(self.active_run_dir.name) == 15
        ):
            gcs_path = f"outputs/{self.active_run_dir.name}/data/test_users_top50.json"
            try:
                if self.cloud_storage_helper.exists(gcs_path):
                    users = self.cloud_storage_helper.read_json(gcs_path)
                    if isinstance(users, list):
                        logger.info(
                            "Loaded real test users from GCS artifact: %s", gcs_path
                        )
                        return users[:limit]
                else:
                    self._warn_missing_cloud_path_once(gcs_path)
            except Exception as e:
                logger.warning(
                    "Unexpected cloud read failure for run-scoped test users at %s: %s",
                    gcs_path,
                    e,
                )

        top_users_path = self.active_run_dir / "data" / "test_users_top50.json"
        if top_users_path.exists():
            try:
                with open(top_users_path, "r", encoding="utf-8") as f:
                    users = json.load(f)
                if isinstance(users, list):
                    return users[:limit]
            except Exception as e:
                logger.warning("Failed to load %s: %s", top_users_path, e)

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

    def _stats_scope_where(
        self,
        *,
        scope: str,
        state: Optional[str],
        city: Optional[str],
        business_alias: str = "b",
    ) -> tuple[str, Dict[str, Any]]:
        """Build SQL WHERE clause + params for stats queries."""
        clauses: List[str] = []
        params: Dict[str, Any] = {}

        if scope == "training":
            training_state = (self.state_filter or "").strip()
            if training_state:
                clauses.append(f"{business_alias}.state = :training_state")
                params["training_state"] = training_state

        if state and state != "All":
            clauses.append(f"{business_alias}.state = :state")
            params["state"] = state

        if city and city != "All":
            clauses.append(f"{business_alias}.city = :city")
            params["city"] = city

        if not clauses:
            return "", params
        return " WHERE " + " AND ".join(clauses), params

    def _cached_stats_value(self, key: tuple[Any, ...], compute_fn):
        cached = self._stats_cache.get(key)
        if cached is not None:
            if isinstance(cached, pd.DataFrame):
                return cached.copy()
            if isinstance(cached, dict):
                return dict(cached)
            return cached

        value = compute_fn()
        if isinstance(value, pd.DataFrame):
            self._stats_cache[key] = value.copy()
            return value
        if isinstance(value, dict):
            self._stats_cache[key] = dict(value)
            return value
        self._stats_cache[key] = value
        return value

    def _query_df_stats(self, query: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Execute a stats query against CloudSQL or DuckDB."""
        if self.backend_type == "cloudsql":
            from sqlalchemy import text

            with self.engine.connect() as conn:
                return pd.read_sql(text(query), conn, params=params)

        # DuckDB uses $name placeholders (not :name).
        import re

        duckdb_query = re.sub(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)", r"$\1", query)
        return self.conn.execute(duckdb_query, params).df()

    def _sql_cast_float(self, expr: str) -> str:
        """Backend-specific float cast expression."""
        if self.backend_type == "cloudsql":
            # PostgreSQL has no TRY_CAST; guard conversion for mixed-quality text columns.
            # Accept plain integer/decimal values and return NULL for malformed strings.
            return (
                "CASE "
                f"WHEN TRIM(CAST({expr} AS TEXT)) ~ '^-?[0-9]+(\\.[0-9]+)?$' "
                f"THEN CAST({expr} AS DOUBLE PRECISION) "
                "ELSE NULL END"
            )
        return f"TRY_CAST({expr} AS DOUBLE)"

    def _sql_cast_int(self, expr: str) -> str:
        """Backend-specific integer cast expression."""
        if self.backend_type == "cloudsql":
            return (
                "CASE "
                f"WHEN TRIM(CAST({expr} AS TEXT)) ~ '^-?[0-9]+$' "
                f"THEN CAST({expr} AS INTEGER) "
                "ELSE NULL END"
            )
        return f"TRY_CAST({expr} AS INTEGER)"

    def _sql_round_avg(self, expr: str, digits: int = 3) -> str:
        """Backend-specific rounded AVG expression."""
        if self.backend_type == "cloudsql":
            return f"ROUND((AVG({expr}))::numeric, {int(digits)})"
        return f"ROUND(AVG({expr}), {int(digits)})"

    def _ensure_stats_materialization(self) -> None:
        """Create precomputed state-level stats structure for faster KPI queries."""
        try:
            threshold = float(self.config.get("pos_threshold", 4.0))
        except Exception:
            threshold = 4.0

        # Try CloudSQL materialized view when available.
        if self.engine is not None:
            self._ensure_stats_materialization_cloudsql(threshold)
        # Also try local DuckDB persisted table when local DB exists.
        self._ensure_stats_materialization_duckdb(threshold)

    def _ensure_stats_materialization_cloudsql(self, threshold: float) -> None:
        """Create CloudSQL materialized view for per-state KPI stats."""
        if self.engine is None:
            return

        from sqlalchemy import text

        mv = self._stats_mv_name
        cast_r = self._sql_cast_float("r.stars")
        cast_b = self._sql_cast_float("b.stars")
        query = f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS {mv} AS
            WITH review_base AS (
                SELECT
                    b.state,
                    r.user_id,
                    r.business_id,
                    {cast_r} AS review_stars,
                    EXTRACT(YEAR FROM r.date) AS year
                FROM reviews r
                JOIN businesses b ON b.business_id = r.business_id
                WHERE b.state IS NOT NULL AND b.state <> ''
            ),
            business_base AS (
                SELECT
                    b.state,
                    b.business_id,
                    {cast_b} AS business_stars
                FROM businesses b
                WHERE b.state IS NOT NULL AND b.state <> ''
            )
            SELECT
                bb.state AS state,
                COUNT(DISTINCT bb.business_id) AS n_businesses,
                COUNT(rb.business_id) AS n_interactions,
                COUNT(DISTINCT rb.user_id) AS n_users,
                COUNT(DISTINCT rb.business_id) AS n_items,
                {self._sql_round_avg('bb.business_stars')} AS avg_business_rating,
                {self._sql_round_avg('rb.review_stars')} AS avg_review_rating,
                MIN(rb.year) AS min_year,
                MAX(rb.year) AS max_year,
                COUNT(*) FILTER (WHERE rb.review_stars >= {threshold}) AS pos_interactions,
                COUNT(DISTINCT rb.user_id) FILTER (WHERE rb.review_stars >= {threshold}) AS pos_users,
                COUNT(DISTINCT rb.business_id) FILTER (WHERE rb.review_stars >= {threshold}) AS pos_items,
                {threshold}::double precision AS pos_threshold
            FROM business_base bb
            LEFT JOIN review_base rb
              ON rb.state = bb.state AND rb.business_id = bb.business_id
            GROUP BY bb.state
        """
        idx_query = f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{mv}_state ON {mv}(state)"

        try:
            with self.engine.begin() as conn:
                conn.execute(text(query))
                conn.execute(text(idx_query))
            self._stats_mv_ready = True
            logger.info("Precomputed CloudSQL materialized view ready: %s", mv)
        except Exception as exc:
            self._stats_mv_ready = False
            logger.warning(
                "Could not create CloudSQL materialized view %s: %s",
                mv,
                exc,
            )

    def _ensure_stats_materialization_duckdb(self, threshold: float) -> None:
        """Create DuckDB persisted table for per-state KPI stats."""
        table = self._stats_mv_name
        query = f"""
            CREATE OR REPLACE TABLE {table} AS
            WITH review_base AS (
                SELECT
                    b.state,
                    r.user_id,
                    r.business_id,
                    TRY_CAST(r.stars AS DOUBLE) AS review_stars,
                    EXTRACT(YEAR FROM r.date) AS year
                FROM yelp_review r
                JOIN yelp_business b ON b.business_id = r.business_id
                WHERE b.state IS NOT NULL AND b.state <> ''
            ),
            business_base AS (
                SELECT
                    b.state,
                    b.business_id,
                    TRY_CAST(b.stars AS DOUBLE) AS business_stars
                FROM yelp_business b
                WHERE b.state IS NOT NULL AND b.state <> ''
            )
            SELECT
                bb.state AS state,
                COUNT(DISTINCT bb.business_id) AS n_businesses,
                COUNT(rb.business_id) AS n_interactions,
                COUNT(DISTINCT rb.user_id) AS n_users,
                COUNT(DISTINCT rb.business_id) AS n_items,
                ROUND(AVG(bb.business_stars), 3) AS avg_business_rating,
                ROUND(AVG(rb.review_stars), 3) AS avg_review_rating,
                MIN(rb.year) AS min_year,
                MAX(rb.year) AS max_year,
                SUM(CASE WHEN rb.review_stars >= {threshold} THEN 1 ELSE 0 END) AS pos_interactions,
                COUNT(DISTINCT CASE WHEN rb.review_stars >= {threshold} THEN rb.user_id END) AS pos_users,
                COUNT(DISTINCT CASE WHEN rb.review_stars >= {threshold} THEN rb.business_id END) AS pos_items,
                {threshold} AS pos_threshold
            FROM business_base bb
            LEFT JOIN review_base rb
              ON rb.state = bb.state AND rb.business_id = bb.business_id
            GROUP BY bb.state
        """
        # If runtime backend is DuckDB, use active connection.
        if self.conn is not None:
            try:
                self.conn.execute(query)
                self._stats_mv_ready = True
                logger.info("Precomputed DuckDB stats table ready: %s", table)
                return
            except Exception as exc:
                self._stats_mv_ready = False
                logger.warning(
                    "Could not create DuckDB stats table %s: %s",
                    table,
                    exc,
                )
                return

        # Otherwise, opportunistically materialize in local DB file if present.
        if not self.duckdb_path.exists():
            return
        try:
            temp_conn = duckdb.connect(str(self.duckdb_path))
            try:
                temp_conn.execute(query)
                logger.info(
                    "Precomputed DuckDB stats table ready in local DB file: %s", table
                )
            finally:
                temp_conn.close()
        except Exception as exc:
            logger.warning(
                "Could not create DuckDB stats table %s in %s: %s",
                table,
                self.duckdb_path,
                exc,
            )

    def _get_precomputed_state_summary(
        self,
        *,
        state: Optional[str],
        pos_threshold: float,
    ) -> Optional[Dict[str, Any]]:
        """Fetch precomputed KPI summary from state-level materialization."""
        if not self._stats_mv_ready or not state:
            return None

        params = {"state": state}
        query = f"""
            SELECT
                n_businesses,
                n_users,
                n_items,
                n_interactions,
                avg_business_rating,
                avg_review_rating,
                min_year,
                max_year,
                pos_users,
                pos_items,
                pos_interactions,
                pos_threshold
            FROM {self._stats_mv_name}
            WHERE state = :state
            LIMIT 1
        """
        try:
            df = self._query_df_stats(query, params)
        except Exception as exc:
            logger.debug("Precomputed state summary unavailable for %s: %s", state, exc)
            return None

        if df.empty:
            return None

        row = df.iloc[0].to_dict()
        stored_threshold = row.get("pos_threshold")
        try:
            if stored_threshold is not None and float(stored_threshold) != float(
                pos_threshold
            ):
                return None
        except Exception:
            return None

        pos_users = int(row.get("pos_users") or 0)
        pos_items = int(row.get("pos_items") or 0)
        pos_interactions = int(row.get("pos_interactions") or 0)
        max_possible = pos_users * pos_items
        density_pct = (100.0 * pos_interactions / max_possible) if max_possible else 0.0
        row["density_pct"] = float(density_pct)
        return row

    def get_dataset_summary(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        city: Optional[str] = None,
        pos_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return KPI summary for Dataset Statistics page."""
        threshold = (
            float(pos_threshold)
            if pos_threshold is not None
            else float(self.config.get("pos_threshold", 4.0))
        )
        cache_key = ("summary", scope, state, city, threshold)

        def _compute() -> Dict[str, Any]:
            b_stars = self._sql_cast_float("b.stars")
            r_stars = self._sql_cast_float("r.stars")
            avg_biz = self._sql_round_avg("stars")
            avg_rev = self._sql_round_avg("stars")
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=city, business_alias="b"
            )
            params["pos_threshold"] = threshold

            # Fast path: when a state filter is active (default UI behavior), use
            # precomputed state-level summary from materialized structure.
            if city is None and state:
                precomputed = self._get_precomputed_state_summary(
                    state=state, pos_threshold=threshold
                )
                if precomputed is not None:
                    return precomputed

            query = f"""
                WITH filtered_reviews AS (
                    SELECT
                        r.user_id,
                        r.business_id,
                        {r_stars} AS stars,
                        EXTRACT(YEAR FROM r.date) AS year
                    FROM {'reviews' if self.backend_type == 'cloudsql' else 'yelp_review'} r
                    JOIN {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                      ON r.business_id = b.business_id
                    {where_sql}
                ),
                filtered_businesses AS (
                    SELECT
                        b.business_id,
                        {b_stars} AS stars
                    FROM {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                    {where_sql}
                ),
                positive AS (
                    SELECT *
                    FROM filtered_reviews
                    WHERE stars >= :pos_threshold
                )
                SELECT
                    (SELECT COUNT(*) FROM filtered_businesses) AS n_businesses,
                    (SELECT {avg_biz} FROM filtered_businesses) AS avg_business_rating,
                    (SELECT COUNT(DISTINCT user_id) FROM filtered_reviews) AS n_users,
                    (SELECT COUNT(DISTINCT business_id) FROM filtered_reviews) AS n_items,
                    (SELECT COUNT(*) FROM filtered_reviews) AS n_interactions,
                    (SELECT {avg_rev} FROM filtered_reviews) AS avg_review_rating,
                    (SELECT MIN(year) FROM filtered_reviews) AS min_year,
                    (SELECT MAX(year) FROM filtered_reviews) AS max_year,
                    (SELECT COUNT(DISTINCT user_id) FROM positive) AS pos_users,
                    (SELECT COUNT(DISTINCT business_id) FROM positive) AS pos_items,
                    (SELECT COUNT(*) FROM positive) AS pos_interactions
            """
            try:
                df = self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "Dataset summary query failed; trying lightweight fallback summary: %s",
                    exc,
                )
                # Lightweight fallback: avoid DISTINCT-heavy aggregates that can exceed
                # CloudSQL temp_file_limit on large review tables.
                b_query = f"""
                    SELECT
                        COUNT(*) AS n_businesses,
                        {self._sql_round_avg(b_stars)} AS avg_business_rating
                    FROM {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                    {where_sql}
                """
                r_query = f"""
                    SELECT
                        COUNT(*) AS n_interactions,
                        {self._sql_round_avg(r_stars)} AS avg_review_rating,
                        MIN(EXTRACT(YEAR FROM r.date)) AS min_year,
                        MAX(EXTRACT(YEAR FROM r.date)) AS max_year,
                        SUM(CASE WHEN {r_stars} >= :pos_threshold THEN 1 ELSE 0 END) AS pos_interactions
                    FROM {'reviews' if self.backend_type == 'cloudsql' else 'yelp_review'} r
                    JOIN {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                      ON r.business_id = b.business_id
                    {where_sql}
                """

                try:
                    b_df = self._query_df_stats(b_query, params)
                    r_df = self._query_df_stats(r_query, params)
                except Exception as fallback_exc:
                    logger.warning(
                        "Lightweight summary fallback also failed; returning empty summary: %s",
                        fallback_exc,
                    )
                    df = pd.DataFrame()
                else:
                    n_businesses = (
                        int((b_df.iloc[0].get("n_businesses") or 0))
                        if not b_df.empty
                        else 0
                    )
                    # n_items approximates active item universe for fallback path.
                    n_items = n_businesses
                    n_interactions = (
                        int((r_df.iloc[0].get("n_interactions") or 0))
                        if not r_df.empty
                        else 0
                    )
                    avg_business_rating = (
                        float((b_df.iloc[0].get("avg_business_rating") or 0.0))
                        if not b_df.empty
                        else 0.0
                    )
                    avg_review_rating = (
                        float((r_df.iloc[0].get("avg_review_rating") or 0.0))
                        if not r_df.empty
                        else 0.0
                    )
                    min_year = r_df.iloc[0].get("min_year") if not r_df.empty else None
                    max_year = r_df.iloc[0].get("max_year") if not r_df.empty else None
                    pos_interactions = (
                        int((r_df.iloc[0].get("pos_interactions") or 0))
                        if not r_df.empty
                        else 0
                    )

                    n_users = 0
                    pos_users = 0
                    pos_items = 0
                    if self.backend_type == "cloudsql" and not where_sql:
                        # Fast global fallback when a users table is present.
                        try:
                            users_df = self._query_df_stats(
                                "SELECT COUNT(*) AS n_users FROM users",
                                {},
                            )
                            if not users_df.empty:
                                n_users = int(users_df.iloc[0].get("n_users") or 0)
                        except Exception:
                            pass

                    max_possible = pos_users * pos_items
                    density_pct = (
                        100.0 * pos_interactions / max_possible if max_possible else 0.0
                    )
                    return {
                        "n_businesses": n_businesses,
                        "n_users": n_users,
                        "n_items": n_items,
                        "n_interactions": n_interactions,
                        "avg_business_rating": avg_business_rating,
                        "avg_review_rating": avg_review_rating,
                        "min_year": min_year,
                        "max_year": max_year,
                        "pos_users": pos_users,
                        "pos_items": pos_items,
                        "pos_interactions": pos_interactions,
                        "density_pct": float(density_pct),
                    }
            if df.empty:
                return {
                    "n_businesses": 0,
                    "n_users": 0,
                    "n_items": 0,
                    "n_interactions": 0,
                    "avg_business_rating": 0.0,
                    "avg_review_rating": 0.0,
                    "min_year": None,
                    "max_year": None,
                    "pos_users": 0,
                    "pos_items": 0,
                    "pos_interactions": 0,
                    "density_pct": 0.0,
                }

            row = df.iloc[0].to_dict()
            pos_users = int(row.get("pos_users") or 0)
            pos_items = int(row.get("pos_items") or 0)
            pos_interactions = int(row.get("pos_interactions") or 0)
            max_possible = pos_users * pos_items
            density_pct = (
                (100.0 * pos_interactions / max_possible) if max_possible else 0.0
            )
            row["density_pct"] = float(density_pct)
            return row

        return self._cached_stats_value(cache_key, _compute)

    def get_review_volume_by_year(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        city: Optional[str] = None,
        rating_min: Optional[float] = None,
        rating_max: Optional[float] = None,
    ) -> pd.DataFrame:
        """Return yearly review counts."""
        cache_key = ("volume_by_year", scope, state, city, rating_min, rating_max)

        def _compute() -> pd.DataFrame:
            r_stars = self._sql_cast_float("r.stars")
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=city, business_alias="b"
            )
            rating_clauses: List[str] = []
            if rating_min is not None:
                rating_clauses.append(f"{r_stars} >= :rating_min")
                params["rating_min"] = float(rating_min)
            if rating_max is not None:
                rating_clauses.append(f"{r_stars} <= :rating_max")
                params["rating_max"] = float(rating_max)

            if rating_clauses:
                if where_sql:
                    where_sql += " AND " + " AND ".join(rating_clauses)
                else:
                    where_sql = " WHERE " + " AND ".join(rating_clauses)

            query = f"""
                SELECT
                    EXTRACT(YEAR FROM r.date) AS year,
                    COUNT(*) AS review_count
                FROM {'reviews' if self.backend_type == 'cloudsql' else 'yelp_review'} r
                JOIN {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                  ON r.business_id = b.business_id
                {where_sql}
                GROUP BY year
                ORDER BY year
            """
            try:
                return self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "Review volume query failed; returning empty frame: %s", exc
                )
                return pd.DataFrame(columns=["year", "review_count"])

        return self._cached_stats_value(cache_key, _compute)

    def get_rating_distribution(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        city: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return review star distribution."""
        cache_key = ("rating_distribution", scope, state, city)

        def _compute() -> pd.DataFrame:
            r_stars = self._sql_cast_float("r.stars")
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=city, business_alias="b"
            )
            query = f"""
                SELECT
                    CAST({r_stars} AS INTEGER) AS rating_bucket,
                    COUNT(*) AS review_count
                FROM {'reviews' if self.backend_type == 'cloudsql' else 'yelp_review'} r
                JOIN {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                  ON r.business_id = b.business_id
                {where_sql}
                GROUP BY rating_bucket
                ORDER BY rating_bucket
            """
            try:
                df = self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "Rating distribution query failed; returning empty frame: %s", exc
                )
                return pd.DataFrame(columns=["stars", "review_count"])
            if not df.empty:
                df = df.rename(columns={"rating_bucket": "stars"})
            return df

        return self._cached_stats_value(cache_key, _compute)

    def get_top_cities(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Return top cities by business count."""
        cache_key = ("top_cities", scope, state, limit)

        def _compute() -> pd.DataFrame:
            b_stars = self._sql_cast_float("b.stars")
            avg_rating_expr = self._sql_round_avg(b_stars)
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=None, business_alias="b"
            )
            params["limit"] = int(limit)
            query = f"""
                SELECT
                    b.city,
                    b.state,
                    COUNT(*) AS n_businesses,
                    {avg_rating_expr} AS avg_rating
                FROM {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                {where_sql}
                GROUP BY b.city, b.state
                ORDER BY n_businesses DESC
                LIMIT :limit
            """
            fallback_query = f"""
                SELECT
                    b.city,
                    b.state,
                    COUNT(*) AS n_businesses,
                    CAST(NULL AS {'DOUBLE PRECISION' if self.backend_type == 'cloudsql' else 'DOUBLE'}) AS avg_rating
                FROM {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                {where_sql}
                GROUP BY b.city, b.state
                ORDER BY n_businesses DESC
                LIMIT :limit
            """
            try:
                return self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "Top cities primary query failed; falling back to count-only query: %s",
                    exc,
                )
                return self._query_df_stats(fallback_query, params)

        return self._cached_stats_value(cache_key, _compute)

    def get_top_categories(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        city: Optional[str] = None,
        limit: int = 20,
    ) -> pd.DataFrame:
        """Return top category combinations by business count."""
        cache_key = ("top_categories", scope, state, city, limit)

        def _compute() -> pd.DataFrame:
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=city, business_alias="b"
            )
            params["limit"] = int(limit)
            if self.backend_type == "cloudsql":
                avg_rating_expr = self._sql_round_avg(self._sql_cast_float("stars"))
                query = f"""
                    SELECT
                        category,
                        COUNT(*) AS n_businesses,
                        {avg_rating_expr} AS avg_rating
                    FROM (
                        SELECT
                            b.business_id,
                            b.stars,
                            TRIM(cat) AS category
                        FROM businesses b
                        CROSS JOIN LATERAL regexp_split_to_table(COALESCE(b.categories, ''), E'\\s*,\\s*') AS cat
                        {where_sql}
                    ) exploded
                    WHERE category IS NOT NULL AND category <> ''
                    GROUP BY category
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """
                fallback_query = """
                    SELECT
                        category,
                        COUNT(*) AS n_businesses,
                        CAST(NULL AS DOUBLE PRECISION) AS avg_rating
                    FROM (
                        SELECT
                            TRIM(cat) AS category
                        FROM businesses b
                        CROSS JOIN LATERAL regexp_split_to_table(COALESCE(b.categories, ''), E'\\s*,\\s*') AS cat
                        {where_sql}
                    ) exploded
                    WHERE category IS NOT NULL AND category <> ''
                    GROUP BY category
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """.format(
                    where_sql=where_sql
                )
            else:
                query = f"""
                    SELECT
                        category,
                        COUNT(*) AS n_businesses,
                        {self._sql_round_avg(self._sql_cast_float('stars'))} AS avg_rating
                    FROM (
                        SELECT
                            b.business_id,
                            b.stars,
                            TRIM(t.cat) AS category
                        FROM yelp_business b,
                             UNNEST(string_split(COALESCE(b.categories, ''), ',')) AS t(cat)
                        {where_sql}
                    ) exploded
                    WHERE category IS NOT NULL AND category <> ''
                    GROUP BY category
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """
                fallback_query = """
                    SELECT
                        category,
                        COUNT(*) AS n_businesses,
                        CAST(NULL AS DOUBLE) AS avg_rating
                    FROM (
                        SELECT
                            TRIM(t.cat) AS category
                        FROM yelp_business b,
                             UNNEST(string_split(COALESCE(b.categories, ''), ',')) AS t(cat)
                        {where_sql}
                    ) exploded
                    WHERE category IS NOT NULL AND category <> ''
                    GROUP BY category
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """.format(
                    where_sql=where_sql
                )
            try:
                return self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "Top categories primary query failed; falling back to count-only query: %s",
                    exc,
                )
                return self._query_df_stats(fallback_query, params)

        return self._cached_stats_value(cache_key, _compute)

    def get_state_distribution(self, *, limit: int = 30) -> pd.DataFrame:
        """Return state-level business distribution (always global)."""
        cache_key = ("state_distribution", limit)

        def _compute() -> pd.DataFrame:
            params = {"limit": int(limit)}
            if self.backend_type == "cloudsql":
                stars = self._sql_cast_float("stars")
                avg_rating_expr = self._sql_round_avg(stars)
                query = """
                    SELECT
                        state,
                        COUNT(*) AS n_businesses,
                        {avg_rating_expr} AS avg_rating
                    FROM businesses
                    WHERE state IS NOT NULL AND state <> ''
                    GROUP BY state
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """
                query = query.format(avg_rating_expr=avg_rating_expr)
                fallback_query = """
                    SELECT
                        state,
                        COUNT(*) AS n_businesses,
                        CAST(NULL AS DOUBLE PRECISION) AS avg_rating
                    FROM businesses
                    WHERE state IS NOT NULL AND state <> ''
                    GROUP BY state
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """
            else:
                query = """
                    SELECT
                        state,
                        COUNT(*) AS n_businesses,
                        ROUND(AVG(TRY_CAST(stars AS DOUBLE)), 3) AS avg_rating
                    FROM yelp_business
                    WHERE state IS NOT NULL AND state <> ''
                    GROUP BY state
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """
                fallback_query = """
                    SELECT
                        state,
                        COUNT(*) AS n_businesses,
                        CAST(NULL AS DOUBLE) AS avg_rating
                    FROM yelp_business
                    WHERE state IS NOT NULL AND state <> ''
                    GROUP BY state
                    ORDER BY n_businesses DESC
                    LIMIT :limit
                """

            try:
                return self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "State distribution primary query failed; falling back to count-only query: %s",
                    exc,
                )
                return self._query_df_stats(fallback_query, params)

        return self._cached_stats_value(cache_key, _compute)

    def get_default_focus_state(self) -> Optional[str]:
        """Return the default state filter (highest businesses, then reviews)."""
        cache_key = ("default_focus_state",)

        def _compute() -> Optional[str]:
            query = f"""
                SELECT
                    b.state,
                    COUNT(DISTINCT b.business_id) AS n_businesses,
                    COUNT(r.business_id) AS n_reviews
                FROM {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                LEFT JOIN {'reviews' if self.backend_type == 'cloudsql' else 'yelp_review'} r
                  ON r.business_id = b.business_id
                WHERE b.state IS NOT NULL AND b.state <> ''
                GROUP BY b.state
                ORDER BY n_businesses DESC, n_reviews DESC, b.state ASC
                LIMIT 1
            """
            try:
                df = self._query_df_stats(query, {})
            except Exception as exc:
                logger.warning("Default focus state query failed: %s", exc)
                return None
            if df.empty or "state" not in df.columns:
                return None
            state = df.iloc[0].get("state")
            return str(state).strip() if state is not None else None

        return self._cached_stats_value(cache_key, _compute)

    def get_activity_distributions(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        city: Optional[str] = None,
        pos_threshold: Optional[float] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return user and item activity distributions for positive interactions."""
        threshold = (
            float(pos_threshold)
            if pos_threshold is not None
            else float(self.config.get("pos_threshold", 4.0))
        )
        cache_key = ("activity_distributions", scope, state, city, threshold)

        def _compute() -> tuple[pd.DataFrame, pd.DataFrame]:
            r_stars = self._sql_cast_float("r.stars")
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=city, business_alias="b"
            )
            params["pos_threshold"] = threshold
            if where_sql:
                where_sql += f" AND {r_stars} >= :pos_threshold"
            else:
                where_sql = f" WHERE {r_stars} >= :pos_threshold"

            base = f"""
                FROM {'reviews' if self.backend_type == 'cloudsql' else 'yelp_review'} r
                JOIN {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                  ON r.business_id = b.business_id
                {where_sql}
            """

            user_query = f"""
                WITH user_activity AS (
                    SELECT r.user_id, COUNT(*) AS cnt
                    {base}
                    GROUP BY r.user_id
                )
                SELECT cnt, COUNT(*) AS num_users
                FROM user_activity
                GROUP BY cnt
                ORDER BY cnt
            """
            item_query = f"""
                WITH item_activity AS (
                    SELECT r.business_id, COUNT(*) AS cnt
                    {base}
                    GROUP BY r.business_id
                )
                SELECT cnt, COUNT(*) AS num_items
                FROM item_activity
                GROUP BY cnt
                ORDER BY cnt
            """
            try:
                return (
                    self._query_df_stats(user_query, params),
                    self._query_df_stats(item_query, params),
                )
            except Exception as exc:
                logger.warning(
                    "Activity distribution query failed; returning empty frames: %s",
                    exc,
                )
                return (
                    pd.DataFrame(columns=["cnt", "num_users"]),
                    pd.DataFrame(columns=["cnt", "num_items"]),
                )

        return self._cached_stats_value(cache_key, _compute)

    def get_sample_business_rows(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        city: Optional[str] = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Return sample business rows for table preview."""
        cache_key = ("sample_businesses", scope, state, city, limit)

        def _compute() -> pd.DataFrame:
            b_stars = self._sql_cast_float("b.stars")
            b_reviews = self._sql_cast_int("b.review_count")
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=city, business_alias="b"
            )
            params["limit"] = int(limit)
            query = f"""
                SELECT
                    b.business_id,
                    b.name,
                    b.city,
                    b.state,
                    {b_stars} AS stars,
                    {b_reviews} AS review_count,
                    b.categories
                FROM {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                {where_sql}
                ORDER BY {b_reviews} DESC NULLS LAST
                LIMIT :limit
            """
            try:
                return self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "Sample businesses query failed; returning empty frame: %s", exc
                )
                return pd.DataFrame(
                    columns=[
                        "business_id",
                        "name",
                        "city",
                        "state",
                        "stars",
                        "review_count",
                        "categories",
                    ]
                )

        return self._cached_stats_value(cache_key, _compute)

    def get_sample_review_rows(
        self,
        *,
        scope: str = "training",
        state: Optional[str] = None,
        city: Optional[str] = None,
        year_min: Optional[int] = None,
        year_max: Optional[int] = None,
        rating_min: Optional[float] = None,
        rating_max: Optional[float] = None,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Return sample review rows for table preview."""
        cache_key = (
            "sample_reviews",
            scope,
            state,
            city,
            year_min,
            year_max,
            rating_min,
            rating_max,
            limit,
        )

        def _compute() -> pd.DataFrame:
            r_stars = self._sql_cast_float("r.stars")
            where_sql, params = self._stats_scope_where(
                scope=scope, state=state, city=city, business_alias="b"
            )
            extra: List[str] = []
            if year_min is not None:
                extra.append("EXTRACT(YEAR FROM r.date) >= :year_min")
                params["year_min"] = int(year_min)
            if year_max is not None:
                extra.append("EXTRACT(YEAR FROM r.date) <= :year_max")
                params["year_max"] = int(year_max)
            if rating_min is not None:
                extra.append(f"{r_stars} >= :rating_min")
                params["rating_min"] = float(rating_min)
            if rating_max is not None:
                extra.append(f"{r_stars} <= :rating_max")
                params["rating_max"] = float(rating_max)

            if extra:
                if where_sql:
                    where_sql += " AND " + " AND ".join(extra)
                else:
                    where_sql = " WHERE " + " AND ".join(extra)

            params["limit"] = int(limit)
            query = f"""
                SELECT
                    r.review_id,
                    r.user_id,
                    r.business_id,
                    b.name AS business_name,
                    b.city,
                    b.state,
                    {r_stars} AS stars,
                    r.date,
                    r.text
                FROM {'reviews' if self.backend_type == 'cloudsql' else 'yelp_review'} r
                JOIN {'businesses' if self.backend_type == 'cloudsql' else 'yelp_business'} b
                  ON r.business_id = b.business_id
                {where_sql}
                ORDER BY r.date DESC
                LIMIT :limit
            """
            try:
                return self._query_df_stats(query, params)
            except Exception as exc:
                logger.warning(
                    "Sample reviews query failed; returning empty frame: %s", exc
                )
                return pd.DataFrame(
                    columns=[
                        "review_id",
                        "user_id",
                        "business_id",
                        "business_name",
                        "city",
                        "state",
                        "stars",
                        "date",
                        "text",
                    ]
                )

        return self._cached_stats_value(cache_key, _compute)

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

    def _normalize_user_matrices_payload(
        self,
        payload: Any,
        *,
        source_path: str,
        expected_run_id: Optional[str],
    ) -> Dict[str, Any]:
        """Validate strict run-scoped precomputed-matrix envelope.

        Required schema:
        - run_id
        - n_items
        - matrices (dict[user_id -> csr_matrix])

        Validation checks:
        - run_id matches active run
        - n_items matches expected model dimensionality
        - every matrix has shape (1, n_items)

        Returns:
            The validated matrices dictionary.
        """
        if not isinstance(payload, dict):
            raise RuntimeError(
                "Invalid precomputed matrices payload type at "
                f"{source_path}: expected dict, got {type(payload).__name__}."
            )

        matrices = payload.get("matrices")
        payload_run_id = payload.get("run_id")
        payload_n_items = payload.get("n_items")

        if matrices is None:
            # Legacy format: the pickle itself is the user_id -> csr_matrix mapping.
            # Older runs were written before the strict metadata envelope existed.
            legacy_matrices = payload
            if not legacy_matrices:
                raise RuntimeError(
                    f"Precomputed matrices are empty or invalid at {source_path}."
                )
            first_matrix = next(iter(legacy_matrices.values()))
            shape = getattr(first_matrix, "shape", None)
            if not shape or len(shape) != 2:
                raise RuntimeError(
                    "Legacy precomputed matrices payload is missing matrix shapes at "
                    f"{source_path}."
                )
            matrices = legacy_matrices
            payload_n_items = int(shape[1])
            payload_run_id = expected_run_id

        if not isinstance(matrices, dict) or not matrices:
            raise RuntimeError(
                f"Precomputed matrices are empty or invalid at {source_path}."
            )

        if (
            expected_run_id
            and payload_run_id
            and str(payload_run_id) != str(expected_run_id)
        ):
            raise RuntimeError(
                "Precomputed matrix run mismatch: "
                f"expected run_id={expected_run_id}, got run_id={payload_run_id} from {source_path}."
            )

        expected_n_items = self.expected_n_items
        if payload_n_items is not None:
            try:
                payload_n_items = int(payload_n_items)
            except (TypeError, ValueError):
                raise RuntimeError(
                    f"Invalid `n_items` in precomputed payload at {source_path}: {payload_n_items}"
                ) from None
            if expected_n_items is None:
                expected_n_items = payload_n_items
            elif payload_n_items != expected_n_items:
                raise RuntimeError(
                    "Precomputed matrix metadata mismatch: "
                    f"payload n_items={payload_n_items}, expected n_items={expected_n_items}, "
                    f"source={source_path}."
                )

        if expected_n_items is None:
            raise RuntimeError(
                "Cannot validate precomputed matrices: expected model n_items is unknown."
            )

        for user_id, matrix in matrices.items():
            shape = getattr(matrix, "shape", None)
            if not shape or len(shape) != 2:
                raise RuntimeError(
                    "Invalid matrix entry in precomputed payload: "
                    f"user={user_id}, shape={shape}, source={source_path}."
                )
            if int(shape[1]) != int(expected_n_items):
                raise RuntimeError(
                    "Precomputed matrix shape mismatch: "
                    f"user={user_id}, matrix.shape={shape}, expected second dimension={expected_n_items}, "
                    f"source={source_path}."
                )

        return matrices

    def _load_precomputed_user_matrices(self) -> Dict[str, Any]:
        """
        Load precomputed CSR matrices for all users.

        Strict contract:
        - Only run-scoped matrices are allowed.
        - Payload must include metadata envelope: run_id, n_items, matrices.
        - Matrix shape[1] must match model n_items.

        Returns:
            Dict[user_id, csr_matrix]
        """
        if not self.active_run_dir:
            if self.strict_run_artifacts:
                raise RuntimeError(
                    "Strict best-run mode requires an active run directory for precomputed user matrices."
                )
            return {}

        run_id = self.active_run_dir.name
        run_local_path = self.active_run_dir / "precomputed" / "user_csr_matrices.pkl"
        cloud_run_path = f"outputs/{run_id}/precomputed/user_csr_matrices.pkl"
        payload = None
        source_path = ""

        if self.active_run_dir:
            if getattr(self, "cloud_storage_helper", None) and len(run_id) == 15:
                try:
                    if self.cloud_storage_helper.exists(cloud_run_path):
                        payload = self.cloud_storage_helper.read_pickle(cloud_run_path)
                        if payload:
                            source_path = f"gs://{cloud_run_path}"
                    else:
                        self._warn_missing_cloud_path_once(cloud_run_path)
                except Exception as e:
                    logger.warning(
                        "Unexpected cloud read failure for active-run matrices at %s: %s",
                        cloud_run_path,
                        e,
                    )

            if payload is None and run_local_path.exists():
                try:
                    with open(run_local_path, "rb") as f:
                        payload = pickle.load(f)
                    source_path = str(run_local_path)
                except Exception as e:
                    logger.debug(
                        "Could not load active-run matrices from local path %s: %s",
                        run_local_path,
                        e,
                    )

        if payload is None:
            message = (
                "Missing required run-scoped precomputed user matrices. "
                f"Tried local path `{run_local_path}` and cloud path `{cloud_run_path}`."
            )
            if self.strict_run_artifacts:
                raise RuntimeError(message)
            # Non-strict mode keeps historical behavior (on-demand encoding).
            logger.debug("%s Falling back to on-demand encoding.", message)
            return {}

        matrices = self._normalize_user_matrices_payload(
            payload,
            source_path=source_path or str(run_local_path),
            expected_run_id=run_id,
        )
        logger.info(
            "✅ Loaded precomputed user matrices (%d users) from %s",
            len(matrices),
            source_path or run_local_path,
        )
        return matrices

    def get_precomputed_user_matrix(self, user_id: str):
        """
        Get precomputed CSR matrix for a user.

        Returns:
            csr_matrix or None if not found
        """
        if self._user_matrices_cache is None:
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
