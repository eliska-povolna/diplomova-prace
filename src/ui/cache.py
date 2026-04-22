"""Streamlit caching and session state management."""

import json
import os
import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

    # Define dummy decorator for non-Streamlit contexts
    def cache_resource(func):
        return func


import yaml

from src.ui.services import (
    DataService,
    InferenceService,
    LabelingService,
    WordcloudService,
)
from src.ui.services.coactivation_service import CoactivationService

logger = logging.getLogger(__name__)


def _get_cloud_storage_helper():
    """Return a CloudStorageHelper when GCS is configured, otherwise None."""
    bucket_name = os.getenv("GCS_BUCKET_NAME") or os.getenv("CLOUD_STORAGE_BUCKET")
    if not bucket_name:
        return None

    try:
        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        return CloudStorageHelper(bucket_name=bucket_name)
    except Exception as e:
        logger.debug(f"Cloud Storage helper unavailable: {e}")
        return None


def _find_latest_model_timestamp(cloud_storage) -> Optional[str]:
    """Find the newest timestamped model prefix in GCS."""
    try:
        blobs = cloud_storage.bucket.list_blobs(prefix="models/")
    except Exception as e:
        logger.debug(f"Failed to list GCS model artifacts: {e}")
        return None

    timestamps = set()
    for blob in blobs:
        parts = blob.name.split("/")
        if len(parts) >= 2 and len(parts[1]) == 15:
            timestamps.add(parts[1])

    if not timestamps:
        return None

    return sorted(timestamps)[-1]


def _download_gcs_file(cloud_storage, gcs_path: str, local_path: Path) -> bool:
    """Download one GCS object to a local path."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob = cloud_storage.bucket.blob(gcs_path)
        blob.download_to_filename(str(local_path))
        return local_path.exists()
    except Exception as e:
        logger.debug(
            f"Failed to download gs://{cloud_storage.bucket_name}/{gcs_path}: {e}"
        )
        return False


def _download_gcs_prefix(cloud_storage, gcs_prefix: str, local_root: Path) -> bool:
    """Download all objects under a GCS prefix into a local directory."""
    try:
        downloaded = False
        for gcs_path in cloud_storage.list_files(prefix=gcs_prefix):
            if not gcs_path.startswith(gcs_prefix):
                continue
            relative_path = gcs_path[len(gcs_prefix) :].lstrip("/")
            if not relative_path:
                continue
            destination = local_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            blob = cloud_storage.bucket.blob(gcs_path)
            blob.download_to_filename(str(destination))
            downloaded = True
        return downloaded
    except Exception as e:
        logger.debug(
            f"Failed to download GCS prefix gs://{cloud_storage.bucket_name}/{gcs_prefix}: {e}"
        )
        return False


def _build_experiment_results(
    manifest: dict,
    *,
    source: str,
    experiment_dir: Path,
) -> Optional[Dict]:
    """Normalize an experiment manifest into the structure expected by the UI."""
    runs = []

    for raw_run in manifest.get("runs", []):
        run = dict(raw_run)
        summary = run.get("summary")

        if not summary:
            summary_path_str = run.get("summary_path")
            if summary_path_str:
                summary_path = Path(summary_path_str)
                if summary_path.exists():
                    try:
                        with summary_path.open("r", encoding="utf-8") as f:
                            summary = json.load(f)
                    except Exception as e:
                        logger.debug(
                            "Could not load run summary from %s: %s", summary_path, e
                        )

        run["summary"] = summary
        runs.append(run)

    if not runs:
        return None

    selected_summary = runs[0].get("summary") or {}
    return {
        "summary": selected_summary,
        "ranking_metrics": selected_summary.get("ranking_metrics"),
        "source": source,
        "experiment": {
            "experiment_id": manifest.get("experiment_id"),
            "created": manifest.get("created"),
            "source_config": manifest.get("source_config"),
            "base_config": manifest.get("base_config"),
            "experiment_dir": str(experiment_dir),
            "manifest": manifest,
        },
        "runs": runs,
    }


def _load_local_experiment_results(outputs_base: Path) -> Optional[Dict]:
    latest_pointer = outputs_base / "LATEST_EXPERIMENT.txt"
    experiment_dir = None

    if latest_pointer.exists():
        try:
            experiment_dir = Path(latest_pointer.read_text(encoding="utf-8").strip())
        except Exception as e:
            logger.warning("Failed to read LATEST_EXPERIMENT.txt: %s", e)

    if not experiment_dir or not experiment_dir.exists():
        experiments_base = outputs_base / "experiments"
        if experiments_base.exists():
            experiment_dirs = [
                d
                for d in experiments_base.iterdir()
                if d.is_dir() and (d / "manifest.json").exists()
            ]
            if experiment_dirs:
                experiment_dir = sorted(experiment_dirs, key=lambda d: d.name)[-1]

    if not experiment_dir or not experiment_dir.exists():
        return None

    manifest_path = experiment_dir / "manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        logger.warning("Failed to load experiment manifest %s: %s", manifest_path, e)
        return None

    logger.info("✅ Loaded experiment manifest from local: %s", experiment_dir)
    return _build_experiment_results(
        manifest,
        source="Local Experiment",
        experiment_dir=experiment_dir,
    )


def _load_gcs_experiment_results() -> Optional[Dict]:
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if not gcs_bucket_name:
        return None

    try:
        from src.ui.services.cloud_storage_helper import CloudStorageHelper

        cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)
        blobs = cloud_storage.bucket.list_blobs(prefix="experiments/")
        experiment_ids = set()

        for blob in blobs:
            parts = blob.name.split("/")
            if len(parts) >= 3 and parts[2] == "manifest.json" and len(parts[1]) == 15:
                experiment_ids.add(parts[1])

        if not experiment_ids:
            return None

        latest_experiment_id = sorted(experiment_ids)[-1]
        manifest = cloud_storage.read_json(
            f"experiments/{latest_experiment_id}/manifest.json"
        )
        logger.info("✅ Loaded experiment manifest from GCS: %s", latest_experiment_id)
        return _build_experiment_results(
            manifest,
            source="GCS Experiment",
            experiment_dir=Path(f"experiments/{latest_experiment_id}"),
        )
    except Exception as e:
        logger.debug("GCS experiment results unavailable: %s", e)
        return None


def _load_summary_from_run_dir(run_dir: Path) -> Optional[Dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            summary_data = json.load(f)
        return {
            "summary": summary_data,
            "ranking_metrics": summary_data.get("ranking_metrics"),
            "source": f"Local Run: {run_dir.name}",
            "run_dir": str(run_dir),
        }
    except Exception as e:
        logger.warning("Failed to load run summary %s: %s", summary_path, e)
        return None


# Use streamlit's cache decorator if available, otherwise use dummy
if HAS_STREAMLIT:
    st_cache_resource = st.cache_resource
else:
    st_cache_resource = cache_resource


@st_cache_resource
def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML and flatten for UI services.

    Note: Cache is based on function signature only (not file mtime).
    To invalidate on config changes, restart the Streamlit app.
    """
    config_path = Path(config_path)
    project_root = config_path.parent.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    def _resolve_to_project_root(path_value: str) -> str:
        """Resolve path values relative to project root unless already absolute."""
        if not path_value:
            return ""
        candidate = Path(path_value)
        if candidate.is_absolute():
            return str(candidate)
        return str((project_root / candidate).resolve())

    # Flatten nested config structure for UI services
    config = {}
    config["project_root"] = str(project_root.resolve())
    config["config_path"] = str(config_path.resolve())

    # Data paths (from data section)
    if "data" in raw_config:
        config["duckdb_path"] = _resolve_to_project_root(
            raw_config["data"].get("db_path", "")
        )
        config["preprocess_dir"] = _resolve_to_project_root("data/preprocessed_yelp")

    # ELSA hyperparameters
    if "elsa" in raw_config:
        config["latent_dim"] = raw_config["elsa"].get("latent_dim", 512)
        config["device"] = raw_config["elsa"].get("device", "cpu")

    # SAE hyperparameters (k = sparsity level)
    if "sae" in raw_config:
        config["k"] = raw_config["sae"].get("k", 32)
        config["width_ratio"] = raw_config["sae"].get("width_ratio", 4)

    # Output & steering defaults
    config["steering_alpha"] = 0.3  # Default steering interpolation
    checkpoint_dir = raw_config.get("model", {}).get(
        "checkpoint_dir",
        raw_config.get("output", {}).get("base_dir", "outputs"),
    )
    config["model_checkpoint_dir"] = _resolve_to_project_root(checkpoint_dir)
    config["neuron_labels_path"] = _resolve_to_project_root(
        "outputs/neuron_labels.json"
    )

    # Compute n_items from database (apply same filters as training)
    # NOTE: This is informational only; the inference service reads n_items from checkpoint metadata
    config["n_items"] = None  # Will be read from checkpoint by inference service

    try:
        from src.ui.services.secrets_helper import get_cloudsql_config

        cloudsql_cfg = get_cloudsql_config()
        state_filter = raw_config.get("data", {}).get("state_filter")

        # Try Cloud SQL first
        if all(
            [
                cloudsql_cfg.get("instance"),
                cloudsql_cfg.get("database"),
                cloudsql_cfg.get("user"),
                cloudsql_cfg.get("password"),
            ]
        ):
            try:
                from src.ui.services.cloud_sql_helper import CloudSQLHelper

                sql_helper = CloudSQLHelper(
                    instance_connection_name=cloudsql_cfg["instance"],
                    database=cloudsql_cfg["database"],
                    user=cloudsql_cfg["user"],
                    password=cloudsql_cfg["password"],
                )
                with sql_helper.engine.connect() as conn:
                    if state_filter:
                        query = f"SELECT COUNT(*) FROM review WHERE state = '{state_filter}'"
                        logger.info(
                            f"   Counting items from Cloud SQL with state_filter='{state_filter}'..."
                        )
                    else:
                        query = "SELECT COUNT(*) FROM review"
                        logger.info(
                            "   Counting all items from Cloud SQL (no state filter)..."
                        )

                    result = conn.execute(query).scalar()
                    config["n_items"] = result if result else None
                    logger.info(
                        f"   Found {config['n_items']} items in Cloud SQL dataset"
                    )
            except Exception as e:
                logger.debug(f"Cloud SQL count failed: {e}, trying local DuckDB...")
                config["n_items"] = None

        # Fall back to local DuckDB if Cloud SQL unavailable
        if config["n_items"] is None:
            duckdb_path = Path(config["duckdb_path"])
            if duckdb_path.exists():
                import duckdb

                conn = duckdb.connect(str(duckdb_path))
                try:
                    if state_filter:
                        query = f"SELECT COUNT(*) FROM review WHERE state = '{state_filter}'"
                        logger.info(
                            f"   Counting items from DuckDB with state_filter='{state_filter}'..."
                        )
                    else:
                        query = "SELECT COUNT(*) FROM review"
                        logger.info(
                            "   Counting all items from DuckDB (no state filter)..."
                        )

                    result = conn.execute(query).fetchall()
                    config["n_items"] = result[0][0] if result else None
                    logger.info(f"   Found {config['n_items']} items in DuckDB dataset")
                except Exception as e:
                    logger.debug(f"Local DuckDB count failed: {e}")
                    config["n_items"] = None
                finally:
                    conn.close()
            else:
                logger.debug(
                    f"DuckDB not found at {duckdb_path}, will use checkpoint metadata"
                )

    except Exception as e:
        logger.debug(f"Could not count items from database: {e}")
        config["n_items"] = None  # Will be read from checkpoint by inference service

    # Include state_filter in config for DataService
    config["state_filter"] = raw_config.get("data", {}).get("state_filter")

    logger.info(f"✅ Loaded config from {config_path}")
    logger.info(
        f"   Device: {config['device']}, Latent dim: {config['latent_dim']}, SAE k: {config['k']}"
    )
    if config.get("state_filter"):
        logger.info(f"   State filter: {config['state_filter']}")
    return config


@st_cache_resource
def load_inference_service(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> InferenceService:
    """
    Load ELSA+SAE models once per session.

    Streamlit will call this once and reuse result across page refreshes.

    Model metadata (n_items, latent_dim, k, width_ratio) is read from checkpoint
    files, NOT from config. This ensures consistency regardless of how data is
    filtered or configured on the inference machine.
    """
    # Prefer the run selected in the Results page, then fall back to config.
    ckpt_base = (
        Path(selected_output_dir)
        if selected_output_dir
        else Path(config["model_checkpoint_dir"])
    )

    if selected_output_dir and not ckpt_base.exists():
        logger.warning("Selected checkpoint dir does not exist: %s", ckpt_base)
        ckpt_base = Path(config["model_checkpoint_dir"])

    # Find latest checkpoint - search in multiple patterns

    logger.info(f"Looking for checkpoints starting from: {ckpt_base}")

    checkpoint_dir = None

    # Strategy 1: Check if base path itself has checkpoint files
    if (ckpt_base / "elsa_best.pt").exists():
        # Find SAE checkpoint with flexible naming (sae_r*_k*_best.pt pattern)
        sae_files = list(ckpt_base.glob("sae_r*_k*_best.pt"))
        if sae_files or (ckpt_base / "sae_best.pt").exists():
            checkpoint_dir = ckpt_base
            logger.info(f"Found checkpoints directly in {ckpt_base}")

    # Strategy 2: Check for checkpoints/ subdirectory
    if not checkpoint_dir and (ckpt_base / "checkpoints").exists():
        ckpt_subdir = ckpt_base / "checkpoints"
        if (ckpt_subdir / "elsa_best.pt").exists():
            sae_files = list(ckpt_subdir.glob("sae_r*_k*_best.pt"))
            if sae_files or (ckpt_subdir / "sae_best.pt").exists():
                checkpoint_dir = ckpt_subdir
                logger.info(f"Found checkpoints in {checkpoint_dir}")

    # Strategy 3: Search timestamp subdirectories for checkpoints/
    if not checkpoint_dir and ckpt_base.exists():
        subdirs = sorted([d for d in ckpt_base.iterdir() if d.is_dir()], reverse=True)
        for subdir in subdirs:
            # Try subdir/checkpoints/ first (most common pattern)
            if (subdir / "checkpoints").exists():
                ckpt_subdir = subdir / "checkpoints"
                if (ckpt_subdir / "elsa_best.pt").exists():
                    sae_files = list(ckpt_subdir.glob("sae_r*_k*_best.pt"))
                    if sae_files or (ckpt_subdir / "sae_best.pt").exists():
                        checkpoint_dir = ckpt_subdir
                        logger.info(f"Found checkpoints in {checkpoint_dir}")
                        break
            # Try subdir directly as fallback
            if not checkpoint_dir and (subdir / "elsa_best.pt").exists():
                sae_files = list(subdir.glob("sae_r*_k*_best.pt"))
                if sae_files or (subdir / "sae_best.pt").exists():
                    checkpoint_dir = subdir
                    logger.info(f"Found checkpoints in {checkpoint_dir}")
                    break

        if not checkpoint_dir:
            cloud_storage = _get_cloud_storage_helper()
            if cloud_storage:
                latest_timestamp = _find_latest_model_timestamp(cloud_storage)
                if latest_timestamp:
                    gcs_checkpoint_prefix = f"models/{latest_timestamp}/checkpoints/"
                    temp_checkpoint_dir = Path(
                        tempfile.mkdtemp(prefix="diplomov_pr_ce_ckpts_")
                    )
                    if _download_gcs_prefix(
                        cloud_storage, gcs_checkpoint_prefix, temp_checkpoint_dir
                    ):
                        checkpoint_dir = temp_checkpoint_dir
                        logger.info(
                            f"✅ Loaded checkpoints from GCS: gs://{cloud_storage.bucket_name}/{gcs_checkpoint_prefix}"
                        )

        if not checkpoint_dir:
            logger.error("No checkpoint subdirs found")
            if ckpt_base.exists():
                logger.error(f"Searched: {[d.name for d in subdirs]}")
                for subdir in subdirs[:3]:  # Show contents of first 3 dirs
                    contents = list(subdir.iterdir())
                    logger.error(f"  {subdir.name}/: {[c.name for c in contents]}")

            error_msg = (
                "❌ Model Checkpoints Not Found\n\n"
                f"Could not find `elsa_best.pt` or `sae_best.pt` in:\n"
                f"`{ckpt_base}`\n\n"
                "**On Streamlit Cloud:**\n"
                "The app now expects checkpoints in GCS under `models/<timestamp>/checkpoints/`.\n\n"
                "**To fix:**\n"
                "1. Ensure training uploads checkpoints to GCS\n"
                "2. Set `GCS_BUCKET_NAME` in Cloud secrets\n"
                "3. Redeploy the app\n\n"
                "**Local development:**\n"
                "If running locally, ensure the `outputs/` directory has trained model files."
            )
            if HAS_STREAMLIT:
                st.error(error_msg)
            raise RuntimeError(f"No model checkpoints found in {ckpt_base}")

    if not checkpoint_dir:
        error_msg = (
            "❌ Model Checkpoint Directory Not Found\n\n"
            f"The configured checkpoint directory does not exist:\n"
            f"`{ckpt_base}`\n\n"
            "**To fix:**\n"
            f"1. Check that `configs/default.yaml` has correct `model_checkpoint_dir`\n"
            f"2. Ensure trained model files are present in `outputs/*/checkpoints/`"
        )
        if HAS_STREAMLIT:
            st.error(error_msg)
        raise RuntimeError(
            f"Checkpoint directory does not exist or has no valid checkpoints: {ckpt_base}"
        )

    # Find SAE checkpoint with flexible naming
    sae_ckpt = None
    sae_files = list(checkpoint_dir.glob("sae_r*_k*_best.pt"))
    if sae_files:
        sae_ckpt = sae_files[0]  # Use first match if multiple exist
    elif (checkpoint_dir / "sae_best.pt").exists():
        sae_ckpt = checkpoint_dir / "sae_best.pt"

    if not sae_ckpt:
        raise RuntimeError(f"SAE checkpoint not found in {checkpoint_dir}")

    # Load models
    elsa_ckpt = checkpoint_dir / "elsa_best.pt"

    logger.info(f"Loading ELSA from {elsa_ckpt}")
    logger.info(f"Loading SAE from {sae_ckpt}")

    # Load labels service
    labels = load_labeling_service(config, selected_output_dir=selected_output_dir)

    # Load data service to pass to inference service
    data_service = load_data_service(config)

    service = InferenceService(
        elsa_ckpt, sae_ckpt, config, labels=labels, data_service=data_service
    )
    if HAS_STREAMLIT:
        if not hasattr(st.session_state, "_startup_diagnostics"):
            st.session_state._startup_diagnostics = {}
        st.session_state._startup_diagnostics["models_loaded"] = True
    logger.info("✅ Models loaded successfully")
    return service


@st_cache_resource
def load_data_service(config: Dict):
    """
    Load POI data once per session.

    Supports both cloud backend (Cloud SQL + Cloud Storage) and local backend.

    The unified DataService automatically detects which backend to use:
    - If Cloud SQL credentials (CLOUDSQL_INSTANCE, etc.) are present -> Uses Cloud SQL
    - Otherwise -> Uses local DuckDB tables

    The USE_CLOUD_STORAGE env var can force local-only mode if set to "false".

    Falls back gracefully if local data isn't available (for Streamlit Cloud).
    """

    logger.info("🔄 Initializing Data Service...")

    # Check if local data files exist before trying to initialize
    duckdb_path = Path(config["duckdb_path"])
    data_available_locally = duckdb_path.exists()

    # Path to item2index mapping - search in multiple locations
    # Priority:
    # 1. Latest training output: outputs/YYYYMMDD_HHMMSS/mappings/item2index.pkl (FILTERED - post-k-core!)
    # 2. Fallback: outputs/YYYYMMDD_HHMMSS/mappings/business2index_universal.pkl (universal - less ideal)
    # 3. In parquet directory (legacy location)
    # 4. Hardcoded path for backward compatibility
    item2index_path = None
    item2index_candidates = []

    # Strategy 1A: Find latest FILTERED mapping (post-k-core) - PRIORITIZE THIS
    outputs_base = Path(__file__).parent.parent.parent / "outputs"
    if outputs_base.exists():
        # List timestamp directories, sorted newest first
        timestamp_dirs = sorted(
            [d for d in outputs_base.iterdir() if d.is_dir() and len(d.name) == 15],
            reverse=True,
        )  # YYYYMMDD_HHMMSS format
        for ts_dir in timestamp_dirs:
            # Try filtered mapping first (what model actually uses)
            candidate = ts_dir / "mappings" / "item2index.pkl"
            if candidate.exists():
                item2index_path = candidate
                logger.info(
                    f"✓ Found filtered item2index.pkl (post-k-core): {item2index_path}"
                )
                break
            item2index_candidates.append(candidate)

    # Strategy 1B: Fallback to universal mapping (backward compat, but NOT ideal for index alignment)
    if not item2index_path and outputs_base.exists():
        timestamp_dirs = sorted(
            [d for d in outputs_base.iterdir() if d.is_dir() and len(d.name) == 15],
            reverse=True,
        )
        for ts_dir in timestamp_dirs:
            candidate = ts_dir / "mappings" / "business2index_universal.pkl"
            if candidate.exists():
                item2index_path = candidate
                logger.warning(
                    f"⚠️ Using universal mapping (may have index mismatch): {item2index_path}"
                )
                break
            item2index_candidates.append(candidate)

    # Strategy 2: Try in preprocessed directory
    if not item2index_path:
        preprocess_dir = Path(config.get("preprocess_dir", ""))
        for candidate in [
            preprocess_dir / "item2index.pkl",  # Preprocessed data directory
            preprocess_dir.parent / "item2index.pkl",  # Parent directory
        ]:
            item2index_candidates.append(candidate)
            if candidate.exists():
                item2index_path = candidate
                logger.info(f"✓ Found item2index at: {item2index_path}")
                break

    # Strategy 3: Try hardcoded legacy path
    if not item2index_path:
        candidate = (
            Path(__file__).parent.parent.parent
            / "data"
            / "processed_yelp_easystudy"
            / "item2index.pkl"
        )
        item2index_candidates.append(candidate)
        if candidate.exists():
            item2index_path = candidate
            logger.info(f"✓ Found item2index at: {item2index_path}")

    if not item2index_path:
        logger.warning(
            f"⚠️ item2index.pkl/business2index_universal.pkl not found in any location. "
            f"Tried: {[str(c) for c in item2index_candidates[:5]]}"
        )
        # Don't fail - item2index is optional, app can work without it

    # Path to local photos folder (support common folder naming variants)
    project_root = Path(__file__).parent.parent.parent
    photo_candidates = [
        project_root / "yelp_photos",
        project_root / "Yelp-Photos",
    ]
    local_photos_path = next((p for p in photo_candidates if p.exists()), None)

    # Check for Cloud SQL credentials
    from src.ui.services.secrets_helper import get_cloudsql_config

    cloudsql_config = get_cloudsql_config()
    cloudsql_available = all(
        [
            cloudsql_config.get("instance"),
            cloudsql_config.get("database"),
            cloudsql_config.get("user"),
            cloudsql_config.get("password"),
        ]
    )

    if not data_available_locally and not cloudsql_available:
        # Neither local data nor Cloud SQL available
        error_msg = (
            "❌ Data Backend Not Available\n\n"
            "Neither local data files nor Cloud SQL credentials found.\n\n"
            "**To fix on Streamlit Cloud:**\n"
            "1. Go to your app settings → **Secrets**\n"
            "2. Add Cloud SQL configuration:\n"
            "   - `CLOUDSQL_INSTANCE`: your-project:region:instance\n"
            "   - `CLOUDSQL_DATABASE`: postgres\n"
            "   - `CLOUDSQL_USER`: postgres\n"
            "   - `CLOUDSQL_PASSWORD`: your-password\n\n"
            "**To fix locally:**\n"
            "Ensure `configs/default.yaml` has correct paths to:\n"
            f"- DuckDB: {duckdb_path}\n"
            "- Optional preprocessed mappings in data/preprocessed_yelp/"
        )
        if HAS_STREAMLIT:
            st.error(error_msg)
        logger.error(error_msg)
        raise RuntimeError(
            "Data backend not available. Configure Cloud SQL or ensure local data exists."
        )

    # Initialize DataService (will try Cloud SQL first, then fallback to local)
    try:
        service = DataService(
            duckdb_path=duckdb_path,
            config=config,
            item2index_path=item2index_path,
            local_photos_dir=local_photos_path,
        )
    except FileNotFoundError as e:
        error_msg = (
            f"❌ Local data files not found: {e}\n\n"
            "Local data path is not accessible. This is expected on Streamlit Cloud.\n\n"
            "**To use local data:**\n"
            "Run locally with:\n"
            "`streamlit run src/ui/main.py`\n\n"
            "**To use Streamlit Cloud:**\n"
            "Configure Cloud SQL credentials in Streamlit Cloud Secrets."
        )
        if HAS_STREAMLIT:
            st.error(error_msg)
        logger.error(error_msg)
        raise RuntimeError(f"Data not available: {e}") from e

    # Report which backend is being used
    backend_info = getattr(service, "backend_type", "unknown")
    if backend_info == "cloudsql":
        if HAS_STREAMLIT:
            if not hasattr(st.session_state, "_startup_diagnostics"):
                st.session_state._startup_diagnostics = {}
            st.session_state._startup_diagnostics["backend"] = "Cloud SQL"
        logger.info("✅ Data Service using Cloud SQL backend")
    else:
        if HAS_STREAMLIT:
            if not hasattr(st.session_state, "_startup_diagnostics"):
                st.session_state._startup_diagnostics = {}
            st.session_state._startup_diagnostics["backend"] = (
                f"DuckDB ({service.num_pois} POIs)"
            )
            if local_photos_path:
                st.session_state._startup_diagnostics["photos"] = (
                    f"Local ({local_photos_path})"
                )
        logger.info(f"✅ Loaded {service.num_pois} POIs (Local Backend - DuckDB)")
        if local_photos_path:
            logger.info(f"📷 Local photos enabled: {local_photos_path}")

    return service


@st_cache_resource
def get_precomputed_cache_dir() -> Optional[Path]:
    """Detect if precomputed UI cache exists and return its path.

    Looks for precomputed_ui_cache/neuron_wordclouds/ in the latest outputs/*/ directories.
    Returns the neuron_wordclouds subdirectory or None if not found (app will compute on-demand).
    """
    project_root = Path(__file__).parent.parent.parent
    outputs_dir = project_root / "outputs"

    if not outputs_dir.exists():
        return None

    # Find all output directories, sorted by modification time
    import os

    output_dirs = sorted(
        [d for d in outputs_dir.iterdir() if d.is_dir()],
        key=lambda d: os.path.getmtime(d),
        reverse=True,
    )

    for output_dir in output_dirs:
        cache_dir = output_dir / "precomputed_ui_cache" / "neuron_wordclouds"
        if cache_dir.exists():
            logger.info(f"✓ Found precomputed cache at {cache_dir}")
            return cache_dir

    logger.info("No precomputed cache found (app will compute on-demand)")
    return None


@st_cache_resource
def load_labeling_service(
    config: Dict,
    data_service=None,
    selected_output_dir: Optional[str] = None,
) -> LabelingService:
    """
    Load neuron labeling service.

    Labels are lazy-loaded on first access (no startup delay).
    If no LLM provider is available (no API keys), uses basic pre-computed labels.
    NeuronInterpreter is only imported if LLM providers are configured.
    """
    # Prefer the run selected in the Results page, then fall back to latest run.
    output_dir = Path(selected_output_dir) if selected_output_dir else None

    if output_dir and not output_dir.exists():
        logger.warning("Selected output dir does not exist: %s", output_dir)
        output_dir = None

    latest_run_path = Path("outputs") / "LATEST_RUN.txt"

    if latest_run_path.exists():
        try:
            with open(latest_run_path, "r", encoding="utf-8") as f:
                output_dir_str = f.read().strip()
            output_dir = Path(output_dir_str)
            logger.debug(f"Found latest output dir: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to read LATEST_RUN.txt: {e}")

    # Fallback: find most recent timestamped directory
    if not output_dir or not output_dir.exists():
        outputs_base = Path("outputs")
        if outputs_base.exists():
            timestamped_dirs = [
                d
                for d in outputs_base.iterdir()
                if d.is_dir() and len(d.name) == 15  # Format: YYYYMMDD_HHMMSS
            ]
            if timestamped_dirs:
                output_dir = sorted(timestamped_dirs)[-1]
                logger.debug(f"Using most recent output dir: {output_dir}")

    if not output_dir or not output_dir.exists():
        cloud_storage = _get_cloud_storage_helper()
        if cloud_storage:
            latest_timestamp = _find_latest_model_timestamp(cloud_storage)
            if latest_timestamp:
                temp_root = Path(tempfile.mkdtemp(prefix="diplomov_pr_ce_labels_"))
                gcs_prefix = f"models/{latest_timestamp}/neuron_interpretations/"
                if _download_gcs_prefix(
                    cloud_storage, gcs_prefix, temp_root / "neuron_interpretations"
                ):
                    output_dir = temp_root
                    logger.info(
                        f"✅ Loaded labels from GCS run: gs://{cloud_storage.bucket_name}/{gcs_prefix}"
                    )

    # Use the neuron_interpretations directory, which stores labels_*.pkl
    labels_path = None
    if output_dir:
        interpretations_dir = output_dir / "neuron_interpretations"
        method_files = sorted(interpretations_dir.glob("labels_*.pkl"))
        if method_files:
            labels_path = interpretations_dir
            logger.info(f"✅ Using labels from run directory: {labels_path}")
        else:
            candidate = interpretations_dir / "neuron_labels.json"
            if candidate.exists():
                labels_path = candidate
                logger.info(f"✅ Using labels from: {labels_path}")

    # Fallback to default path
    if not labels_path:
        labels_path = Path("outputs") / "neuron_labels.json"
        logger.debug(f"Using fallback labels path: {labels_path}")

    # Check if LLM providers are configured before importing NeuronInterpreter
    interpreter = None
    from src.ui.services.secrets_helper import get_gemini_api_key
    import os

    has_gemini = bool(get_gemini_api_key())

    if has_gemini:
        # Only import NeuronInterpreter if we have Gemini API key configured
        try:
            from src.interpret.neuron_interpreter import NeuronInterpreter

            try:
                interpreter = NeuronInterpreter()  # Uses GOOGLE_API_KEY
                logger.info("✅ LLM interpreter available (Gemini API)")
            except ValueError as e:
                logger.debug(f"LLM interpreter not available: {e}")
                interpreter = None
        except ImportError as e:
            logger.debug(f"NeuronInterpreter not available: {e}")
            interpreter = None
    else:
        logger.debug("No GOOGLE_API_KEY configured. Using basic pre-computed labels.")

    service = LabelingService(
        labels_json_path=labels_path,
        interpreter=interpreter,
        config=config,
        data_service=data_service,
    )
    return service


@st_cache_resource
def load_wordcloud_service(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> "WordcloudService":
    """
    Load wordcloud service for neuron feature visualization.

    Provides wordcloud generation from neuron category data.
    """
    try:
        from src.ui.services import WordcloudService
    except ImportError:
        logger.error("WordcloudService not available")
        return None

    # Prefer the run selected in the Results page, then fall back to latest run.
    output_dir = Path(selected_output_dir) if selected_output_dir else None

    if output_dir and not output_dir.exists():
        logger.warning("Selected output dir does not exist: %s", output_dir)
        output_dir = None

    latest_run_path = Path("outputs") / "LATEST_RUN.txt"

    if latest_run_path.exists():
        try:
            with open(latest_run_path, "r") as f:
                output_dir_str = f.read().strip()
            output_dir = Path(output_dir_str)
            logger.debug(f"Found latest output dir from LATEST_RUN.txt: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to read LATEST_RUN.txt: {e}")

    # Fallback: find most recent timestamped directory
    if not output_dir or not output_dir.exists():
        outputs_base = Path("outputs")
        if outputs_base.exists():
            timestamped_dirs = [
                d
                for d in outputs_base.iterdir()
                if d.is_dir() and len(d.name) == 15  # Format: YYYYMMDD_HHMMSS
            ]
            if timestamped_dirs:
                output_dir = sorted(timestamped_dirs)[-1]  # Most recent
                logger.debug(f"Using most recent output dir: {output_dir}")

    if not output_dir or not output_dir.exists():
        cloud_storage = _get_cloud_storage_helper()
        if cloud_storage:
            latest_timestamp = _find_latest_model_timestamp(cloud_storage)
            if latest_timestamp:
                temp_root = Path(tempfile.mkdtemp(prefix="diplomov_pr_ce_wordclouds_"))
                labels_prefix = f"models/{latest_timestamp}/neuron_interpretations/"
                if _download_gcs_prefix(
                    cloud_storage,
                    labels_prefix,
                    temp_root / "neuron_interpretations",
                ) and _download_gcs_file(
                    cloud_storage,
                    f"models/{latest_timestamp}/neuron_category_metadata.json",
                    temp_root / "neuron_category_metadata.json",
                ):
                    output_dir = temp_root
                    logger.info(
                        f"✅ Loaded interpretability artifacts from GCS run: gs://{cloud_storage.bucket_name}/models/{latest_timestamp}/"
                    )

    # Look for label and metadata files
    labels_path = None
    metadata_path = None

    if output_dir:
        interpretations_dir = output_dir / "neuron_interpretations"
        labels_path = interpretations_dir / "neuron_labels.json"
        metadata_path = output_dir / "neuron_category_metadata.json"

        if labels_path.exists():
            logger.info(f"Found labels at: {labels_path}")
        else:
            logger.warning(f"Labels not found at: {labels_path}")
            labels_path = None

        if metadata_path.exists():
            logger.info(f"Found metadata at: {metadata_path}")
        else:
            logger.warning(f"Metadata not found at: {metadata_path}")
            metadata_path = None

    service = WordcloudService(
        category_metadata_path=metadata_path if metadata_path else None,
        labels_path=labels_path,
    )

    logger.info("✅ Wordcloud service initialized")
    return service


@st_cache_resource
def load_coactivation_service(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> Optional["CoactivationService"]:
    """
    Load co-activation service for neuron relationship visualization.

    Provides co-activation relationships between neurons.
    """
    # Prefer the run selected in the Results page, then fall back to latest run.
    output_dir = Path(selected_output_dir) if selected_output_dir else None

    if output_dir and not output_dir.exists():
        logger.warning("Selected output dir does not exist: %s", output_dir)
        output_dir = None

    latest_run_path = Path("outputs") / "LATEST_RUN.txt"

    if latest_run_path.exists():
        try:
            with open(latest_run_path, "r") as f:
                output_dir_str = f.read().strip()
            output_dir = Path(output_dir_str)
            logger.debug(f"Found latest output dir from LATEST_RUN.txt: {output_dir}")
        except Exception as e:
            logger.warning(f"Failed to read LATEST_RUN.txt: {e}")

    # Fallback: find most recent timestamped directory
    if not output_dir or not output_dir.exists():
        outputs_base = Path("outputs")
        if outputs_base.exists():
            timestamped_dirs = [
                d
                for d in outputs_base.iterdir()
                if d.is_dir() and len(d.name) == 15  # Format: YYYYMMDD_HHMMSS
            ]
            if timestamped_dirs:
                output_dir = sorted(timestamped_dirs)[-1]  # Most recent
                logger.debug(f"Using most recent output dir: {output_dir}")

    if not output_dir or not output_dir.exists():
        cloud_storage = _get_cloud_storage_helper()
        if cloud_storage:
            latest_timestamp = _find_latest_model_timestamp(cloud_storage)
            if latest_timestamp:
                temp_root = Path(
                    tempfile.mkdtemp(prefix="diplomov_pr_ce_coactivation_")
                )
                if _download_gcs_file(
                    cloud_storage,
                    f"models/{latest_timestamp}/neuron_coactivation.json",
                    temp_root / "neuron_coactivation.json",
                ):
                    output_dir = temp_root
                    logger.info(
                        f"✅ Loaded coactivation data from GCS run: gs://{cloud_storage.bucket_name}/models/{latest_timestamp}/"
                    )

    # Look for coactivation file
    coactivation_path = None

    if output_dir:
        coactivation_path = output_dir / "neuron_coactivation.json"

        if coactivation_path.exists():
            logger.info(f"Found co-activation data at: {coactivation_path}")
        else:
            logger.debug(f"Co-activation file not found at: {coactivation_path}")
            coactivation_path = None

    service = CoactivationService(coactivation_path=coactivation_path)
    logger.info("✅ Co-activation service initialized")
    return service


@st_cache_resource
def load_training_results(
    config: Dict,
    selected_output_dir: Optional[str] = None,
) -> Optional[Dict]:
    """
    Load training results from GCS first, then local filesystem for offline use.

    Tries in order:
    1. Latest results from GCS (if GCS_BUCKET_NAME configured)
    2. Latest results from local outputs/ directory as an offline fallback

    Returns:
        Dict with 'summary' and 'ranking_metrics' keys, or None if not found
    """
    import os

    results = {"summary": None, "ranking_metrics": None, "source": None}

    if selected_output_dir:
        selected_run = _load_summary_from_run_dir(Path(selected_output_dir))
        if selected_run:
            return selected_run

    # Prefer latest experiment sweep if one exists.
    experiment_results = _load_gcs_experiment_results()
    if not experiment_results:
        experiment_results = _load_local_experiment_results(Path("outputs"))

    if experiment_results:
        return experiment_results

    # Try GCS first
    gcs_bucket_name = os.getenv("GCS_BUCKET_NAME")
    if gcs_bucket_name:
        try:
            from src.ui.services.cloud_storage_helper import CloudStorageHelper

            cloud_storage = CloudStorageHelper(bucket_name=gcs_bucket_name)

            # List all model result directories in GCS
            models_prefix = "models/"
            blobs = cloud_storage.bucket.list_blobs(prefix=models_prefix)
            timestamps = set()
            for blob in blobs:
                # Extract timestamp from path like "models/YYYYMMDD_HHMMSS/summary.json"
                parts = blob.name.split("/")
                if len(parts) >= 2:
                    timestamp = parts[1]
                    if len(timestamp) == 15:  # YYYYMMDD_HHMMSS format
                        timestamps.add(timestamp)

            if timestamps:
                latest_timestamp = sorted(timestamps)[-1]
                logger.info(f"Found latest training results in GCS: {latest_timestamp}")

                # Download summary
                try:
                    summary_data = cloud_storage.read_json(
                        f"models/{latest_timestamp}/summary.json"
                    )
                    results["summary"] = summary_data
                    results["ranking_metrics"] = summary_data.get("ranking_metrics")
                    results["source"] = "GCS"
                    logger.info(
                        f"✅ Loaded training results from GCS (timestamp: {latest_timestamp})"
                    )
                    return results
                except Exception as e:
                    logger.warning(f"Failed to download summary from GCS: {e}")

        except Exception as e:
            logger.debug(f"GCS results unavailable: {e}")

    # Fall back to local filesystem
    outputs_base = Path("outputs")
    if outputs_base.exists():
        timestamped_dirs = [
            d
            for d in outputs_base.iterdir()
            if d.is_dir() and len(d.name) == 15  # Format: YYYYMMDD_HHMMSS
        ]

        if timestamped_dirs:
            output_dir = sorted(timestamped_dirs)[-1]  # Most recent
            summary_path = output_dir / "summary.json"

            if summary_path.exists():
                import json

                try:
                    with open(summary_path) as f:
                        summary_data = json.load(f)
                    results["summary"] = summary_data
                    results["ranking_metrics"] = summary_data.get("ranking_metrics")
                    results["source"] = "Local"
                    logger.info(
                        f"✅ Loaded training results from local (timestamp: {output_dir.name})"
                    )
                    return results
                except Exception as e:
                    logger.warning(f"Failed to load local summary: {e}")

    logger.warning("No training results found (GCS or local)")
    return None


@st_cache_resource
def load_semantic_search_model():
    """Load sentence transformer model for semantic search (cached at startup).

    This model is cached as a resource, so it's loaded once at startup
    and reused for all semantic search queries. Uses the lightweight
    all-MiniLM-L6-v2 model (~22MB) for fast inference.

    Returns:
        SentenceTransformer model or None if loading fails
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Semantic search model loaded and cached")
        return model
    except Exception as e:
        logger.warning(f"Could not load semantic search model: {e}")
        return None


def cache_all_label_embeddings(labels_service, max_neuron: int):
    """Cache all label embeddings in session state for efficient semantic search.

    Encodes all feature labels once using batch encoding and stores them
    in session_state to avoid re-encoding on every search.

    Args:
        labels_service: LabelingService for fetching labels
        max_neuron: Maximum neuron index (0 to max_neuron inclusive)
    """
    cache_key = "all_label_embeddings"

    # Return cached version if available
    if cache_key in st.session_state and st.session_state[cache_key] is not None:
        logger.info(
            f"✅ Using cached label embeddings: {len(st.session_state[cache_key])} embeddings"
        )
        return st.session_state[cache_key]

    try:
        semantic_model = load_semantic_search_model()
        if semantic_model is None:
            logger.warning("Cannot cache embeddings: semantic model is None")
            return None

        logger.info(
            f"Encoding all {max_neuron + 1} feature labels for semantic search..."
        )

        # Get all labels and their indices
        all_labels = []
        label_indices = []
        for idx in range(max_neuron + 1):
            try:
                label = labels_service.get_label(idx)
                all_labels.append(label)
                label_indices.append(idx)
            except Exception as e:
                logger.debug(f"Failed to get label for {idx}: {e}")

        logger.info(f"Found {len(all_labels)} labels to encode")

        if not all_labels:
            logger.warning("No labels found for embedding")
            return None

        # Batch encode all labels at once (much faster than one-by-one)
        logger.info(f"Batch encoding {len(all_labels)} labels...")
        label_embeddings = semantic_model.encode(all_labels, show_progress_bar=False)
        logger.info(
            f"Batch encoding complete. Shape: {label_embeddings.shape}, dtype: {label_embeddings.dtype}"
        )

        # Create dictionary mapping label index to embedding
        # Keep as numpy arrays for consistency and easy computation
        embeddings_dict = {
            idx: embedding for idx, embedding in zip(label_indices, label_embeddings)
        }

        # Cache in session state
        st.session_state[cache_key] = embeddings_dict
        logger.info(
            f"✅ Cached {len(embeddings_dict)} label embeddings in session state"
        )
        return embeddings_dict

    except Exception as e:
        logger.error(f"Failed to cache label embeddings: {e}", exc_info=True)
        return None


def init_session_state():
    """
    Initialize Streamlit session state.

    Called once per session to set up variables persisted across reruns.
    """
    if not HAS_STREAMLIT:
        logger.debug("Streamlit not available, skipping session state init")
        return

    if "current_user_id" not in st.session_state:
        st.session_state.current_user_id = None

    if "current_recommendations" not in st.session_state:
        st.session_state.current_recommendations = []

    if "steering_modified" not in st.session_state:
        st.session_state.steering_modified = False

    if "user_history" not in st.session_state:
        st.session_state.user_history = {}


def get_services() -> tuple:
    """
    Retrieve cached services from session state.

    Must be called after main.py initializes them.

    Returns:
        (inference, data, labels)
    """
    if not HAS_STREAMLIT:
        raise RuntimeError(
            "Streamlit not available - cannot retrieve services from session"
        )

    inference = st.session_state.get("inference")
    data = st.session_state.get("data")
    labels = st.session_state.get("labels")

    if not all([inference, data, labels]):
        st.error("Services not initialized. Check main.py setup.")
        st.stop()

    return inference, data, labels
