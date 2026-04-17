"""Streamlit caching and session state management."""

import logging
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
        config["parquet_dir"] = _resolve_to_project_root(
            raw_config["data"].get("parquet_dir", "")
        )

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
def load_inference_service(config: Dict) -> InferenceService:
    """
    Load ELSA+SAE models once per session.

    Streamlit will call this once and reuse result across page refreshes.

    Model metadata (n_items, latent_dim, k, width_ratio) is read from checkpoint
    files, NOT from config. This ensures consistency regardless of how data is
    filtered or configured on the inference machine.
    """
    # Find latest checkpoint - search in multiple patterns
    ckpt_base = Path(config["model_checkpoint_dir"])

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
            logger.error("No checkpoint subdirs found")
            logger.error(f"Searched: {[d.name for d in subdirs]}")
            for subdir in subdirs[:3]:  # Show contents of first 3 dirs
                contents = list(subdir.iterdir())
                logger.error(f"  {subdir.name}/: {[c.name for c in contents]}")

            error_msg = (
                "❌ Model Checkpoints Not Found\n\n"
                f"Could not find `elsa_best.pt` or `sae_best.pt` in:\n"
                f"`{ckpt_base}`\n\n"
                "**On Streamlit Cloud:**\n"
                "This error occurs when model checkpoint files aren't committed to Git.\n"
                "Checkpoint files must be explicitly added to the repository.\n\n"
                "**To fix:**\n"
                "1. Ensure checkpoint files exist locally in `outputs/*/checkpoints/`\n"
                "2. Add them to Git: `git add outputs/**/checkpoints/ --force`\n"
                "3. Commit and push: `git push`\n"
                "4. Redeploy on Streamlit Cloud\n\n"
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
    labels = load_labeling_service(config)

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
    - Otherwise -> Uses local DuckDB + Parquet files

    The USE_CLOUD_STORAGE env var can force local-only mode if set to "false".

    Falls back gracefully if local data isn't available (for Streamlit Cloud).
    """

    logger.info("🔄 Initializing Data Service...")

    # Path to UNIVERSAL item2index mapping (all ~17k businesses)
    # This preserves user interaction history before filtering
    item2index_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "processed_yelp_easystudy"
        / "item2index.pkl"
    )

    # Path to local photos folder (support common folder naming variants)
    project_root = Path(__file__).parent.parent.parent
    photo_candidates = [
        project_root / "yelp_photos",
        project_root / "Yelp-Photos",
    ]
    local_photos_path = next((p for p in photo_candidates if p.exists()), None)

    # Check if local data files exist before trying to initialize
    duckdb_path = Path(config["duckdb_path"])
    parquet_dir = Path(config["parquet_dir"])
    data_available_locally = duckdb_path.exists() and parquet_dir.exists()

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
            f"- Parquet: {parquet_dir}"
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
            parquet_dir=parquet_dir,
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
            st.session_state._startup_diagnostics[
                "backend"
            ] = f"DuckDB ({service.num_pois} POIs)"
            if local_photos_path:
                st.session_state._startup_diagnostics[
                    "photos"
                ] = f"Local ({local_photos_path})"
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
def load_labeling_service(config: Dict, data_service=None) -> LabelingService:
    """
    Load neuron labeling service.

    Labels are lazy-loaded on first access (no startup delay).
    If no LLM provider is available (no API keys), uses basic pre-computed labels.
    NeuronInterpreter is only imported if LLM providers are configured.
    """
    # Find latest timestamped output directory
    latest_run_path = Path("outputs") / "LATEST_RUN.txt"
    output_dir = None

    if latest_run_path.exists():
        try:
            with open(latest_run_path, "r") as f:
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

    # Use labels from output dir if available
    labels_path = None
    if output_dir:
        candidate = output_dir / "neuron_labels.json"
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
    has_github = bool(os.environ.get("GITHUB_TOKEN"))

    if has_gemini or has_github:
        # Only import NeuronInterpreter if we have API keys configured
        try:
            from src.interpret.neuron_interpreter import NeuronInterpreter

            try:
                interpreter = NeuronInterpreter()  # Auto-detect provider
                logger.info(f"✅ LLM provider available: {interpreter.provider}")
            except ValueError as e:
                logger.debug(f"LLM provider not available: {e}")
                interpreter = None
        except ImportError as e:
            logger.debug(f"NeuronInterpreter not available: {e}")
            interpreter = None
    else:
        logger.debug(
            "No LLM API keys configured (GITHUB_TOKEN, GOOGLE_API_KEY). "
            "Using basic pre-computed labels."
        )

    service = LabelingService(
        labels_json_path=labels_path,
        interpreter=interpreter,
        config=config,
        data_service=data_service,
    )
    return service


@st_cache_resource
def load_wordcloud_service(config: Dict) -> "WordcloudService":
    """
    Load wordcloud service for neuron feature visualization.

    Provides wordcloud generation from neuron category data.
    """
    try:
        from src.ui.services import WordcloudService
    except ImportError:
        logger.error("WordcloudService not available")
        return None

    # Find latest timestamped output directory
    latest_run_path = Path("outputs") / "LATEST_RUN.txt"
    output_dir = None

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

    # Look for label and metadata files
    labels_path = None
    metadata_path = None

    if output_dir:
        labels_path = output_dir / "neuron_labels.json"
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
def load_coactivation_service(config: Dict) -> Optional["CoactivationService"]:
    """
    Load co-activation service for neuron relationship visualization.

    Provides co-activation relationships between neurons.
    """
    # Find latest timestamped output directory
    latest_run_path = Path("outputs") / "LATEST_RUN.txt"
    output_dir = None

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
