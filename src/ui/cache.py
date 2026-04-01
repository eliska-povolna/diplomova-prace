"""Streamlit caching and session state management."""

from pathlib import Path
from typing import Dict, Optional
import logging

try:
    import streamlit as st

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

    # Define dummy decorator for non-Streamlit contexts
    def cache_resource(func):
        return func


import yaml

from services import (
    InferenceService,
    DataService,
    LabelingService,
    ModelLoader,
)

logger = logging.getLogger(__name__)

# Use streamlit's cache decorator if available, otherwise use dummy
if HAS_STREAMLIT:
    st_cache_resource = st.cache_resource
else:
    st_cache_resource = cache_resource


@st_cache_resource
def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML and flatten for UI services."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    # Flatten nested config structure for UI services
    config = {}

    # Data paths (from data section)
    if "data" in raw_config:
        config["duckdb_path"] = raw_config["data"].get("db_path", "")
        config["parquet_dir"] = raw_config["data"].get("parquet_dir", "")

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
    config["model_checkpoint_dir"] = raw_config.get("output", {}).get(
        "base_dir", "outputs"
    )
    config["neuron_labels_path"] = "outputs/neuron_labels.json"

    # Compute n_items from parquet data (apply same filters as training)
    # NOTE: This is now only for informational purposes. The inference service
    # reads n_items from checkpoint metadata, not from config.
    try:
        import duckdb
        import pandas as pd

        parquet_pattern = str(
            Path(config["parquet_dir"]) / "business" / "**" / "*.parquet"
        )
        conn = duckdb.connect(":memory:")

        # Get state filter from config if available
        state_filter = raw_config.get("data", {}).get("state_filter")

        if state_filter:
            # Apply state filter like training did
            query = f"""
                SELECT COUNT(*) 
                FROM read_parquet('{parquet_pattern}')
                WHERE state = '{state_filter}'
            """
            logger.info(f"   Counting items with state_filter='{state_filter}'...")
        else:
            query = f"SELECT COUNT(*) FROM read_parquet('{parquet_pattern}')"
            logger.info("   Counting all items (no state filter)...")

        result = conn.execute(query).fetchall()
        config["n_items"] = result[0][0] if result else 50000  # Fallback estimate
        logger.info(f"   Found {config['n_items']} items in dataset")
    except Exception as e:
        logger.warning(f"Could not count items from parquet: {e}. Using placeholder.")
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
            logger.error(f"No checkpoint subdirs found")
            logger.error(f"Searched: {[d.name for d in subdirs]}")
            for subdir in subdirs[:3]:  # Show contents of first 3 dirs
                contents = list(subdir.iterdir())
                logger.error(f"  {subdir.name}/: {[c.name for c in contents]}")
            raise RuntimeError(f"No model checkpoints found in {ckpt_base}")

    if not checkpoint_dir:
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

    service = InferenceService(elsa_ckpt, sae_ckpt, config)
    if HAS_STREAMLIT:
        st.success("✅ Models loaded")
    logger.info("✅ Models loaded successfully")
    return service


@st_cache_resource
def load_data_service(config: Dict) -> DataService:
    """
    Load POI data once per session.

    DuckDB + Parquet files cached in memory.
    Uses item2index mapping to ensure POI indices match model coordinate space.
    Loads local photos from yelp_photos folder if available.
    """
    # Path to k-core FILTERED item2index mapping from training
    # This mapping has 2212 items (after k-core filtering), matching the model
    # Project layout: src/ui/cache.py -> need to go up 3 levels to project root
    item2index_path = Path(__file__).parent.parent.parent / "data" / "processed_yelp_easystudy" / "item2index_filtered.pkl"
    
    # Path to local photos folder
    local_photos_path = Path(__file__).parent.parent.parent / "yelp_photos"
    if not local_photos_path.exists():
        local_photos_path = None
    
    service = DataService(
        duckdb_path=Path(config["duckdb_path"]),
        parquet_dir=Path(config["parquet_dir"]),
        config=config,
        item2index_path=item2index_path,
        local_photos_dir=local_photos_path,
    )
    if HAS_STREAMLIT:
        st.success(f"✅ Loaded {service.num_pois} POIs")
        if local_photos_path:
            st.info(f"📷 Local photos enabled: {local_photos_path}")
    logger.info(f"✅ Loaded {service.num_pois} POIs")
    if local_photos_path:
        logger.info(f"📷 Local photos enabled: {local_photos_path}")
    return service


@st_cache_resource
def load_labeling_service(config: Dict) -> LabelingService:
    """
    Load neuron labeling service.

    Labels are lazy-loaded on first access (no startup delay).
    If no LLM provider is available (no API keys), uses basic pre-computed labels.
    """
    # Try to load NeuronInterpreter - let it auto-detect provider
    interpreter = None
    try:
        from src.interpret.neuron_interpreter import NeuronInterpreter

        # Auto-detect provider based on environment variables
        # (github_models if GITHUB_TOKEN is set, gemini if GOOGLE_API_KEY is set)
        try:
            interpreter = NeuronInterpreter()  # Auto-detect provider
            logger.info(
                f"✅ NeuronInterpreter initialized with provider: {interpreter.provider}"
            )
        except ValueError as e:
            # No API keys available - use basic labels instead
            logger.info(f"LLM provider not available ({e}), using basic labels")
            interpreter = None
    except ImportError:
        logger.warning("NeuronInterpreter not available, labels will be basic")
        interpreter = None

    service = LabelingService(
        labels_json_path=Path(
            config.get("neuron_labels_path", "outputs/neuron_labels.json")
        ),
        interpreter=interpreter,
        config=config,
    )
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
