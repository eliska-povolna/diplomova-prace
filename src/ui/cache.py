"""Streamlit caching and session state management."""

from pathlib import Path
from typing import Dict, Optional
import logging

import streamlit as st
import yaml

from services import (
    InferenceService,
    DataService,
    LabelingService,
    ModelLoader,
)

logger = logging.getLogger(__name__)


@st.cache_resource
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
    config["model_checkpoint_dir"] = raw_config.get("output", {}).get("base_dir", "outputs")
    config["neuron_labels_path"] = "outputs/neuron_labels.json"
    
    # Compute n_items from parquet data
    try:
        import duckdb
        import pandas as pd
        parquet_pattern = str(Path(config["parquet_dir"]) / "business" / "**" / "*.parquet")
        conn = duckdb.connect(":memory:")
        result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_pattern}')").fetchall()
        config["n_items"] = result[0][0] if result else 50000  # Fallback estimate
        logger.info(f"   Found {config['n_items']} items in dataset")
    except Exception as e:
        logger.warning(f"Could not count items from parquet: {e}. Using estimate.")
        config["n_items"] = 50000  # Safe estimate for Yelp business data
    
    logger.info(f"✅ Loaded config from {config_path}")
    logger.info(f"   Device: {config['device']}, Latent dim: {config['latent_dim']}, SAE k: {config['k']}")
    return config


@st.cache_resource
def load_inference_service(config: Dict) -> InferenceService:
    """
    Load ELSA+SAE models once per session.

    Streamlit will call this once and reuse result across page refreshes.
    """
    # Find latest checkpoint
    outputs_dir = Path(config["model_checkpoint_dir"]).parent
    checkpoint_dir = ModelLoader.find_latest_checkpoint(outputs_dir)

    if not checkpoint_dir:
        raise RuntimeError("No model checkpoints found!")

    # Load models
    elsa_ckpt = checkpoint_dir / "elsa_best.pt"
    sae_ckpt = checkpoint_dir / "sae_best.pt"

    if not elsa_ckpt.exists() or not sae_ckpt.exists():
        raise FileNotFoundError(f"Missing checkpoints in {checkpoint_dir}")

    service = InferenceService(elsa_ckpt, sae_ckpt, config)
    st.success("✅ Models loaded")
    return service


@st.cache_resource
def load_data_service(config: Dict) -> DataService:
    """
    Load POI data once per session.

    DuckDB + Parquet files cached in memory.
    """
    service = DataService(
        duckdb_path=Path(config["duckdb_path"]),
        parquet_dir=Path(config["parquet_dir"]),
        config=config,
    )
    st.success(f"✅ Loaded {service.num_pois} POIs")
    return service


@st.cache_resource
def load_labeling_service(config: Dict) -> LabelingService:
    """
    Load neuron labeling service.

    Labels are lazy-loaded on first access (no startup delay).
    """
    # Try to load NeuronInterpreter from notebook
    try:
        from src.interpret.neuron_interpreter import NeuronInterpreter

        interpreter = NeuronInterpreter(provider="gemini")
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
    inference = st.session_state.get("inference")
    data = st.session_state.get("data")
    labels = st.session_state.get("labels")

    if not all([inference, data, labels]):
        st.error("Services not initialized. Check main.py setup.")
        st.stop()

    return inference, data, labels
