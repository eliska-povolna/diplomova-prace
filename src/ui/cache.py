"""Streamlit caching and session state management."""

from pathlib import Path
from typing import Dict, Optional
import logging

import streamlit as st
import yaml

from src.ui.services import (
    InferenceService,
    DataService,
    LabelingService,
    ModelLoader,
)

logger = logging.getLogger(__name__)


@st.cache_resource
def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML (cached for session)."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"✅ Loaded config from {config_path}")
    return config


@st.cache_resource
def load_inference_service(config: Dict) -> InferenceService:
    """
    Load ELSA+SAE models once per session.
    
    Streamlit will call this once and reuse result across page refreshes.
    """
    # Find latest checkpoint
    outputs_dir = Path(config['model_checkpoint_dir']).parent
    checkpoint_dir = ModelLoader.find_latest_checkpoint(outputs_dir)
    
    if not checkpoint_dir:
        raise RuntimeError("No model checkpoints found!")
    
    # Load models
    elsa_ckpt = checkpoint_dir / 'elsa_best.pt'
    sae_ckpt = checkpoint_dir / 'sae_best.pt'
    
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
        duckdb_path=Path(config['duckdb_path']),
        parquet_dir=Path(config['parquet_dir']),
        config=config
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
        interpreter = NeuronInterpreter(provider='gemini')
    except ImportError:
        logger.warning("NeuronInterpreter not available, labels will be basic")
        interpreter = None
    
    service = LabelingService(
        labels_json_path=Path(config.get(
            'neuron_labels_path',
            'outputs/neuron_labels.json'
        )),
        interpreter=interpreter,
        config=config
    )
    return service


def init_session_state():
    """
    Initialize Streamlit session state.
    
    Called once per session to set up variables persisted across reruns.
    """
    if 'current_user_id' not in st.session_state:
        st.session_state.current_user_id = None
    
    if 'current_recommendations' not in st.session_state:
        st.session_state.current_recommendations = []
    
    if 'steering_modified' not in st.session_state:
        st.session_state.steering_modified = False
    
    if 'user_history' not in st.session_state:
        st.session_state.user_history = {}


def get_services() -> tuple:
    """
    Retrieve cached services from session state.
    
    Must be called after main.py initializes them.
    
    Returns:
        (inference, data, labels)
    """
    inference = st.session_state.get('inference')
    data = st.session_state.get('data')
    labels = st.session_state.get('labels')
    
    if not all([inference, data, labels]):
        st.error("Services not initialized. Check main.py setup.")
        st.stop()
    
    return inference, data, labels
