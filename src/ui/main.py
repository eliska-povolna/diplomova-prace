"""
POI Recommender Streamlit App — Main Entry Point

Multi-page app with 4 pages:
1. 🏠 Home - Welcome + quick stats
2. 📊 Results - Evaluation metrics
3. 🎛️ Live Demo - Interactive steering (main page)
4. 🔍 Interpretability - Feature browser
"""

from pathlib import Path
import sys
import logging

import streamlit as st
import yaml

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="POI Recommender — Sparse Features",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load config
config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
if not config_path.exists():
    st.error(f"Config not found: {config_path}")
    st.stop()

with open(config_path) as f:
    config = yaml.safe_load(f)

logger.info(f"✅ Loaded config from {config_path}")

# Initialize services (via @st.cache_resource)
try:
    from src.ui.cache import (
        load_config,
        load_inference_service,
        load_data_service,
        load_labeling_service,
        init_session_state,
    )

    # Initialize session state first
    init_session_state()

    # Load services (cached across page refreshes)
    logger.info("Initializing services...")
    with st.spinner("Loading models..."):
        inference = load_inference_service(config)

    with st.spinner("Loading POI data..."):
        data = load_data_service(config)

    with st.spinner("Initializing labeling service..."):
        labels = load_labeling_service(config)

    # Store in session for access from pages
    st.session_state.inference = inference
    st.session_state.data = data
    st.session_state.labels = labels

    logger.info("✅ All services initialized")

except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    logger.exception("Service initialization failed")
    st.stop()

# Define multi-page app
from src.ui.pages import home, results, live_demo, interpretability

pages = [
    st.Page(home.show, title="🏠 Home"),
    st.Page(results.show, title="📊 Results"),
    st.Page(live_demo.show, title="🎛️ Live Demo"),
    st.Page(interpretability.show, title="🔍 Interpretability"),
]

# Navigation
navigation = st.navigation(pages)
navigation.run()
