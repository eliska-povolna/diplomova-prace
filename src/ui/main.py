"""
POI Recommender Streamlit App — Main Entry Point

Multi-page app with 4 pages:
1. 🏠 Home - Welcome + quick stats
2. 📊 Results - Evaluation metrics
3. 🎛️ Live Demo - Interactive steering (main page)
4. 🔍 Interpretability - Feature browser
"""

import logging
import sys
import warnings
from pathlib import Path

# Suppress ZoeDepth library warning about __path__ deprecation
warnings.filterwarnings(
    "ignore",
    message=".*Accessing `__path__` from.*zoedepth.*",
    category=DeprecationWarning,
)

import streamlit as st

# Add project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Add UI directory so "from cache import ..." is stable from any working directory.
sys.path.insert(0, str(Path(__file__).parent))

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

logger.info(f"Loading config from {config_path}")


def _show_startup_diagnostics(config: dict) -> None:
    """Display high-signal diagnostics for common path/config issues."""
    checks = [
        ("Config", config.get("config_path")),
        ("Checkpoint root", config.get("model_checkpoint_dir")),
        ("DuckDB", config.get("duckdb_path")),
        ("Parquet root", config.get("parquet_dir")),
    ]

    missing = []
    with st.expander("Startup diagnostics", expanded=False):
        # Display path checks
        for label, path_str in checks:
            exists = bool(path_str and Path(path_str).exists())
            status = "OK" if exists else "MISSING"
            st.write(f"{label}: `{path_str}` [{status}]")
            if not exists:
                missing.append(label)

        # Display service status from session state
        diag = st.session_state.get("_startup_diagnostics", {})
        if diag.get("models_loaded"):
            st.write("✅ **Models**: Loaded")
        if diag.get("backend"):
            st.write(f"☁️ **Backend**: {diag['backend']}")
        if not diag.get("local_data_available", True):
            st.write(
                "ℹ️ **Local data**: Not available (using Cloud SQL, expected on Streamlit Cloud)"
            )

    if missing:
        st.session_state._startup_diagnostics = st.session_state.get(
            "_startup_diagnostics", {}
        )
        st.session_state._startup_diagnostics["local_data_available"] = False


# Initialize services (via @st.cache_resource)
try:
    from cache import (
        init_session_state,
        load_coactivation_service,
        load_config,
        load_data_service,
        load_inference_service,
        load_labeling_service,
        load_wordcloud_service,
    )

    # Initialize session state first
    init_session_state()

    # Load and flatten config
    logger.info("Loading configuration...")
    config = load_config(config_path)
    _show_startup_diagnostics(config)
    with st.spinner("Loading models..."):
        inference = load_inference_service(config)

    with st.spinner("Loading POI data..."):
        data = load_data_service(config)

    with st.spinner("Initializing labeling service..."):
        labels = load_labeling_service(config)

    with st.spinner("Initializing wordcloud service..."):
        wordcloud = load_wordcloud_service(config)

    with st.spinner("Initializing co-activation service..."):
        coactivation = load_coactivation_service(config)

    # Store in session for access from pages
    st.session_state.inference = inference
    st.session_state.data = data
    st.session_state.labels = labels
    st.session_state.wordcloud = wordcloud
    st.session_state.coactivation = coactivation

    logger.info("✅ All services initialized")

except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    logger.exception("Service initialization failed")
    st.stop()

# Define multi-page app
from src.ui.pages import home, interpretability, live_demo, results


# Create function aliases with unique names for Streamlit navigation
def show_home():
    return home.show()


def show_results():
    return results.show()


def show_live_demo():
    return live_demo.show()


def show_interpretability():
    return interpretability.show()


pages = [
    st.Page(show_home, title="🏠 Home"),
    st.Page(show_results, title="📊 Results"),
    st.Page(show_live_demo, title="🎛️ Live Demo"),
    st.Page(show_interpretability, title="🔍 Interpretability"),
]

# Navigation
navigation = st.navigation(pages)
navigation.run()
