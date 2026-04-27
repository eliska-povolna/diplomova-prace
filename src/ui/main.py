"""POI Recommender Streamlit app entrypoint (strict best-run mode)."""

import logging
import sys
import warnings
from pathlib import Path
import os

warnings.filterwarnings(
    "ignore",
    message=".*Accessing `__path__` from.*zoedepth.*",
    category=DeprecationWarning,
)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

st.set_page_config(
    page_title="POI Recommender - Sparse Features",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
if not config_path.exists():
    st.error(f"Config not found: {config_path}")
    st.stop()


def _render_global_label_method_selector(labels_service) -> None:
    available_methods = getattr(labels_service, "available_methods", [])
    if not available_methods:
        return

    default_method = st.session_state.get(
        "global_label_method",
        getattr(labels_service, "selected_method", available_methods[0]),
    )
    if default_method not in available_methods:
        default_method = available_methods[0]

    with st.sidebar:
        st.divider()
        st.subheader("Global Labeling")
        selected_method = st.selectbox(
            "Label source",
            options=available_methods,
            index=available_methods.index(default_method),
            key="global_label_method",
        )
        st.caption("Applies across Live Demo and Interpretability.")
        if len(available_methods) == 1 and available_methods[0] == "weighted-category":
            st.warning("Only weighted-category artifact found for this run.")

    if hasattr(labels_service, "set_method"):
        labels_service.set_method(selected_method)


try:
    from cache import (
        init_session_state,
        load_coactivation_service,
        load_config,
        load_data_service,
        load_inference_service,
        load_labeling_service,
        load_run_artifact_bundle,
        load_semantic_search_model,
        load_training_results,
        validate_cloud_run_artifacts,
        load_wordcloud_service,
    )

    init_session_state()
    config = load_config(config_path)

    with st.spinner("Loading experiment results..."):
        training_results = load_training_results(config, None)
    startup_notice = (training_results or {}).get("startup_notice")
    if startup_notice:
        st.warning(startup_notice)
        st.session_state["startup_notice"] = startup_notice

    default_run_dir = (
        training_results.get("default_run_dir") if training_results else None
    )
    if not default_run_dir:
        raise RuntimeError("No best run found in the latest experiment manifest.")

    selected_output_dir = default_run_dir
    st.session_state.selected_result_run_dir = selected_output_dir

    with st.spinner("Validating strict best-run artifacts..."):
        run_bundle = load_run_artifact_bundle(selected_output_dir)
    cloud_artifact_report = validate_cloud_run_artifacts(selected_output_dir)

    with st.spinner("Loading models..."):
        inference = load_inference_service(config, selected_output_dir)

    with st.spinner("Loading POI data..."):
        data = load_data_service(
            config,
            selected_output_dir=selected_output_dir,
            expected_n_items=inference.n_items,
        )

    with st.spinner("Loading labeling artifacts..."):
        labels = load_labeling_service(
            config,
            _data_service=data,
            selected_output_dir=selected_output_dir,
        )

    inference.labels = labels
    inference.data_service = data

    with st.spinner("Loading wordcloud artifacts..."):
        wordcloud = load_wordcloud_service(config, selected_output_dir)

    with st.spinner("Loading co-activation artifacts..."):
        coactivation = load_coactivation_service(config, selected_output_dir)

    with st.spinner("Loading semantic search model..."):
        semantic_search_model = load_semantic_search_model()

    st.session_state.inference = inference
    st.session_state.data = data
    st.session_state.labels = labels
    st.session_state.wordcloud = wordcloud
    st.session_state.coactivation = coactivation
    st.session_state.training_results = training_results
    st.session_state.strict_run_bundle = run_bundle
    st.session_state.cloud_artifact_report = cloud_artifact_report
    st.session_state.config = config
    st.session_state.semantic_search_model = semantic_search_model

    _render_global_label_method_selector(labels)
    logger.info("All services initialized in strict best-run mode")

except Exception as e:
    st.error(f"Failed to initialize services: {e}")
    logger.exception("Service initialization failed")
    st.stop()

from src.ui.pages import dataset_statistics, home, interpretability, live_demo, results


def show_home():
    return home.show()


def show_results():
    return results.show()


def show_dataset_statistics():
    return dataset_statistics.show()


def show_live_demo():
    return live_demo.show()


def show_interpretability():
    return interpretability.show()


pages = [
    st.Page(show_home, title="🏠 Home"),
    st.Page(show_dataset_statistics, title="🗂️ Dataset Statistics"),
    st.Page(show_results, title="📊 Results"),
    st.Page(show_live_demo, title="🎛️ Live Demo"),
    st.Page(show_interpretability, title="🔍 Interpretability"),
]

st.session_state["_streamlit_pages"] = pages
st.session_state["_home_page"] = pages[0]
st.session_state["_dataset_statistics_page"] = pages[1]
st.session_state["_results_page"] = pages[2]
st.session_state["_live_demo_page"] = pages[3]
st.session_state["_interpretability_page"] = pages[4]

navigation = st.navigation(pages)
navigation.run()
