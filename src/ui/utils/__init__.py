"""UI utilities package."""

import streamlit as st
from .formatting import (
    format_feature_id,
    format_features_list,
    format_feature_explanation,
)


def info_section(title, *info_texts):
    """
    Display a section header with inline info icon and hoverable tooltip.

    Args:
        title: Section title (includes emoji icon)
        *info_texts: One or more strings describing the section; they'll be
            joined together for the tooltip and detailed box. This keeps the
            function backward-compatible with existing two-argument calls.
    """
    help_text = " ".join(str(t).strip() for t in info_texts if t)

    col_title, col_info = st.columns([0.85, 0.15])
    with col_title:
        st.subheader(title)
    with col_info:
        if st.button("ℹ️", key=f"info_{title}", help=help_text):
            st.session_state[f"show_info_{title}"] = not st.session_state.get(
                f"show_info_{title}", False
            )

    # Show info if button was clicked
    if st.session_state.get(f"show_info_{title}", False):
        st.info(help_text)


__all__ = [
    "info_section",
    "format_feature_id",
    "format_features_list",
    "format_feature_explanation",
]
