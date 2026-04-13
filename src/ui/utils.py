"""Shared utility functions for UI pages."""

import streamlit as st


def info_section(title, info_text):
    """
    Display a section header with inline info icon and hoverable tooltip.

    Args:
        title: Section title (includes emoji icon)
        info_text: Tooltip text explaining the section
    """
    col_title, col_info = st.columns([0.85, 0.15])
    with col_title:
        st.subheader(title)
    with col_info:
        if st.button("ℹ️", key=f"info_{title}", help=info_text):
            st.session_state[f"show_info_{title}"] = not st.session_state.get(
                f"show_info_{title}", False
            )

    # Show info if button was clicked
    if st.session_state.get(f"show_info_{title}", False):
        st.info(info_text)
