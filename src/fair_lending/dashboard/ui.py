"""Shared Streamlit presentation helpers kept separate from research logic."""

from __future__ import annotations

import streamlit as st

from fair_lending.dashboard.state import RESEARCH_DISCLAIMER, initialize_state


def configure_page(title: str, icon: str = "🏠") -> None:
    """Apply restrained academic styling and initialize cross-page state."""
    st.set_page_config(page_title=f"{title} · Synthetic Fair Lending", page_icon=icon, layout="wide")
    initialize_state(st.session_state)
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1320px;}
        [data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .75rem 1rem; border-radius: .45rem;}
        .research-card {border-left: 4px solid #861F41; padding: .15rem 1rem; margin: .7rem 0 1.1rem 0;}
        .small-note {color: #666; font-size: .9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    """Display the required compact research limitation."""
    st.info(RESEARCH_DISCLAIMER, icon="ℹ️")


def page_intro(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)
