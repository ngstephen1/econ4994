"""Shared Streamlit state keys and research language."""

from __future__ import annotations


DATA_KEY = "active_simulation_data"
METADATA_KEY = "active_simulation_metadata"
SUMMARY_KEY = "active_simulation_summary"
STATISTICS_KEY = "active_statistical_results"
REQUEST_KEY = "active_simulation_request"

RESEARCH_DISCLAIMER = (
    "This dashboard demonstrates synthetic mechanisms under researcher-chosen "
    "assumptions. It does not estimate real-world discrimination and should not be "
    "used to make lending decisions."
)


def initialize_state(session_state: object) -> None:
    """Create stable keys without overwriting an active cross-page experiment."""
    for key in (DATA_KEY, METADATA_KEY, SUMMARY_KEY, STATISTICS_KEY, REQUEST_KEY):
        if key not in session_state:
            session_state[key] = None
