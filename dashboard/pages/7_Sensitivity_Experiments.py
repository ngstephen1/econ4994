"""Read-only explorer for precomputed Prompt 9 Monte Carlo results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from fair_lending.dashboard.data import DashboardArtifactError, load_result_table
from fair_lending.dashboard.ui import configure_page, disclaimer, page_intro
from fair_lending.simulation.config import PROJECT_ROOT


configure_page("Sensitivity experiments", "📊")
page_intro(
    "Sensitivity and Monte Carlo Experiments",
    "Read-only exploration of saved Prompt 9 summaries; this page never runs the Monte Carlo pipeline.",
)


@st.cache_data(show_spinner=False)
def cached_table(name: str) -> pd.DataFrame:
    return load_result_table(name)


FAMILIES = {
    "Direct-effect sweep": ("sensitivity_direct_summary", "direct_black_log_odds", "Direct effect (log odds)"),
    "Upstream-inequality sweep": ("sensitivity_upstream_summary", "upstream_strength", "Upstream strength"),
    "Mixed grid": ("sensitivity_mixed_summary", "direct_black_log_odds", "Direct effect (log odds)"),
    "Sample-size and detection": ("sensitivity_sample_size_summary", "n_rows", "Applications"),
}
FIGURES = {
    "Direct-effect sweep": "sensitivity_direct_ml_gap.png",
    "Upstream-inequality sweep": "sensitivity_upstream_observed_gap.png",
    "Mixed grid": "sensitivity_mixed_raw_gap_heatmap.png",
    "Sample-size and detection": "sensitivity_detection_probability.png",
}

try:
    paper = cached_table("sensitivity_paper_summary")
    thresholds = cached_table("detection_thresholds")
    signatures = cached_table("mechanism_signatures")
except DashboardArtifactError as error:
    st.error(str(error))
    disclaimer()
    st.stop()

st.markdown("### Paper-ready mechanism comparison")
percentage_columns = [
    "raw_approval_gap",
    "adjusted_probability_gap",
    "race_blind_ml_gap",
    "race_aware_ml_gap",
    "true_direct_probability_contrast",
    "detection_probability",
    "coverage",
]
st.dataframe(
    paper.style.format({column: "{:.1%}" for column in percentage_columns}),
    hide_index=True,
    use_container_width=True,
)

family_label = st.selectbox("Experiment family", list(FAMILIES))
table_key, x_field, x_label = FAMILIES[family_label]
try:
    summary = cached_table(table_key)
except DashboardArtifactError as error:
    st.warning(str(error))
    disclaimer()
    st.stop()

sample_sizes = sorted(summary["n_rows"].dropna().astype(int).unique())
selected_n = st.selectbox("Sample size", sample_sizes, index=len(sample_sizes) - 1)
filtered = summary.loc[summary["n_rows"].eq(selected_n)].copy()
if family_label == "Mixed grid":
    upstream_values = sorted(filtered["upstream_strength"].unique())
    selected_upstream = st.selectbox("Upstream strength", upstream_values)
    filtered = filtered.loc[filtered["upstream_strength"].eq(selected_upstream)]
elif family_label == "Sample-size and detection":
    direct_values = sorted(filtered["direct_black_log_odds"].unique(), reverse=True)
    selected_direct = st.selectbox("Direct effect", direct_values)
    filtered = filtered.loc[filtered["direct_black_log_odds"].eq(selected_direct)]

estimands = [
    value
    for value in (
        "raw_approval_gap",
        "model_2_adjusted_probability_gap",
        "race_blind_predicted_probability_gap",
        "race_aware_predicted_probability_gap",
    )
    if value in set(filtered["estimand"])
]
selected_estimand = st.selectbox("Estimand", estimands)
chart = filtered.loc[filtered["estimand"].eq(selected_estimand)].sort_values(x_field)
if len(chart):
    st.line_chart(chart.set_index(x_field)["mean"], x_label=x_label, y_label="Probability difference")
st.dataframe(
    chart.loc[:, [
        "n_rows", "direct_black_log_odds", "upstream_strength", "estimand",
        "n_completed", "mean", "mc_sd", "q025", "q975", "bias", "rmse",
        "detection_probability", "coverage_probability",
    ]],
    hide_index=True,
    use_container_width=True,
)
st.caption("All probability gaps use Black minus White internally; multiply by 100 for percentage points.")

figure_path = PROJECT_ROOT / "results" / "figures" / FIGURES[family_label]
if figure_path.exists():
    st.image(str(figure_path), caption=f"Precomputed figure: {family_label}", use_container_width=True)
else:
    st.info("The selected precomputed figure is missing. Rerun the sensitivity experiment to recreate it.")

threshold_tab, signature_tab = st.tabs(["Detection thresholds", "Mechanism signatures"])
with threshold_tab:
    st.dataframe(thresholds, hide_index=True, use_container_width=True)
with signature_tab:
    modal = signatures.loc[signatures["is_modal_signature"].astype(bool)]
    st.dataframe(modal, hide_index=True, use_container_width=True)
    st.caption("Mechanism signatures are synthetic diagnostics, not definitive causal classifiers.")
disclaimer()
