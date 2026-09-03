"""Statsmodels recovery analysis for the active interactive simulation."""

from __future__ import annotations

import streamlit as st

from fair_lending.dashboard.charts import coefficient_path_figure
from fair_lending.dashboard.statistical_service import (
    run_statistical_analysis,
    statistical_interpretation,
)
from fair_lending.dashboard.state import DATA_KEY, METADATA_KEY, STATISTICS_KEY
from fair_lending.dashboard.ui import configure_page, disclaimer, page_intro


configure_page("Statistical analysis", "📐")
page_intro(
    "Statistical Analysis",
    "Run the predeclared statsmodels logistic-regression sequence on the active sample.",
)

data = st.session_state[DATA_KEY]
metadata = st.session_state[METADATA_KEY]
if data is None:
    st.warning("No active sample. Generate one on the **Simulation Lab** page first.")
    disclaimer()
    st.stop()

st.caption(
    f"Active sample: {metadata['scenario']} · {metadata['effect_level']} · "
    f"n={len(data):,} · seed={metadata['seed']}"
)
st.markdown(
    """
    - **Model 0:** approved ~ Black indicator
    - **Model 1:** Black + DGP-scaled income, credit, employment, assets, DTI, and LTV
    - **Model 2:** Model 1 + loan purpose, loan type, and occupancy type
    - **Model 3:** Model 2 + age group, sex, and ethnicity
    """
)

if st.button("Run statistical analysis", type="primary"):
    with st.spinner("Fitting four statsmodels logistic regressions…"):
        st.session_state[STATISTICS_KEY] = run_statistical_analysis(
            data, metadata["resolved_configuration"]
        )

results = st.session_state[STATISTICS_KEY]
if results is None:
    st.info("Select **Run statistical analysis**. Models are not refit on ordinary page reruns.")
    disclaimer()
    st.stop()

display = results.loc[
    :,
    [
        "model_label",
        "black_coefficient",
        "black_odds_ratio",
        "black_ci_low",
        "black_ci_high",
        "black_p_value",
        "adjusted_probability_gap",
        "converged",
    ],
].copy()
display.columns = [
    "Model",
    "Black coefficient",
    "Odds ratio",
    "CI low",
    "CI high",
    "p-value",
    "Adjusted probability gap",
    "Converged",
]
st.dataframe(
    display.style.format(
        {
            "Black coefficient": "{:+.3f}",
            "Odds ratio": "{:.3f}",
            "CI low": "{:+.3f}",
            "CI high": "{:+.3f}",
            "p-value": "{:.4g}",
            "Adjusted probability gap": "{:+.2%}",
        }
    ),
    hide_index=True,
    use_container_width=True,
)
st.caption("Adjusted probability gap is standardized p(Black) − p(White), holding all other modeled values fixed.")
st.plotly_chart(coefficient_path_figure(results), use_container_width=True)
st.info(statistical_interpretation(results))
st.warning(
    "Logistic coefficients are non-collapsible. A change between Model 0 and Model 2 "
    "is not, by itself, a formal estimate of mediation or of an amount ‘explained’ by controls."
)
st.download_button(
    "Download active regression table",
    results.to_csv(index=False).encode("utf-8"),
    "active_statistical_analysis.csv",
    "text/csv",
)
disclaimer()
