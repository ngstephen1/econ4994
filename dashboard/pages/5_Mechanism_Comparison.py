"""Cross-method synthesis of the four validated synthetic worlds."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fair_lending.dashboard.charts import mechanism_gap_figure
from fair_lending.dashboard.data import DashboardArtifactError, build_mechanism_matrix, load_result_table
from fair_lending.dashboard.ui import configure_page, disclaimer, page_intro


configure_page("Mechanism comparison", "🧩")
page_intro(
    "Mechanism Comparison",
    "Validated Prompt 6 and Prompt 7 results, assembled without rerunning experiments.",
)


@st.cache_data(show_spinner=False)
def cached_table(name: str) -> pd.DataFrame:
    return load_result_table(name)


try:
    matrix = build_mechanism_matrix(
        cached_table("statistical_recovery"), cached_table("ml_benchmark")
    )
except DashboardArtifactError as error:
    st.error(str(error))
    disclaimer()
    st.stop()

st.plotly_chart(mechanism_gap_figure(matrix), use_container_width=True)

display = matrix.loc[:, [
    "scenario_label", "upstream_differences", "direct_race_effect", "observed_gap",
    "adjusted_black_coefficient", "adjusted_probability_gap", "race_blind_ml_gap",
    "race_aware_ml_gap", "true_direct_log_odds",
]].copy()
display.columns = [
    "Scenario", "Upstream differences", "Direct race effect", "Raw approval gap",
    "Adjusted Black coefficient", "Adjusted probability gap", "Race-blind ML gap",
    "Race-aware ML gap", "True direct effect (log odds)",
]
st.dataframe(
    display.style.format({
        "Raw approval gap": "{:+.2%}", "Adjusted Black coefficient": "{:+.3f}",
        "Adjusted probability gap": "{:+.2%}", "Race-blind ML gap": "{:+.2%}",
        "Race-aware ML gap": "{:+.2%}", "True direct effect (log odds)": "{:+.2f}",
    }),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "Raw and adjusted statistical values use the 100,000-row Prompt 6 samples. ML gaps "
    "use Prompt 7 held-out test predictions; different finite samples need not match exactly."
)

st.markdown("### Key takeaways")
for row in matrix.itertuples(index=False):
    observed = abs(float(row.observed_gap))
    adjusted = abs(float(row.adjusted_probability_gap))
    blind = abs(float(row.race_blind_ml_gap))
    aware = abs(float(row.race_aware_ml_gap))
    if not row.upstream_differences and not row.direct_race_effect:
        statement = (
            "No race mechanism is configured. The saved observed and modeled gaps are "
            f"small (largest shown magnitude {100 * max(observed, adjusted, blind, aware):.2f} pp)."
        )
    elif row.direct_race_effect and not row.upstream_differences:
        statement = (
            "A direct race penalty creates an outcome gap. The race-aware mean-prediction "
            f"gap is {100 * row.race_aware_ml_gap:+.2f} pp, compared with "
            f"{100 * row.race_blind_ml_gap:+.2f} pp when race is withheld."
        )
    elif row.upstream_differences and not row.direct_race_effect:
        statement = (
            "Upstream financial distributions differ even though race does not enter the "
            f"approval equation. Adjustment leaves a {100 * row.adjusted_probability_gap:+.2f} pp "
            "conditional contrast, while race-blind prediction reproduces predictor-linked disparity."
        )
    else:
        statement = (
            "Both pathways operate. The race-blind model primarily accesses the upstream "
            f"signal ({100 * row.race_blind_ml_gap:+.2f} pp); adding race changes the mean-prediction "
            f"gap to {100 * row.race_aware_ml_gap:+.2f} pp."
        )
    with st.expander(row.scenario_label, expanded=True):
        st.write(statement)

st.warning(
    "Observed disparity alone does not identify which mechanism caused it. Likewise, "
    "a small race-blind prediction gap does not show that the simulated decision process lacked a direct race effect."
)
disclaimer()
