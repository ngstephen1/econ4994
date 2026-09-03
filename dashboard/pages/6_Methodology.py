"""Readable methodology, data dictionary, and limitations."""

from __future__ import annotations

import streamlit as st

from fair_lending.dashboard.data import data_dictionary
from fair_lending.dashboard.ui import configure_page, disclaimer, page_intro
from fair_lending.simulation.calibration import load_calibration_artifact


configure_page("Methodology", "📚")
page_intro(
    "Methodology and Data Dictionary",
    "The scientific design underneath the interactive presentation layer.",
)

st.markdown("### Research questions")
st.markdown(
    "**Main:** Do mortgage lending decisions differ across demographic groups, and can machine learning help detect those differences?"
)
st.markdown(
    "**Statistical:** Are some demographic groups more likely to be denied mortgages after accounting for differences in borrower and loan characteristics?"
)

st.markdown("### Synthetic-first design")
st.write(
    "One row represents one synthetic mortgage application. Demographics are generated "
    "before upstream/contextual conditions, financial characteristics, loan attributes, "
    "the underwriting score, and the Bernoulli approval decision. Because the full DGP "
    "is known, model signals can be compared with configured truth."
)
st.latex(
    r"p(approved_i=1)=\operatorname{sigmoid}(\alpha + X_i\beta + L_i\gamma + \delta_{race(i)})"
)
st.caption(
    "X contains transformed financial terms, L contains declared loan/product terms, "
    "and the direct group effect is zero unless the selected scenario enables it."
)

st.markdown("### Four scenarios")
st.table(
    {
        "Scenario": ["Fair baseline", "Direct discrimination", "Upstream inequality", "Mixed mechanism"],
        "Upstream financial treatment": ["No", "No", "Yes", "Yes"],
        "Direct Black approval term": ["No", "Yes", "No", "Yes"],
    }
)

st.markdown("### Reproducibility and the frozen intercept")
try:
    calibration = load_calibration_artifact()
    st.write(
        f"The fair-baseline intercept is frozen at **{float(calibration['intercept']):.12f}** "
        f"after calibration to a mean probability target of **{float(calibration['target_mean_probability']):.0%}**. "
        "Scenario and custom treatments reuse it; the dashboard does not recalibrate away their effects."
    )
except FileNotFoundError:
    st.warning("The calibrated-intercept artifact is missing. Run the documented simulation calibration workflow before generating data.")
st.write(
    "Every interactive request records its scenario, effect level, sample size, seed, "
    "resolved treatments, random-stream spawn keys, and configuration fingerprint. "
    "Identical inputs reproduce identical rows."
)

st.markdown("### Statistical and machine-learning methods")
st.write(
    "The statistical page uses four nested statsmodels logistic specifications and a "
    "standardized Black–White probability contrast. The benchmark compares logistic "
    "regression, random forest, and histogram gradient boosting with held-out tests. "
    "Race-blind models exclude race; race-aware models are explicitly labeled sensitivity analyses."
)
st.write(
    "The synthetic oracle is the DGP probability itself, not a fitted model. It is used "
    "only to judge probability recovery and would not exist in observational HMDA data."
)

st.markdown("### 24-field data dictionary")
dictionary = data_dictionary()
for group in dictionary["Group"].drop_duplicates():
    with st.expander(group):
        st.dataframe(
            dictionary.loc[dictionary["Group"].eq(group), ["Field", "Type / unit", "Description"]],
            hide_index=True,
            use_container_width=True,
        )

st.markdown("### Current limitations and later HMDA phase")
st.write(
    "Synthetic findings demonstrate statistical behavior under chosen assumptions; they "
    "do not prove real-world discrimination. Observed and adjusted disparities are not "
    "automatically causal effects. Error-rate metrics compare predictions with simulated "
    "lender decisions, not with an independent repayment outcome. The current population "
    "and category design are intentionally simplified."
)
st.write(
    "A later phase may apply the validated workflow to 2024 HMDA records. That application "
    "will use disparity language because the true causal mechanism is not observed. No HMDA "
    "data are loaded by this dashboard."
)
disclaimer()
