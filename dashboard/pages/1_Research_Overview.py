"""Accessible overview of the research questions and four synthetic worlds."""

from __future__ import annotations

import streamlit as st

from fair_lending.dashboard.ui import configure_page, disclaimer, page_intro


configure_page("Research overview", "🧭")
page_intro(
    "Research Overview",
    "Why a synthetic-first design helps separate observed disparity from its mechanism.",
)

st.markdown("### Main research question")
st.markdown(
    "> Do mortgage lending decisions differ across demographic groups, and can "
    "machine learning help detect those differences?"
)
st.markdown("### Statistical research question")
st.markdown(
    "> Are some demographic groups more likely to be denied mortgages after "
    "accounting for differences in borrower and loan characteristics?"
)

st.markdown("### Why synthetic data first?")
st.write(
    "In observational mortgage data, a group gap can be consistent with several "
    "different mechanisms. Here the mechanism is chosen by the researcher, so an "
    "estimated association can be compared with known synthetic ground truth before "
    "the framework is taken to HMDA."
)

st.markdown("### Four simulated worlds")
worlds = [
    (
        "Fair baseline",
        "Applicants are generated similarly across racial groups, and race does not directly affect approval.",
    ),
    (
        "Direct discrimination",
        "Applicants have comparable modeled financial characteristics, but Black applicants receive an explicit approval penalty.",
    ),
    (
        "Upstream inequality",
        "Race does not enter the approval equation, but Black applicants receive different modeled distributions of income, credit score, and liquid assets.",
    ),
    (
        "Mixed mechanism",
        "Both upstream inequality and a direct Black approval penalty are present.",
    ),
]
columns = st.columns(2)
for index, (title, body) in enumerate(worlds):
    with columns[index % 2]:
        st.markdown(
            f'<div class="research-card"><h4>{title}</h4><p>{body}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("### What the dashboard can—and cannot—show")
st.write(
    "It can show how the configured mechanisms produce different raw gaps, adjusted "
    "regression estimates, and model predictions. It cannot establish which mechanism "
    "causes an observed gap in real mortgage markets."
)
disclaimer()
