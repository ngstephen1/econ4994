"""Entry point for the synthetic fair-lending research dashboard."""

from __future__ import annotations

import streamlit as st

from fair_lending.dashboard.ui import configure_page, disclaimer


configure_page("Research home", "🏠")

st.title("Synthetic Fair-Lending Research Lab")
st.subheader("Mortgage approval disparities under known simulated mechanisms")
st.write(
    "This ECON 4994 capstone uses controlled synthetic mortgage applications to "
    "study how direct race effects and upstream economic inequality appear in "
    "descriptive statistics, regression, and machine-learning predictions."
)

question, statistical = st.columns(2)
with question:
    st.markdown("#### Main research question")
    st.markdown(
        "> Do mortgage lending decisions differ across demographic groups, and can "
        "machine learning help detect those differences?"
    )
with statistical:
    st.markdown("#### Statistical research question")
    st.markdown(
        "> Are some demographic groups more likely to be denied mortgages after "
        "accounting for differences in borrower and loan characteristics?"
    )

st.divider()
st.markdown("### Explore the project")
pages = [
    ("pages/1_Research_Overview.py", "Research Overview", "Questions and four synthetic worlds"),
    ("pages/2_Simulation_Lab.py", "Simulation Lab", "Generate and inspect an in-memory experiment"),
    ("pages/3_Statistical_Analysis.py", "Statistical Analysis", "Run the validated regression sequence"),
    ("pages/4_ML_Benchmark.py", "ML Benchmark", "Explore held-out benchmark and threshold results"),
    ("pages/5_Mechanism_Comparison.py", "Mechanism Comparison", "Compare mechanisms across methods"),
    ("pages/6_Methodology.py", "Methodology", "Study design, schema, and limitations"),
    ("pages/7_Sensitivity_Experiments.py", "Sensitivity Experiments", "Explore precomputed Monte Carlo summaries"),
]
for path, label, help_text in pages:
    left, right = st.columns([1, 4])
    with left:
        st.page_link(path, label=label)
    with right:
        st.caption(help_text)

disclaimer()
