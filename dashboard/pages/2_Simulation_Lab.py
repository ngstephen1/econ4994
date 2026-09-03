"""Interactive synthetic sample generation and mechanism exploration."""

from __future__ import annotations

import streamlit as st

from fair_lending.dashboard.charts import (
    approval_rate_figure,
    distribution_figure,
    sensitivity_figure,
)
from fair_lending.dashboard.formatting import (
    format_currency,
    format_number,
    format_percentage,
    format_percentage_points,
)
from fair_lending.dashboard.simulation_service import (
    CustomTreatments,
    SimulationRequest,
    configured_treatments,
    export_dataset_csv,
    export_dataset_parquet,
    export_reproducibility_json,
    export_summary_csv,
    generate_dashboard_simulation,
    race_outcome_table,
    resolve_dashboard_config,
    summarize_simulation,
)
from fair_lending.dashboard.state import (
    DATA_KEY,
    METADATA_KEY,
    REQUEST_KEY,
    STATISTICS_KEY,
    SUMMARY_KEY,
)
from fair_lending.dashboard.ui import configure_page, disclaimer, page_intro


configure_page("Simulation lab", "🧪")
page_intro(
    "Simulation Lab",
    "Generate one deterministic mortgage-application sample under a known mechanism.",
)

SCENARIO_OPTIONS = {
    "Fair baseline": "fair_baseline",
    "Direct discrimination": "direct_discrimination",
    "Upstream inequality": "upstream_inequality",
    "Mixed mechanism": "mixed_mechanism",
}
VARIABLES = {
    "Annual income": ("annual_income", "Annual income (USD)"),
    "Credit score": ("credit_score", "Synthetic credit score"),
    "Liquid assets": ("liquid_assets", "Liquid assets (USD)"),
    "Debt-to-income ratio": ("debt_to_income_ratio", "Debt-to-income ratio"),
    "Loan-to-value ratio": ("loan_to_value_ratio", "Loan-to-value ratio"),
    "Property value": ("property_value", "Property value (USD)"),
    "Loan amount": ("loan_amount", "Loan amount (USD)"),
}


@st.cache_data(show_spinner=False)
def cached_generate(request: SimulationRequest):
    return generate_dashboard_simulation(request)


@st.cache_data(show_spinner=False)
def cached_sensitivity(seed: int):
    rows = []
    for direct_effect in (0.0, -0.10, -0.25, -0.50):
        request = SimulationRequest(
            scenario="fair_baseline",
            effect_level="moderate",
            n_samples=10_000,
            random_seed=seed,
            custom_treatments=CustomTreatments(direct_effect, 1.0, 0.0, 1.0),
        )
        data, metadata = generate_dashboard_simulation(request)
        summary = summarize_simulation(data, metadata)
        rows.append(
            {
                "direct_black_log_odds": direct_effect,
                "gap_percentage_points": 100.0 * summary["black_white_approval_gap"],
            }
        )
    import pandas as pd

    return pd.DataFrame(rows)


with st.sidebar:
    st.header("Experiment controls")
    scenario_label = st.selectbox("Scenario", list(SCENARIO_OPTIONS))
    scenario = SCENARIO_OPTIONS[scenario_label]
    effect_level = st.selectbox("Effect level", ["mild", "moderate", "strong"], index=1)
    n_samples = st.selectbox(
        "Number of applications", [1_000, 5_000, 10_000, 50_000, 100_000], index=2
    )
    seed = int(st.number_input("Random seed", min_value=0, value=4_994, step=1))
    advanced = st.toggle("Advanced / custom experiment", value=False)
    custom = None
    if advanced:
        canonical = configured_treatments(
            resolve_dashboard_config(
                SimulationRequest(scenario, effect_level, n_samples, seed)
            )
        )
        st.caption(
            "These are researcher-controlled synthetic treatment parameters, not "
            "estimates of real-world discrimination."
        )
        direct = st.slider(
            "Direct Black effect (log odds)", -0.50, 0.0, float(canonical["direct_black_log_odds"]), 0.05
        )
        income = st.slider(
            "Black income multiplier", 0.70, 1.00, float(canonical["black_income_multiplier"]), 0.01
        )
        credit = st.slider(
            "Black credit-score shift", -75.0, 0.0, float(canonical["black_credit_score_shift"]), 1.0
        )
        assets = st.slider(
            "Black liquid-assets multiplier", 0.30, 1.00, float(canonical["black_liquid_assets_multiplier"]), 0.01
        )
        custom = CustomTreatments(direct, income, credit, assets)
    generate = st.button("Generate simulation", type="primary", use_container_width=True)

request = SimulationRequest(scenario, effect_level, n_samples, seed, custom)
if generate:
    with st.spinner("Generating the deterministic synthetic sample…"):
        data, metadata = cached_generate(request)
        summary = summarize_simulation(data, metadata)
    st.session_state[DATA_KEY] = data
    st.session_state[METADATA_KEY] = metadata
    st.session_state[SUMMARY_KEY] = summary
    st.session_state[REQUEST_KEY] = request
    st.session_state[STATISTICS_KEY] = None
    st.success(f"Generated {len(data):,} applications in memory. No repository data file was written.")

data = st.session_state[DATA_KEY]
metadata = st.session_state[METADATA_KEY]
summary = st.session_state[SUMMARY_KEY]
if data is None:
    st.info("Choose an experiment and select **Generate simulation**. The default is a 10,000-row fair baseline.")
    disclaimer()
    st.stop()

st.caption(
    f"Active sample: {metadata['scenario']} · {metadata['effect_level']} · "
    f"n={metadata['n_rows']:,} · seed={metadata['seed']}"
)
top = st.columns(5)
top[0].metric("Applications", format_number(summary["n_applications"]))
top[1].metric("Overall approval", format_percentage(summary["overall_approval_rate"]))
top[2].metric("White approval", format_percentage(summary["white_approval_rate"]))
top[3].metric("Black approval", format_percentage(summary["black_approval_rate"]))
top[4].metric("Black − White gap", format_percentage_points(summary["black_white_approval_gap"]))
bottom = st.columns(4)
bottom[0].metric("Median income", format_currency(summary["median_income"]))
bottom[1].metric("Median credit score", format_number(summary["median_credit_score"]))
bottom[2].metric("Median DTI", format_percentage(summary["median_dti"]))
bottom[3].metric("Median LTV", format_percentage(summary["median_ltv"]))
st.caption(
    "Finite samples may show small differences even when the true configured direct effect is zero."
)

distribution_tab, rates_tab, mechanism_tab, sensitivity_tab, export_tab = st.tabs(
    ["Distributions", "Approval rates", "Observed vs true", "Sensitivity", "Exports"]
)
with distribution_tab:
    selected = st.selectbox("Variable", list(VARIABLES), key="distribution_variable")
    variable, unit = VARIABLES[selected]
    st.plotly_chart(distribution_figure(data, variable, unit), use_container_width=True)
with rates_tab:
    rates = race_outcome_table(data, metadata)
    st.plotly_chart(approval_rate_figure(rates), use_container_width=True)
    table = rates.loc[:, ["race", "n_applications", "approval_rate", "denial_rate"]].copy()
    table.columns = ["Race", "Applications", "Approval rate", "Denial rate"]
    st.dataframe(
        table.style.format({"Applications": "{:,.0f}", "Approval rate": "{:.1%}", "Denial rate": "{:.1%}"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("These are unadjusted outcome differences; the chart alone does not identify a causal mechanism.")
with mechanism_tab:
    observed, configured, fixed = st.columns(3)
    observed.metric("Observed outcome disparity", format_percentage_points(summary["black_white_approval_gap"]))
    configured.metric("Configured direct race effect", f"{summary['direct_black_log_odds']:+.2f} log odds")
    fixed.metric("True fixed-feature contrast", format_percentage_points(summary["true_direct_probability_gap"]))
    st.markdown("#### Configured upstream treatment for Black applicants")
    treatment_table = {
        "Income multiplier": f"{summary['black_income_multiplier']:.2f}×",
        "Credit-score shift": f"{summary['black_credit_score_shift']:+.0f} points",
        "Liquid-assets multiplier": f"{summary['black_liquid_assets_multiplier']:.2f}×",
    }
    st.table(treatment_table)
    st.caption("Observed disparity, configured treatment, and estimated model results are distinct quantities.")
with sensitivity_tab:
    st.write(
        "Run a small fixed-seed experiment that changes only the direct Black log-odds term "
        "while holding upstream treatment at identity."
    )
    if st.button("Run direct-effect sensitivity", key="run_sensitivity"):
        st.session_state["direct_sensitivity"] = cached_sensitivity(seed)
    if "direct_sensitivity" in st.session_state:
        st.plotly_chart(sensitivity_figure(st.session_state["direct_sensitivity"]), use_container_width=True)
        st.caption("Synthetic sensitivity experiment; each point uses 10,000 applications and the same seed.")
with export_tab:
    st.write("Downloads are constructed in memory and do not persist dashboard samples to the repository.")
    prefix = f"synthetic_{metadata['scenario']}_n{metadata['n_rows']}_seed{metadata['seed']}"
    cols = st.columns(4)
    cols[0].download_button("Download CSV", export_dataset_csv(data), f"{prefix}.csv", "text/csv")
    cols[1].download_button("Download Parquet", export_dataset_parquet(data), f"{prefix}.parquet", "application/octet-stream")
    cols[2].download_button("Download summary", export_summary_csv(summary), f"{prefix}_summary.csv", "text/csv")
    cols[3].download_button("Download config JSON", export_reproducibility_json(metadata), f"{prefix}_config.json", "application/json")

with st.expander("Reproducibility details"):
    st.json(
        {
            "scenario": metadata["scenario"],
            "effect_level": metadata["effect_level"],
            "rows": metadata["n_rows"],
            "seed": metadata["seed"],
            "configured_treatments": metadata["configured_treatments"],
            "frozen_intercept": metadata["intercept"],
            "configuration_fingerprint": metadata["config_fingerprint"],
            "persisted": metadata["persisted"],
        }
    )
disclaimer()
