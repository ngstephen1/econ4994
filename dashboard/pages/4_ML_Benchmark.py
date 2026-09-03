"""Interactive views of the validated Prompt 7 held-out ML benchmark."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from fair_lending.dashboard.charts import calibration_figure, regime_gap_figure
from fair_lending.dashboard.data import (
    DashboardArtifactError,
    filter_benchmark_row,
    load_result_table,
    load_test_predictions,
)
from fair_lending.dashboard.formatting import format_percentage, format_percentage_points
from fair_lending.dashboard.ui import configure_page, disclaimer, page_intro
from fair_lending.models.evaluation import calibration_bins
from fair_lending.models.group_audit import black_white_disparities, group_prediction_metrics


configure_page("ML benchmark", "🤖")
page_intro(
    "Machine-Learning Benchmark",
    "Explore held-out Prompt 7 results without retraining models on Streamlit reruns.",
)


@st.cache_data(show_spinner=False)
def cached_table(name: str) -> pd.DataFrame:
    return load_result_table(name)


@st.cache_data(show_spinner=False)
def cached_predictions() -> pd.DataFrame:
    return load_test_predictions()


SCENARIOS = {
    "Fair baseline": "fair_baseline",
    "Direct discrimination": "direct_discrimination",
    "Upstream inequality": "upstream_inequality",
    "Mixed mechanism": "mixed_mechanism",
}
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "Histogram Gradient Boosting": "hist_gradient_boosting",
}
REGIMES = {
    "Race-blind": "race_blind",
    "Race-aware sensitivity": "race_aware_sensitivity",
}

try:
    benchmark = cached_table("ml_benchmark")
    oracle = cached_table("ml_oracle_benchmark")
    calibration = cached_table("ml_calibration_bins")
except DashboardArtifactError as error:
    st.error(str(error))
    disclaimer()
    st.stop()

selector_columns = st.columns(3)
scenario_label = selector_columns[0].selectbox("Scenario", list(SCENARIOS))
model_label = selector_columns[1].selectbox("Model", list(MODELS))
regime_label = selector_columns[2].selectbox("Feature regime", list(REGIMES))
scenario = SCENARIOS[scenario_label]
model = MODELS[model_label]
regime = REGIMES[regime_label]
row = filter_benchmark_row(benchmark, scenario, model, regime)
oracle_row = oracle.loc[oracle["scenario"].eq(scenario)].iloc[0]

st.markdown("### Held-out performance")
metric_columns = st.columns(6)
for column, label, field in zip(
    metric_columns,
    ["Accuracy", "Balanced accuracy", "ROC-AUC", "PR-AUC", "Log loss", "Brier score"],
    ["accuracy", "balanced_accuracy", "roc_auc", "average_precision", "log_loss", "brier_score"],
    strict=True,
):
    column.metric(label, f"{float(row[field]):.3f}")
with st.expander("Synthetic oracle benchmark — not a trained model"):
    st.write(
        "The oracle uses `approval_probability_true`, which is available only because "
        "this is a simulation. It provides a lower-bound reference for probability error."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": ["ROC-AUC", "Log loss", "Brier score"],
                "Oracle value": [oracle_row["roc_auc"], oracle_row["log_loss"], oracle_row["brier_score"]],
            }
        ).style.format({"Oracle value": "{:.3f}"}),
        hide_index=True,
    )

st.markdown("### Race-blind versus race-aware mean predictions")
st.plotly_chart(regime_gap_figure(benchmark, scenario, model), use_container_width=True)
st.caption(
    "The observed bar is a held-out outcome gap. Model bars compare average predicted "
    "probabilities; none of these quantities alone establishes normative fairness."
)

st.markdown("### Threshold explorer")
threshold = st.slider("Global classification threshold", 0.20, 0.80, 0.50, 0.01)
st.caption(
    "The threshold changes the final classification rule but does not change the "
    "underlying predicted probabilities. One threshold is used for every group."
)
try:
    predictions = cached_predictions()
    selected_predictions = predictions.loc[
        predictions["scenario"].eq(scenario)
        & predictions["model"].eq(model)
        & predictions["feature_regime"].eq(regime)
    ].copy()
    if selected_predictions.empty:
        raise DashboardArtifactError("No held-out predictions matched the current selectors.")
    audit = group_prediction_metrics(
        selected_predictions,
        selected_predictions["predicted_probability"].to_numpy(),
        threshold,
    )
    disparity = black_white_disparities(audit)
    predicted = selected_predictions["predicted_probability"].to_numpy() >= threshold
    threshold_columns = st.columns(4)
    threshold_columns[0].metric("Overall predicted approval", format_percentage(float(predicted.mean())))
    threshold_columns[1].metric("White predicted approval", format_percentage(float(audit.set_index('race').loc['White', 'predicted_approval_rate'])))
    threshold_columns[2].metric("Black predicted approval", format_percentage(float(audit.set_index('race').loc['Black', 'predicted_approval_rate'])))
    threshold_columns[3].metric("Black − White predicted gap", format_percentage_points(disparity["predicted_approval_gap"]))
    focus = audit.loc[audit["race"].isin(["White", "Black"]), [
        "race", "n", "actual_approval_rate", "mean_predicted_probability",
        "predicted_approval_rate", "accuracy", "tpr", "fpr", "fnr", "tnr", "brier_score",
    ]].copy()
    focus.columns = [
        "Race", "n", "Actual approval", "Mean predicted probability", "Predicted approval",
        "Accuracy", "TPR", "FPR", "FNR", "TNR", "Brier score",
    ]
    st.dataframe(
        focus.style.format({
            "n": "{:,.0f}", "Actual approval": "{:.1%}", "Mean predicted probability": "{:.1%}",
            "Predicted approval": "{:.1%}", "Accuracy": "{:.1%}", "TPR": "{:.1%}",
            "FPR": "{:.1%}", "FNR": "{:.1%}", "TNR": "{:.1%}", "Brier score": "{:.3f}",
        }),
        hide_index=True,
        use_container_width=True,
    )
    st.warning(
        "These error rates measure agreement with simulated lender decisions. They do "
        "not measure qualification or normative fairness because no independent "
        "repayment/creditworthiness outcome is defined."
    )
except DashboardArtifactError as error:
    st.warning(str(error))

st.markdown("### Recovery of synthetic true approval probability")
recovery_columns = st.columns(3)
recovery_columns[0].metric("MAE", f"{float(row['true_probability_mae']):.4f}")
recovery_columns[1].metric("RMSE", f"{float(row['true_probability_rmse']):.4f}")
recovery_columns[2].metric("Correlation", f"{float(row['true_probability_correlation']):.3f}")
st.write(
    f"Mean model probability minus synthetic truth: "
    f"**{format_percentage_points(float(row['mean_prediction_minus_true_probability']))}**."
)

st.markdown("### Calibration")
bins = calibration.loc[
    calibration["scenario"].eq(scenario)
    & calibration["model"].eq(model)
    & calibration["feature_regime"].eq(regime)
]
st.plotly_chart(
    calibration_figure(bins, f"Calibration: {scenario_label}, {model_label}, {regime_label}"),
    use_container_width=True,
)
if 'selected_predictions' in locals():
    group_bins = []
    for race in ("White", "Black"):
        group = selected_predictions.loc[selected_predictions["race"].eq(race)]
        group_bins.append(
            calibration_bins(
                group["approved"], group["predicted_probability"], group["approval_probability_true"]
            ).assign(race=race)
        )
    st.plotly_chart(
        calibration_figure(pd.concat(group_bins, ignore_index=True), "White and Black group calibration", group_field="race"),
        use_container_width=True,
    )
disclaimer()
