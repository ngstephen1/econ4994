"""Small Plotly chart constructors with consistent research units."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


RACE_COLORS = {"White": "#5B7C99", "Black": "#C56A31"}
REGIME_COLORS = {"Observed": "#861F41", "Race-blind": "#159A9C", "Race-aware": "#E87722"}
SCENARIO_ORDER = [
    "fair_baseline",
    "direct_discrimination",
    "upstream_inequality",
    "mixed_mechanism",
]
SCENARIO_LABELS = {
    "fair_baseline": "Fair baseline",
    "direct_discrimination": "Direct discrimination",
    "upstream_inequality": "Upstream inequality",
    "mixed_mechanism": "Mixed mechanism",
}


def distribution_figure(data: pd.DataFrame, variable: str, unit: str) -> go.Figure:
    """Overlay normalized White and Black histograms for a selected variable."""
    frame = data.loc[data["race"].astype(object).isin(["White", "Black"])]
    figure = px.histogram(
        frame,
        x=variable,
        color="race",
        color_discrete_map=RACE_COLORS,
        histnorm="probability density",
        barmode="overlay",
        opacity=0.58,
        nbins=45,
        category_orders={"race": ["White", "Black"]},
        labels={variable: unit, "race": "Race", "value": "Density"},
        title=f"White and Black distributions: {variable.replace('_', ' ')}",
    )
    figure.update_layout(yaxis_title="Probability density", legend_title="Race")
    return figure


def approval_rate_figure(race_table: pd.DataFrame) -> go.Figure:
    frame = race_table.copy()
    frame["approval_percent"] = 100.0 * frame["approval_rate"]
    figure = px.bar(
        frame,
        x="race",
        y="approval_percent",
        color="race",
        color_discrete_map=RACE_COLORS,
        text=frame["approval_percent"].map(lambda value: f"{value:.1f}%"),
        labels={"race": "Race", "approval_percent": "Approval rate (%)"},
        title="Unadjusted simulated approval rate by race",
    )
    figure.update_layout(showlegend=False, yaxis_range=[0, 100])
    return figure


def coefficient_path_figure(results: pd.DataFrame) -> go.Figure:
    frame = results.copy()
    frame["ci_minus"] = frame["black_coefficient"] - frame["black_ci_low"]
    frame["ci_plus"] = frame["black_ci_high"] - frame["black_coefficient"]
    figure = go.Figure(
        go.Scatter(
            x=frame["model_label"],
            y=frame["black_coefficient"],
            mode="lines+markers",
            error_y={
                "type": "data",
                "array": frame["ci_plus"],
                "arrayminus": frame["ci_minus"],
                "visible": True,
            },
            name="Estimated Black coefficient",
            line={"color": "#5B7C99"},
        )
    )
    figure.add_hline(y=0.0, line_color="#333333", line_dash="dash")
    figure.update_layout(
        title="Black log-odds coefficient across regression specifications",
        xaxis_title="Specification",
        yaxis_title="Black coefficient (log odds), with 95% CI",
    )
    return figure


def regime_gap_figure(benchmark: pd.DataFrame, scenario: str, model: str) -> go.Figure:
    subset = benchmark.loc[
        benchmark["scenario"].eq(scenario) & benchmark["model"].eq(model)
    ].set_index("feature_regime")
    values = pd.DataFrame(
        {
            "Series": ["Observed", "Race-blind", "Race-aware"],
            "Gap": [
                float(subset.loc["race_blind", "observed_black_white_gap"]),
                float(subset.loc["race_blind", "predicted_probability_gap"]),
                float(subset.loc["race_aware_sensitivity", "predicted_probability_gap"]),
            ],
        }
    )
    values["Gap (pp)"] = 100.0 * values["Gap"]
    figure = px.bar(
        values,
        x="Gap (pp)",
        y="Series",
        orientation="h",
        color="Series",
        color_discrete_map=REGIME_COLORS,
        text=values["Gap (pp)"].map(lambda value: f"{value:+.2f} pp"),
        title="Observed and model mean-prediction gaps (Black minus White)",
    )
    figure.add_vline(x=0.0, line_color="#333333")
    figure.update_layout(showlegend=False, xaxis_title="Black–White gap (percentage points)")
    return figure


def calibration_figure(
    bins: pd.DataFrame, title: str, *, group_field: str | None = None
) -> go.Figure:
    color = group_field if group_field is not None else None
    figure = px.line(
        bins,
        x="mean_predicted_probability",
        y="observed_approval_rate",
        color=color,
        markers=True,
        color_discrete_map=RACE_COLORS,
        labels={
            "mean_predicted_probability": "Mean predicted probability",
            "observed_approval_rate": "Observed approval rate",
            "race": "Race",
        },
        title=title,
    )
    if "mean_true_probability" in bins and group_field is None:
        figure.add_trace(
            go.Scatter(
                x=bins["mean_predicted_probability"],
                y=bins["mean_true_probability"],
                mode="lines+markers",
                name="Synthetic true mean probability",
                line={"dash": "dot", "color": "#6F4E7C"},
            )
        )
    figure.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#555"})
    figure.update_xaxes(range=[0, 1])
    figure.update_yaxes(range=[0, 1])
    return figure


def mechanism_gap_figure(matrix: pd.DataFrame) -> go.Figure:
    columns = {
        "Observed gap": "observed_gap",
        "Adjusted regression": "adjusted_probability_gap",
        "Race-blind ML": "race_blind_ml_gap",
        "Race-aware ML": "race_aware_ml_gap",
    }
    long = matrix.melt(
        id_vars=["scenario_label"],
        value_vars=list(columns.values()),
        var_name="measure",
        value_name="gap",
    )
    inverse = {value: key for key, value in columns.items()}
    long["Measure"] = long["measure"].map(inverse)
    long["Gap (pp)"] = 100.0 * long["gap"]
    figure = px.bar(
        long,
        x="scenario_label",
        y="Gap (pp)",
        color="Measure",
        barmode="group",
        labels={"scenario_label": "Scenario"},
        title="Black–White gaps across observed, statistical, and ML measures",
    )
    figure.add_hline(y=0.0, line_color="#333333")
    return figure


def sensitivity_figure(frame: pd.DataFrame) -> go.Figure:
    figure = px.line(
        frame,
        x="direct_black_log_odds",
        y="gap_percentage_points",
        markers=True,
        labels={
            "direct_black_log_odds": "Configured Black direct effect (log odds)",
            "gap_percentage_points": "Observed Black–White approval gap (pp)",
        },
        title="Synthetic sensitivity experiment: direct-effect strength",
    )
    figure.add_hline(y=0.0, line_color="#333333", line_dash="dash")
    return figure
