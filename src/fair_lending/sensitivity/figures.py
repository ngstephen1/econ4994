"""Research-ready figures for the sensitivity and Monte Carlo experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fair_lending.sensitivity.metrics import detection_table
from fair_lending.sensitivity.runner import records_frame
from fair_lending.simulation.config import PROJECT_ROOT


COLORS = {
    "raw": "#861F41",
    "adjusted": "#159A9C",
    "blind": "#5B7C99",
    "aware": "#E87722",
    "truth": "#6F4E7C",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _summary(frame: pd.DataFrame, x: str, values: list[str]) -> pd.DataFrame:
    aggregation: dict[str, tuple[str, Any]] = {}
    for value in values:
        aggregation[f"{value}_mean"] = (value, "mean")
        aggregation[f"{value}_q025"] = (value, lambda item: item.quantile(0.025))
        aggregation[f"{value}_q975"] = (value, lambda item: item.quantile(0.975))
    return frame.groupby(x, as_index=False).agg(**aggregation).sort_values(x)


def _curve(
    summary: pd.DataFrame,
    x: str,
    series: list[tuple[str, str, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for field, label, color in series:
        mean = summary[f"{field}_mean"].to_numpy()
        low = summary[f"{field}_q025"].to_numpy()
        high = summary[f"{field}_q975"].to_numpy()
        x_values = summary[x].to_numpy()
        ax.plot(x_values, 100.0 * mean, marker="o", label=label, color=color)
        ax.fill_between(x_values, 100.0 * low, 100.0 * high, color=color, alpha=0.12)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if len(series) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _heatmap(frame: pd.DataFrame, value: str, title: str, path: Path) -> None:
    table = frame.pivot_table(
        index="upstream_strength",
        columns="direct_black_log_odds",
        values=value,
        aggfunc="mean",
    ).sort_index(ascending=False)
    table = table.reindex(sorted(table.columns, reverse=True), axis=1)
    values = 100.0 * table.to_numpy()
    bound = max(abs(np.nanmin(values)), abs(np.nanmax(values)), 0.1)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(values, cmap="PuOr", vmin=-bound, vmax=bound, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:+.1f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(table.columns)), [f"{value:g}" for value in table.columns])
    ax.set_yticks(range(len(table.index)), [f"{value:g}" for value in table.index])
    ax.set_xlabel("Configured Black direct effect (log odds)")
    ax.set_ylabel("Upstream strength")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Black − White gap (percentage points)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    records: list[dict[str, Any]], project_root: Path | str | None = None
) -> dict[str, Path]:
    """Generate the declared compact set of 13 sensitivity figures."""
    root = Path(project_root) if project_root is not None else PROJECT_ROOT
    directory = root / "results" / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    all_runs = records_frame(records, {"direct", "upstream", "mixed", "sample_size", "detection"})
    all_runs = all_runs.drop_duplicates("run_id")
    mixed_runs = records_frame(records, {"mixed"}).drop_duplicates("run_id")
    maximum_rows = int(all_runs["n_rows"].max())
    direct = all_runs.loc[
        all_runs["upstream_strength"].eq(0.0) & all_runs["n_rows"].eq(maximum_rows)
    ]
    upstream = all_runs.loc[
        all_runs["direct_black_log_odds"].eq(0.0) & all_runs["n_rows"].eq(maximum_rows)
    ]
    mixed = mixed_runs.loc[mixed_runs["n_rows"].eq(maximum_rows)]
    paths = {
        "direct_observed": directory / "sensitivity_direct_observed_gap.png",
        "direct_adjusted": directory / "sensitivity_direct_adjusted_gap.png",
        "direct_ml": directory / "sensitivity_direct_ml_gap.png",
        "upstream_observed": directory / "sensitivity_upstream_observed_gap.png",
        "upstream_adjusted": directory / "sensitivity_upstream_adjusted_gap.png",
        "upstream_ml": directory / "sensitivity_upstream_ml_gap.png",
        "mixed_raw": directory / "sensitivity_mixed_raw_gap_heatmap.png",
        "mixed_adjusted": directory / "sensitivity_mixed_adjusted_gap_heatmap.png",
        "mixed_blind": directory / "sensitivity_mixed_race_blind_ml_heatmap.png",
        "sample_uncertainty": directory / "sensitivity_sample_size_coefficient_uncertainty.png",
        "detection": directory / "sensitivity_detection_probability.png",
        "coverage": directory / "sensitivity_ci_coverage.png",
        "fair_variation": directory / "sensitivity_fair_raw_gap_distribution.png",
    }
    _style()
    direct_summary = _summary(
        direct,
        "direct_black_log_odds",
        [
            "raw_approval_gap",
            "model_2_adjusted_probability_gap",
            "race_blind_predicted_probability_gap",
            "race_aware_predicted_probability_gap",
        ],
    )
    _curve(direct_summary, "direct_black_log_odds", [("raw_approval_gap", "Raw outcome gap", COLORS["raw"])], "Direct penalty and observed approval disparity", "Configured Black direct effect (log odds)", "Black − White gap (percentage points)", paths["direct_observed"])
    _curve(direct_summary, "direct_black_log_odds", [("model_2_adjusted_probability_gap", "Model 2 adjusted gap", COLORS["adjusted"])], "Direct penalty and adjusted regression contrast", "Configured Black direct effect (log odds)", "Adjusted gap (percentage points)", paths["direct_adjusted"])
    _curve(direct_summary, "direct_black_log_odds", [("race_blind_predicted_probability_gap", "Race-blind", COLORS["blind"]), ("race_aware_predicted_probability_gap", "Race-aware sensitivity", COLORS["aware"])], "Direct penalty and ML mean-prediction disparity", "Configured Black direct effect (log odds)", "Prediction gap (percentage points)", paths["direct_ml"])

    upstream_summary = _summary(
        upstream,
        "upstream_strength",
        [
            "raw_approval_gap",
            "model_2_adjusted_probability_gap",
            "race_blind_predicted_probability_gap",
        ],
    )
    _curve(upstream_summary, "upstream_strength", [("raw_approval_gap", "Raw outcome gap", COLORS["raw"])], "Upstream strength and observed approval disparity", "Upstream strength (1.0 = moderate)", "Black − White gap (percentage points)", paths["upstream_observed"])
    _curve(upstream_summary, "upstream_strength", [("model_2_adjusted_probability_gap", "Model 2 adjusted gap", COLORS["adjusted"])], "Upstream strength and adjusted regression contrast", "Upstream strength (1.0 = moderate)", "Adjusted gap (percentage points)", paths["upstream_adjusted"])
    _curve(upstream_summary, "upstream_strength", [("race_blind_predicted_probability_gap", "Race-blind", COLORS["blind"])], "Upstream strength and race-blind ML disparity", "Upstream strength (1.0 = moderate)", "Prediction gap (percentage points)", paths["upstream_ml"])

    _heatmap(mixed, "raw_approval_gap", "Mixed mechanisms: raw approval gap", paths["mixed_raw"])
    _heatmap(mixed, "model_2_adjusted_probability_gap", "Mixed mechanisms: adjusted regression gap", paths["mixed_adjusted"])
    _heatmap(mixed, "race_blind_predicted_probability_gap", "Mixed mechanisms: race-blind ML gap", paths["mixed_blind"])

    selected = all_runs.loc[
        all_runs["direct_black_log_odds"].eq(-0.25)
        & all_runs["upstream_strength"].isin([0.0, 1.0])
    ].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for upstream_value, label, color in ((0.0, "Moderate direct", COLORS["aware"]), (1.0, "Moderate mixed", COLORS["truth"])):
        values = selected.loc[selected["upstream_strength"].eq(upstream_value)]
        summary = values.groupby("n_rows")["model_2_black_coefficient"].agg(
            mean="mean", sd="std"
        ).reset_index()
        ax.errorbar(summary["n_rows"], summary["mean"], yerr=summary["sd"], marker="o", capsize=3, label=label, color=color)
    ax.axhline(-0.25, color="#333333", linestyle="--", label="Configured direct effect")
    ax.set_xscale("log")
    ax.set(xlabel="Applications (log scale)", ylabel="Model 2 coefficient, mean ± Monte Carlo SD", title="Sample size and coefficient uncertainty")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(paths["sample_uncertainty"], bbox_inches="tight"); plt.close(fig)

    detection = detection_table(all_runs)
    direct_detection = detection.loc[detection["upstream_strength"].eq(0.0)].copy()
    preferred_sizes = [5_000, 10_000, 25_000, 50_000, 100_000]
    available_preferred = sorted(
        set(direct_detection["n_rows"]).intersection(preferred_sizes)
    )
    selected_sizes = available_preferred or sorted(direct_detection["n_rows"].unique())
    power = direct_detection.loc[
        direct_detection["n_rows"].isin(selected_sizes)
    ].copy()
    power["direct_magnitude"] = power["direct_black_log_odds"].abs()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for n_rows, values in power.groupby("n_rows"):
        values = values.sort_values("direct_magnitude")
        ax.plot(values["direct_magnitude"], values["model_2_negative_detection_probability"], marker="o", label=f"n={n_rows:,}")
    ax.axhline(0.80, color="#555555", linestyle="--", linewidth=1.0)
    ax.set(xlabel="Absolute direct log-odds penalty", ylabel="Monte Carlo detection probability", title="Model 2 negative-effect detection probability")
    ax.set_ylim(0, 1.02)
    if len(power):
        ax.legend(frameon=False, ncols=2)
    fig.tight_layout(); fig.savefig(paths["detection"], bbox_inches="tight"); plt.close(fig)

    coverage = detection.loc[detection["direct_black_log_odds"].eq(-0.25)]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for upstream_value, label, color in ((0.0, "Direct", COLORS["aware"]), (1.0, "Mixed", COLORS["truth"])):
        values = coverage.loc[coverage["upstream_strength"].eq(upstream_value)].sort_values("n_rows")
        ax.plot(values["n_rows"], values["model_2_coverage_probability"], marker="o", label=label, color=color)
    ax.axhline(0.95, color="#555555", linestyle="--", linewidth=1.0)
    ax.set_xscale("log"); ax.set_ylim(0, 1.02)
    ax.set(xlabel="Applications (log scale)", ylabel="Empirical 95% CI coverage", title="Model 2 coverage of configured direct effect")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(paths["coverage"], bbox_inches="tight"); plt.close(fig)

    fair = all_runs.loc[
        all_runs["direct_black_log_odds"].eq(0.0)
        & all_runs["upstream_strength"].eq(0.0)
        & all_runs["n_rows"].eq(maximum_rows)
    ].drop_duplicates("run_id")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.hist(100.0 * fair["raw_approval_gap"], bins="auto", color=COLORS["raw"], alpha=0.8)
    ax.axvline(0.0, color="#333333", linestyle="--")
    ax.set(xlabel="Black − White raw approval gap (percentage points)", ylabel="Replications", title=f"Fair-baseline finite-sample gaps (n={maximum_rows:,})")
    fig.tight_layout(); fig.savefig(paths["fair_variation"], bbox_inches="tight"); plt.close(fig)
    return paths
