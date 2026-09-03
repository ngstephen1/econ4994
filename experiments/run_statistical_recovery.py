"""Run the first deterministic descriptive and logistic recovery experiment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels

from fair_lending.analysis.descriptive import (
    black_white_raw_gap,
    descriptive_outcomes,
)
from fair_lending.analysis.estimands import (
    standardized_black_white_contrast,
    true_direct_effect,
)
from fair_lending.analysis.logit import MODEL_ORDER, fit_logit_sequence
from fair_lending.simulation.config import (
    PROJECT_ROOT,
    git_revision,
    package_version,
)
from fair_lending.simulation.generator import (
    generate_synthetic_data,
    save_synthetic_dataset,
)
from fair_lending.simulation.validation import validate_generated_data


SCENARIOS = (
    "fair_baseline",
    "direct_discrimination",
    "upstream_inequality",
    "mixed_mechanism",
)
SCENARIO_LABELS = {
    "fair_baseline": "Fair baseline",
    "direct_discrimination": "Direct discrimination",
    "upstream_inequality": "Upstream inequality",
    "mixed_mechanism": "Mixed mechanism",
}
N_ROWS = 100_000
SEED = 4_994
EFFECT_LEVEL = "moderate"
MAIN_MODEL = "model_2"
COLORS = {
    "white": "#7083A5",
    "black": "#E87722",
    "raw": "#861F41",
    "adjusted": "#159A9C",
    "estimate": "#3A506B",
    "truth": "#E87722",
}


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def _figure_approval_rates(descriptive: pd.DataFrame, path: Path) -> None:
    race_rows = descriptive.loc[
        (descriptive["row_type"] == "race")
        & descriptive["race"].isin(["White", "Black"])
    ]
    white = race_rows.loc[race_rows["race"] == "White"].set_index("scenario")
    black = race_rows.loc[race_rows["race"] == "Black"].set_index("scenario")
    x = np.arange(len(SCENARIOS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(
        x - width / 2,
        100.0 * white.loc[list(SCENARIOS), "approval_rate"],
        width,
        label="White",
        color=COLORS["white"],
    )
    ax.bar(
        x + width / 2,
        100.0 * black.loc[list(SCENARIOS), "approval_rate"],
        width,
        label="Black",
        color=COLORS["black"],
    )
    ax.set_ylabel("Approval rate (%)")
    ax.set_title("Black and White approval rates across synthetic scenarios")
    ax.set_xticks(x, [SCENARIO_LABELS[item] for item in SCENARIOS], rotation=16, ha="right")
    ax.legend(frameon=False, ncols=2)
    ax.set_ylim(0.0, 100.0)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _figure_raw_adjusted_gaps(
    gaps: pd.DataFrame, recovery: pd.DataFrame, path: Path
) -> None:
    raw = gaps.set_index("scenario").loc[list(SCENARIOS), "raw_approval_gap"]
    adjusted = (
        recovery.loc[recovery["model"] == MAIN_MODEL]
        .set_index("scenario")
        .loc[list(SCENARIOS), "adjusted_probability_gap"]
    )
    x = np.arange(len(SCENARIOS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.bar(
        x - width / 2,
        100.0 * raw,
        width,
        label="Raw approval-rate gap",
        color=COLORS["raw"],
    )
    ax.bar(
        x + width / 2,
        100.0 * adjusted,
        width,
        label="Model 2 adjusted contrast",
        color=COLORS["adjusted"],
    )
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_ylabel("Black − White gap (percentage points)")
    ax.set_title("Raw and adjusted Black–White probability gaps")
    ax.set_xticks(x, [SCENARIO_LABELS[item] for item in SCENARIOS], rotation=16, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _figure_main_coefficients(recovery: pd.DataFrame, path: Path) -> None:
    main = (
        recovery.loc[recovery["model"] == MAIN_MODEL]
        .set_index("scenario")
        .loc[list(SCENARIOS)]
    )
    y = np.arange(len(SCENARIOS))
    coefficient = main["black_coefficient"].to_numpy()
    errors = np.vstack(
        [
            coefficient - main["black_ci_low"].to_numpy(),
            main["black_ci_high"].to_numpy() - coefficient,
        ]
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.errorbar(
        coefficient,
        y,
        xerr=errors,
        fmt="o",
        color=COLORS["estimate"],
        ecolor=COLORS["estimate"],
        capsize=4,
        label="Estimated coefficient (95% CI)",
    )
    ax.scatter(
        main["true_direct_log_odds"],
        y,
        marker="D",
        s=42,
        color=COLORS["truth"],
        label="Configured direct effect",
        zorder=3,
    )
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_yticks(y, [SCENARIO_LABELS[item] for item in SCENARIOS])
    ax.invert_yaxis()
    ax.set_xlabel("Black log-odds coefficient")
    ax.set_title("Model 2 recovery of the configured direct race effect")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_experiment() -> dict[str, Path]:
    """Generate samples, fit models, and persist the predeclared artifacts."""
    tables_directory = PROJECT_ROOT / "results" / "tables"
    figures_directory = PROJECT_ROOT / "results" / "figures"
    metrics_directory = PROJECT_ROOT / "results" / "metrics"
    for directory in (tables_directory, figures_directory, metrics_directory):
        directory.mkdir(parents=True, exist_ok=True)

    descriptive_frames = []
    raw_gaps = []
    logit_rows: list[dict] = []
    recovery_rows: list[dict] = []
    dataset_records = []

    for scenario in SCENARIOS:
        data, metadata = generate_synthetic_data(
            scenario=scenario,
            effect_level=EFFECT_LEVEL,
            n_rows=N_ROWS,
            seed=SEED,
        )
        dataset_path, metadata_path, saved_metadata = save_synthetic_dataset(
            data, metadata
        )
        validation = validate_generated_data(data, saved_metadata)
        if not validation["valid"] or validation["warnings"]:
            raise RuntimeError(
                f"Generator validation failed for {scenario}: "
                f"errors={validation['errors']}, warnings={validation['warnings']}"
            )

        descriptive_frames.append(
            descriptive_outcomes(data, saved_metadata["resolved_configuration"], scenario)
        )
        raw_gap = black_white_raw_gap(data, scenario)
        raw_gaps.append(raw_gap)

        fitted_models = fit_logit_sequence(
            data, saved_metadata["resolved_configuration"]
        )
        truth = true_direct_effect(
            fitted_models[0]["sample"],
            saved_metadata["resolved_configuration"],
            saved_metadata["intercept"],
        )
        for fitted in fitted_models:
            summary = dict(fitted["summary"])
            summary["scenario"] = scenario
            summary["matrix_columns"] = " | ".join(summary["matrix_columns"])
            summary["duplicate_column_pairs"] = json.dumps(
                summary["duplicate_column_pairs"]
            )
            summary["zero_variance_columns"] = json.dumps(
                summary["zero_variance_columns"]
            )
            adjusted_gap = standardized_black_white_contrast(
                fitted["result"], fitted["design_matrix"]
            )
            summary["adjusted_probability_gap"] = adjusted_gap
            summary["adjusted_probability_gap_percentage_points"] = 100.0 * adjusted_gap
            logit_rows.append(summary)
            recovery_rows.append(
                {
                    "scenario": scenario,
                    "model": summary["model"],
                    "n": summary["n"],
                    "raw_approval_gap": raw_gap["raw_approval_gap"],
                    "black_coefficient": summary["black_coefficient"],
                    "black_se": summary["black_se"],
                    "black_ci_low": summary["black_ci_low"],
                    "black_ci_high": summary["black_ci_high"],
                    "black_odds_ratio": summary["black_odds_ratio"],
                    "adjusted_probability_gap": adjusted_gap,
                    **truth,
                    "coefficient_error": summary["black_coefficient"]
                    - truth["true_direct_log_odds"],
                    "probability_gap_error": adjusted_gap
                    - truth["true_direct_probability_gap"],
                }
            )

        dataset_records.append(
            {
                "scenario": scenario,
                "dataset_path": str(dataset_path.resolve()),
                "metadata_path": str(metadata_path.resolve()),
                "config_fingerprint": saved_metadata["config_fingerprint"],
                "intercept": saved_metadata["intercept"],
                "rows": len(data),
                "columns": len(data.columns),
            }
        )

    descriptive = pd.concat(descriptive_frames, ignore_index=True)
    gaps = pd.DataFrame(raw_gaps)
    logit_effects = pd.DataFrame(logit_rows)
    recovery = pd.DataFrame(recovery_rows)

    paths = {
        "descriptive": tables_directory / "descriptive_by_scenario.csv",
        "gaps": tables_directory / "race_approval_gaps.csv",
        "logit": tables_directory / "logit_black_effects.csv",
        "recovery": tables_directory / "statistical_recovery.csv",
        "figure_approval": figures_directory / "approval_rates_by_scenario.png",
        "figure_gaps": figures_directory / "raw_vs_adjusted_gaps.png",
        "figure_coefficients": figures_directory / "model2_black_coefficients.png",
        "metadata": metrics_directory / "statistical_recovery_metadata.json",
    }
    descriptive.to_csv(paths["descriptive"], index=False)
    gaps.to_csv(paths["gaps"], index=False)
    logit_effects.to_csv(paths["logit"], index=False)
    recovery.to_csv(paths["recovery"], index=False)

    _configure_plot_style()
    _figure_approval_rates(descriptive, paths["figure_approval"])
    _figure_raw_adjusted_gaps(gaps, recovery, paths["figure_gaps"])
    _figure_main_coefficients(recovery, paths["figure_coefficients"])

    revision, dirty = git_revision()
    analysis_metadata = {
        "experiment": "first_statistical_recovery",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": list(SCENARIOS),
        "effect_level": EFFECT_LEVEL,
        "rows_per_scenario": N_ROWS,
        "seed": SEED,
        "outcome": "approved (1=approved, 0=denied)",
        "regression_sample": "White and Black applications",
        "main_adjusted_model": MAIN_MODEL,
        "package_version": package_version(),
        "statsmodels_version": statsmodels.__version__,
        "git_revision": revision,
        "git_worktree_dirty": dirty,
        "datasets": dataset_records,
        "model_order": list(MODEL_ORDER),
        "raw_gap_ci_method": "unpooled Wald difference in independent proportions",
        "artifacts": {name: str(path.resolve()) for name, path in paths.items() if name != "metadata"},
    }
    paths["metadata"].write_text(
        json.dumps(analysis_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    paths = run_experiment()
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
