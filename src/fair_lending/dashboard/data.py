"""Centralized access to generated statistical and ML benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fair_lending.simulation.config import PROJECT_ROOT


TABLE_PATHS = {
    "race_approval_gaps": "race_approval_gaps.csv",
    "logit_black_effects": "logit_black_effects.csv",
    "statistical_recovery": "statistical_recovery.csv",
    "ml_benchmark": "ml_benchmark.csv",
    "ml_group_audit": "ml_group_audit.csv",
    "ml_race_regime_comparison": "ml_race_regime_comparison.csv",
    "ml_oracle_benchmark": "ml_oracle_benchmark.csv",
    "ml_calibration_bins": "ml_calibration_bins.csv",
    "ml_group_calibration_bins": "ml_group_calibration_bins.csv",
    "statistical_ml_comparison": "statistical_ml_comparison.csv",
    "sensitivity_direct_summary": "sensitivity_direct_summary.csv",
    "sensitivity_upstream_summary": "sensitivity_upstream_summary.csv",
    "sensitivity_mixed_summary": "sensitivity_mixed_summary.csv",
    "sensitivity_sample_size_summary": "sensitivity_sample_size_summary.csv",
    "monte_carlo_detection": "monte_carlo_detection.csv",
    "monte_carlo_coverage": "monte_carlo_coverage.csv",
    "mechanism_signatures": "mechanism_signatures.csv",
    "detection_thresholds": "detection_thresholds.csv",
    "sensitivity_paper_summary": "sensitivity_paper_summary.csv",
}
PREDICTION_FILENAME = "ml_test_predictions.parquet"
METADATA_FILENAMES = {
    "statistical": "statistical_recovery_metadata.json",
    "ml": "ml_benchmark_metadata.json",
}


class DashboardArtifactError(FileNotFoundError):
    """Raised with a research-facing recovery instruction for missing outputs."""


def _root(project_root: Path | str | None) -> Path:
    return Path(project_root) if project_root is not None else PROJECT_ROOT


def load_result_table(
    name: str, project_root: Path | str | None = None
) -> pd.DataFrame:
    """Load one declared CSV result table or provide a controlled error."""
    if name not in TABLE_PATHS:
        raise KeyError(f"Unknown dashboard result table: {name}")
    path = _root(project_root) / "results" / "tables" / TABLE_PATHS[name]
    if not path.exists():
        if name.startswith("sensitivity_") or name in {
            "monte_carlo_detection",
            "monte_carlo_coverage",
            "mechanism_signatures",
            "detection_thresholds",
        }:
            experiment = "experiments/run_sensitivity.py --experiment all --resume"
        elif name.startswith("ml_") or name == "statistical_ml_comparison":
            experiment = "experiments/run_ml_benchmark.py"
        else:
            experiment = "experiments/run_statistical_recovery.py"
        raise DashboardArtifactError(
            f"Required result table '{path.name}' was not found. Run {experiment} first."
        )
    return pd.read_csv(path)


def load_test_predictions(
    project_root: Path | str | None = None,
) -> pd.DataFrame:
    """Load held-out probabilities used by the no-retraining threshold explorer."""
    path = _root(project_root) / "results" / "metrics" / PREDICTION_FILENAME
    if not path.exists():
        raise DashboardArtifactError(
            "Held-out benchmark predictions were not found. Run "
            "experiments/run_ml_benchmark.py first."
        )
    return pd.read_parquet(path)


def load_experiment_metadata(
    experiment: str, project_root: Path | str | None = None
) -> dict[str, Any]:
    """Load declared experiment metadata with the same controlled failure mode."""
    if experiment not in METADATA_FILENAMES:
        raise KeyError(f"Unknown experiment metadata: {experiment}")
    path = _root(project_root) / "results" / "metrics" / METADATA_FILENAMES[experiment]
    if not path.exists():
        script = (
            "experiments/run_ml_benchmark.py"
            if experiment == "ml"
            else "experiments/run_statistical_recovery.py"
        )
        raise DashboardArtifactError(
            f"Required experiment metadata '{path.name}' was not found. Run {script} first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def filter_benchmark_row(
    benchmark: pd.DataFrame, scenario: str, model: str, feature_regime: str
) -> pd.Series:
    """Return exactly one benchmark row for dashboard selectors."""
    selected = benchmark.loc[
        benchmark["scenario"].eq(scenario)
        & benchmark["model"].eq(model)
        & benchmark["feature_regime"].eq(feature_regime)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected one benchmark row for "
            f"scenario={scenario}, model={model}, regime={feature_regime}; "
            f"found {len(selected)}"
        )
    return selected.iloc[0]


def build_mechanism_matrix(
    statistical_recovery: pd.DataFrame,
    ml_benchmark: pd.DataFrame,
    *,
    model: str = "logistic_regression",
) -> pd.DataFrame:
    """Combine saved Prompt 6 and Prompt 7 results with declared mechanisms."""
    statistical = statistical_recovery.loc[
        statistical_recovery["model"].eq("model_2")
    ].set_index("scenario")
    machine_learning = ml_benchmark.loc[ml_benchmark["model"].eq(model)].set_index(
        ["scenario", "feature_regime"]
    )
    labels = {
        "fair_baseline": "Fair baseline",
        "direct_discrimination": "Direct discrimination",
        "upstream_inequality": "Upstream inequality",
        "mixed_mechanism": "Mixed mechanism",
    }
    switches = {
        "fair_baseline": (False, False),
        "direct_discrimination": (False, True),
        "upstream_inequality": (True, False),
        "mixed_mechanism": (True, True),
    }
    rows = []
    for scenario, label in labels.items():
        upstream, direct = switches[scenario]
        stat = statistical.loc[scenario]
        blind = machine_learning.loc[(scenario, "race_blind")]
        aware = machine_learning.loc[(scenario, "race_aware_sensitivity")]
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": label,
                "upstream_differences": upstream,
                "direct_race_effect": direct,
                "observed_gap": float(stat["raw_approval_gap"]),
                "adjusted_black_coefficient": float(stat["black_coefficient"]),
                "adjusted_probability_gap": float(stat["adjusted_probability_gap"]),
                "race_blind_ml_gap": float(blind["predicted_probability_gap"]),
                "race_aware_ml_gap": float(aware["predicted_probability_gap"]),
                "true_direct_log_odds": float(stat["true_direct_log_odds"]),
            }
        )
    return pd.DataFrame(rows)


def data_dictionary() -> pd.DataFrame:
    """Return the documented 24-field schema grouped for dashboard reading."""
    rows = [
        ("Identifiers", "application_id", "String", "Unique synthetic mortgage-application identifier."),
        ("Protected attributes", "race", "Category", "Protected race group; audit variable and focal synthetic treatment."),
        ("Protected attributes", "ethnicity", "Category", "Hispanic/Latino or not Hispanic/Latino audit attribute."),
        ("Protected attributes", "sex", "Category", "Male or Female audit attribute in the version-1 scope."),
        ("Protected attributes", "age_group", "Ordered category", "Applicant age band; audit attribute and employment-history constraint."),
        ("Borrower financial characteristics", "annual_income", "USD/year", "Modeled annual repayment capacity; possible upstream mediator."),
        ("Borrower financial characteristics", "credit_score", "Integer, 500–850", "Synthetic modeled creditworthiness score; possible upstream mediator."),
        ("Borrower financial characteristics", "employment_years", "Years", "Modeled accumulated employment history."),
        ("Borrower financial characteristics", "liquid_assets", "USD", "Modeled liquid resources and reserve capacity; possible upstream mediator."),
        ("Borrower financial characteristics", "existing_monthly_debt", "USD/month", "Existing monthly obligations before the proposed mortgage."),
        ("Borrower financial characteristics", "debt_to_income_ratio", "Proportion, 0–0.65", "Back-end DTI including an internal projected housing payment."),
        ("Loan/property characteristics", "loan_amount", "USD", "Requested principal, derived from property value and LTV."),
        ("Loan/property characteristics", "property_value", "USD", "Modeled value of the property associated with the application."),
        ("Loan/property characteristics", "loan_to_value_ratio", "Proportion", "Requested loan divided by property value."),
        ("Loan/property characteristics", "loan_purpose", "Category", "Home purchase, refinance, or home improvement."),
        ("Loan/property characteristics", "loan_type", "Category", "Conventional, FHA, VA, or USDA/RHS product family."),
        ("Loan/property characteristics", "occupancy_type", "Category", "Principal, second-home, or investment-property occupancy."),
        ("Context", "neighborhood_income_index", "Index", "Modeled local socioeconomic context relative to a reference level."),
        ("Context", "neighborhood_minority_share", "Proportion", "Modeled neighborhood composition; excluded from primary predictors due to proxy risk."),
        ("Context", "local_unemployment_rate", "Proportion", "Modeled local labor-market conditions."),
        ("Derived variables", "income_to_loan_ratio", "Ratio", "Annual income divided by requested loan amount."),
        ("Outcomes / synthetic truth", "approval_probability_true", "Probability", "Exact DGP approval probability; hidden synthetic truth, never a predictor."),
        ("Outcomes / synthetic truth", "approved", "Binary", "Simulated lender decision: 1 approved, 0 denied."),
        ("Outcomes / synthetic truth", "denial_reason", "Nullable category", "Reserved post-decision field; null throughout version 1."),
    ]
    return pd.DataFrame(rows, columns=["Group", "Field", "Type / unit", "Description"])
