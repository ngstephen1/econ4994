"""Validation summaries for the Version 2 economic population."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fair_lending.economic_lending.profit import expected_profit_from_options


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
    }


def group_diagnostics(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    loan_outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Return observed baseline summaries by audit group."""

    joined = (
        applicants.merge(
            loan_options.loc[:, ["applicant_id", "first_period_dti", "requested_ltv"]],
            on="applicant_id",
            validate="one_to_one",
        )
        .merge(simulation_truth, on="applicant_id", validate="one_to_one")
        .merge(
            loan_outcomes.loc[:, ["applicant_id", "completed_all_payments"]],
            on="applicant_id",
            validate="one_to_one",
        )
    )
    fields = [
        "annual_income",
        "credit_score",
        "employment_years",
        "liquid_assets",
        "existing_monthly_debt",
        "property_value",
        "requested_loan_amount",
        "first_period_dti",
        "requested_ltv",
        "repayment_probability_per_period_true",
        "full_repayment_probability_true",
        "completed_all_payments",
    ]
    summary = joined.groupby("group", observed=True)[fields].mean().reset_index()
    summary.insert(1, "n", joined.groupby("group", observed=True).size().to_numpy())
    return summary


def population_validation_report(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    simulation_truth: pd.DataFrame,
    loan_outcomes: pd.DataFrame,
    config: dict[str, Any],
    clipping: dict[str, dict[str, float]],
    spawn_keys: dict[str, list[int]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build a JSON-serializable validation report and group audit table."""

    assessments = expected_profit_from_options(loan_options, simulation_truth)
    rho = simulation_truth["repayment_probability_per_period_true"]
    full = simulation_truth["full_repayment_probability_true"]
    completed = loan_outcomes["completed_all_payments"]
    default_counts = (
        loan_outcomes["default_period"]
        .value_counts(dropna=False, sort=False)
        .sort_index()
    )
    default_distribution = {
        ("completed" if pd.isna(period) else str(int(period))): int(count)
        for period, count in default_counts.items()
    }

    financial = {
        "annual_income": _numeric_summary(applicants["annual_income"]),
        "credit_score": _numeric_summary(applicants["credit_score"]),
        "liquid_assets": _numeric_summary(applicants["liquid_assets"]),
        "existing_monthly_debt": _numeric_summary(applicants["existing_monthly_debt"]),
        "property_value": _numeric_summary(applicants["property_value"]),
        "requested_loan_amount": _numeric_summary(applicants["requested_loan_amount"]),
        "requested_ltv": _numeric_summary(loan_options["requested_ltv"]),
        "first_period_dti": _numeric_summary(loan_options["first_period_dti"]),
    }
    quantiles = rho.quantile([0.05, 0.25, 0.75, 0.95])
    full_quantiles = full.quantile([0.05, 0.25, 0.75, 0.95])
    observed_default_period = loan_outcomes["default_period"].dropna().astype(float)
    risk = {
        "mean_rho_true": float(rho.mean()),
        "median_rho_true": float(rho.median()),
        "p05_rho_true": float(quantiles.loc[0.05]),
        "p25_rho_true": float(quantiles.loc[0.25]),
        "p75_rho_true": float(quantiles.loc[0.75]),
        "p95_rho_true": float(quantiles.loc[0.95]),
        "mean_full_repayment_probability_true": float(full.mean()),
        "median_full_repayment_probability_true": float(full.median()),
        "p05_full_repayment_probability_true": float(full_quantiles.loc[0.05]),
        "p25_full_repayment_probability_true": float(full_quantiles.loc[0.25]),
        "p75_full_repayment_probability_true": float(full_quantiles.loc[0.75]),
        "p95_full_repayment_probability_true": float(full_quantiles.loc[0.95]),
        "realized_full_repayment_rate": float(completed.mean()),
        "realized_default_rate": float((~completed).mean()),
        "default_period_summary_among_defaults": {
            "mean": float(observed_default_period.mean()),
            "median": float(observed_default_period.median()),
            "p25": float(observed_default_period.quantile(0.25)),
            "p75": float(observed_default_period.quantile(0.75)),
            "p95": float(observed_default_period.quantile(0.95)),
        },
        "default_period_counts": default_distribution,
    }
    economics = {
        "mean_expected_receipts_true": float(assessments["expected_receipts_true"].mean()),
        "mean_expected_profit_true": float(assessments["expected_profit_true"].mean()),
        "positive_expected_profit_share": float(
            (assessments["expected_profit_true"] > 0).mean()
        ),
        "nonpositive_expected_profit_share": float(
            (assessments["expected_profit_true"] <= 0).mean()
        ),
    }
    group_table = group_diagnostics(
        applicants, loan_options, simulation_truth, loan_outcomes
    )
    indexed_groups = group_table.set_index("group")
    group_differences = {
        column: float(indexed_groups.loc["B", column] - indexed_groups.loc["A", column])
        for column in group_table.columns
        if column not in {"group", "n"}
    }

    warnings: list[str] = []
    targets = config["true_risk"]["calibration_targets"]
    target_values = {
        "mean_conditional_repayment_probability": risk["mean_rho_true"],
        "mean_full_repayment_probability": risk[
            "mean_full_repayment_probability_true"
        ],
        "realized_default_rate": risk["realized_default_rate"],
        "positive_expected_profit_share": economics["positive_expected_profit_share"],
    }
    for name, value in target_values.items():
        bounds = targets[name]
        if not float(bounds["minimum"]) <= value <= float(bounds["maximum"]):
            warnings.append(
                f"{name}={value:.6f} is outside "
                f"[{bounds['minimum']}, {bounds['maximum']}]"
            )
    for variable, values in clipping.items():
        if values["total_clipped_share"] > 0.05:
            warnings.append(
                f"{variable} clipping share is {values['total_clipped_share']:.3%}"
            )

    report = {
        "config_version": config["version"],
        "config_fingerprint": config["metadata"]["config_fingerprint"],
        "n_applicants": int(len(applicants)),
        "random_seed": int(config["randomness"]["seed"]),
        "random_stream_spawn_keys": spawn_keys,
        "population": {
            "total_rows": int(len(applicants)),
            "group_a_share": float((applicants["group"] == "A").mean()),
            "group_b_share": float((applicants["group"] == "B").mean()),
            "cohort_counts": {
                str(name): int(value)
                for name, value in applicants["cohort"].value_counts(sort=False).items()
            },
        },
        "financial": financial,
        "risk": risk,
        "economics": economics,
        "group_invariance": {
            "structural_group_effect": False,
            "difference_definition": "Group B mean minus Group A mean",
            "observed_differences": group_differences,
        },
        "clipping": clipping,
        "warnings": warnings,
    }
    return report, group_table
