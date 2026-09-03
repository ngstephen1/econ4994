"""Deterministic diagnostics for DTI and population calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fair_lending.simulation.approval import approval_probabilities


DTI_COMPONENT_COLUMNS = [
    "monthly_gross_income",
    "projected_principal_interest",
    "tax_insurance_estimate",
    "total_projected_housing_cost",
    "total_monthly_obligation",
    "pre_clip_dti",
    "loan_to_income_ratio",
]


def calculate_dti_components(
    applications: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Reconstruct internal DTI components without changing persisted fields."""
    payment = config["internal_housing_payment"]
    components = pd.DataFrame(index=applications.index)
    components["monthly_gross_income"] = applications["annual_income"] / 12.0
    components["projected_principal_interest"] = (
        applications["loan_amount"]
        * payment["monthly_principal_interest_factor"]
    )
    components["tax_insurance_estimate"] = (
        applications["property_value"]
        * payment["monthly_tax_insurance_factor_of_property_value"]
    )
    components["total_projected_housing_cost"] = (
        components["projected_principal_interest"]
        + components["tax_insurance_estimate"]
    )
    components["total_monthly_obligation"] = (
        applications["existing_monthly_debt"]
        + components["total_projected_housing_cost"]
    )
    components["pre_clip_dti"] = (
        components["total_monthly_obligation"]
        / components["monthly_gross_income"]
    )
    components["loan_to_income_ratio"] = (
        applications["loan_amount"] / applications["annual_income"]
    )
    return components.loc[:, DTI_COMPONENT_COLUMNS]


def distribution_statistics(values: pd.Series) -> dict[str, float]:
    """Return the common diagnostic statistics for a numeric series."""
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "minimum": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "maximum": float(values.max()),
    }


def component_distribution_table(
    applications: pd.DataFrame,
    components: pd.DataFrame,
    *,
    calibration_label: str,
) -> pd.DataFrame:
    """Describe borrower, loan, obligation, and unclipped-DTI components."""
    combined = pd.concat(
        [
            applications[
                [
                    "annual_income",
                    "existing_monthly_debt",
                    "property_value",
                    "loan_amount",
                    "loan_to_value_ratio",
                ]
            ],
            components,
        ],
        axis=1,
    )
    rows = []
    for field in combined:
        rows.append(
            {
                "calibration": calibration_label,
                "variable": field,
                **distribution_statistics(combined[field]),
            }
        )
    return pd.DataFrame(rows)


def summarize_diagnostic_run(
    applications: pd.DataFrame,
    config: dict[str, Any],
    *,
    candidate: str,
    intercept: float,
) -> dict[str, Any]:
    """Create one flat candidate-comparison record."""
    components = calculate_dti_components(applications, config)
    pre_clip_dti = components["pre_clip_dti"]
    loan_to_income = components["loan_to_income_ratio"]
    probability = approval_probabilities(applications, config, intercept)
    return {
        "candidate": candidate,
        "property_income_elasticity": float(
            config["loan_property_variables"]["property_value"]["income_elasticity"]
        ),
        "property_log_residual_standard_deviation": float(
            config["loan_property_variables"]["property_value"][
                "log_residual_standard_deviation"
            ]
        ),
        "existing_debt_income_elasticity": float(
            config["financial_variables"]["existing_monthly_debt"][
                "income_elasticity"
            ]
        ),
        "dti_mean": float(pre_clip_dti.mean()),
        "dti_standard_deviation": float(pre_clip_dti.std()),
        "dti_minimum": float(pre_clip_dti.min()),
        "dti_p25": float(pre_clip_dti.quantile(0.25)),
        "dti_median": float(pre_clip_dti.median()),
        "dti_p75": float(pre_clip_dti.quantile(0.75)),
        "dti_p90": float(pre_clip_dti.quantile(0.90)),
        "dti_p95": float(pre_clip_dti.quantile(0.95)),
        "dti_p99": float(pre_clip_dti.quantile(0.99)),
        "dti_maximum": float(pre_clip_dti.max()),
        "dti_above_0_50_share": float((pre_clip_dti > 0.50).mean()),
        "dti_above_0_60_share": float((pre_clip_dti > 0.60).mean()),
        "dti_at_or_above_0_65_share": float((pre_clip_dti >= 0.65).mean()),
        "median_income": float(applications["annual_income"].median()),
        "median_credit_score": float(applications["credit_score"].median()),
        "median_liquid_assets": float(applications["liquid_assets"].median()),
        "median_existing_monthly_debt": float(
            applications["existing_monthly_debt"].median()
        ),
        "median_property_value": float(applications["property_value"].median()),
        "median_loan_amount": float(applications["loan_amount"].median()),
        "median_ltv": float(applications["loan_to_value_ratio"].median()),
        "median_loan_to_income_ratio": float(loan_to_income.median()),
        "p95_loan_to_income_ratio": float(loan_to_income.quantile(0.95)),
        "mean_approval_probability_with_supplied_intercept": float(
            probability.mean()
        ),
    }


def high_dti_comparison_table(
    applications: pd.DataFrame,
    config: dict[str, Any],
    *,
    calibration_label: str,
) -> pd.DataFrame:
    """Compare four pre-clip DTI bands using continuous and category summaries."""
    components = calculate_dti_components(applications, config)
    diagnostic = applications.copy()
    diagnostic["pre_clip_dti"] = components["pre_clip_dti"]
    diagnostic["loan_to_income_ratio"] = components["loan_to_income_ratio"]
    bands = pd.cut(
        diagnostic["pre_clip_dti"],
        [-np.inf, 0.50, 0.60, 0.65, np.inf],
        right=False,
        labels=["A: <0.50", "B: 0.50-<0.60", "C: 0.60-<0.65", "D: >=0.65"],
    )
    continuous = [
        "annual_income",
        "credit_score",
        "employment_years",
        "liquid_assets",
        "existing_monthly_debt",
        "property_value",
        "loan_amount",
        "loan_to_value_ratio",
        "income_to_loan_ratio",
        "loan_to_income_ratio",
    ]
    category_fields = ("loan_purpose", "loan_type", "occupancy_type")
    rows: list[dict[str, Any]] = []
    for band in bands.cat.categories:
        subset = diagnostic.loc[bands == band]
        row: dict[str, Any] = {
            "calibration": calibration_label,
            "dti_band": str(band),
            "n_observations": int(len(subset)),
            "sample_share": float(len(subset) / len(diagnostic)),
        }
        for field in continuous:
            row[f"{field}_median"] = float(subset[field].median())
            row[f"{field}_p90"] = float(subset[field].quantile(0.90))
        for field in category_fields:
            realized = subset[field].astype(object).value_counts(normalize=True)
            for category in config["loan_property_variables"][field]["shares"]:
                safe_category = (
                    category.lower().replace(" ", "_").replace("/", "_")
                )
                row[f"{field}_share_{safe_category}"] = float(
                    realized.get(category, 0.0)
                )
        rows.append(row)
    return pd.DataFrame(rows)


def dti_correlation_table(
    applications: pd.DataFrame, config: dict[str, Any], *, calibration_label: str
) -> pd.DataFrame:
    """Return the requested DTI-driver correlation matrix in long form."""
    components = calculate_dti_components(applications, config)
    values = applications[
        [
            "annual_income",
            "property_value",
            "loan_amount",
            "loan_to_value_ratio",
            "existing_monthly_debt",
        ]
    ].copy()
    values["pre_clip_dti"] = components["pre_clip_dti"]
    values["loan_to_income_ratio"] = components["loan_to_income_ratio"]
    matrix = values.corr()
    return (
        matrix.rename_axis("variable_1")
        .reset_index()
        .melt(id_vars="variable_1", var_name="variable_2", value_name="correlation")
        .assign(calibration=calibration_label)
        .loc[:, ["calibration", "variable_1", "variable_2", "correlation"]]
    )
