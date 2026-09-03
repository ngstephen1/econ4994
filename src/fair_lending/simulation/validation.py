"""Structured generator validation and concise validation-artifact output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import logit

from fair_lending.simulation.approval import counterfactual_direct_probabilities
from fair_lending.simulation.config import EXPECTED_SCENARIO_SWITCHES, PROJECT_ROOT
from fair_lending.simulation.generator import OUTPUT_COLUMNS


AGE_GROUP_MAX_EMPLOYMENT = {
    "18-24": 6.0,
    "25-34": 16.0,
    "35-44": 26.0,
    "45-54": 36.0,
    "55-64": 45.0,
    "65+": 45.0,
}


def _record_check(
    checks: dict[str, bool],
    errors: list[str],
    name: str,
    condition: bool,
    message: str,
) -> None:
    checks[name] = bool(condition)
    if not condition:
        errors.append(message)


def validate_generated_data(
    data: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Validate schema, invariants, distributions, mechanisms, and boundaries."""
    config = metadata["resolved_configuration"]
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    _record_check(
        checks,
        errors,
        "exact_schema",
        list(data.columns) == OUTPUT_COLUMNS and len(data.columns) == 24,
        "Dataset does not have the exact ordered 24-field schema.",
    )
    _record_check(
        checks,
        errors,
        "exact_row_count",
        len(data) == config["simulation"]["n_samples"],
        "Row count does not match the resolved configuration.",
    )
    _record_check(
        checks,
        errors,
        "unique_application_ids",
        data["application_id"].is_unique,
        "Application IDs are not unique.",
    )
    unexpected_missing = data.drop(columns=["denial_reason"]).isna().sum()
    _record_check(
        checks,
        errors,
        "no_unexpected_missingness",
        int(unexpected_missing.sum()) == 0,
        "Non-denial-reason fields contain missing values.",
    )
    _record_check(
        checks,
        errors,
        "denial_reason_all_null",
        data["denial_reason"].isna().all(),
        "denial_reason must be entirely null in version 1.",
    )
    numeric_fields = [
        field
        for field in OUTPUT_COLUMNS
        if field
        not in {
            "application_id",
            "race",
            "ethnicity",
            "sex",
            "age_group",
            "loan_purpose",
            "loan_type",
            "occupancy_type",
            "denial_reason",
        }
    ]
    _record_check(
        checks,
        errors,
        "numeric_field_types",
        all(pd.api.types.is_numeric_dtype(data[field]) for field in numeric_fields),
        "One or more numerical fields have a non-numeric dtype.",
    )
    _record_check(
        checks,
        errors,
        "application_id_string_type",
        pd.api.types.is_string_dtype(data["application_id"]),
        "application_id must contain strings.",
    )

    financial = config["financial_variables"]
    loan = config["loan_property_variables"]
    context = config["context_variables"]
    bound_specs = {
        "annual_income": financial["annual_income"]["bounds"],
        "credit_score": financial["credit_score"]["bounds"],
        "liquid_assets": financial["liquid_assets"]["bounds"],
        "existing_monthly_debt": financial["existing_monthly_debt"]["bounds"],
        "debt_to_income_ratio": financial["debt_to_income_ratio"]["bounds"],
        "property_value": loan["property_value"]["bounds"],
        "loan_to_value_ratio": loan["loan_to_value_ratio"]["bounds"],
        "neighborhood_income_index": context["neighborhood_income_index"]["bounds"],
        "neighborhood_minority_share": context["neighborhood_minority_share"]["bounds"],
        "local_unemployment_rate": context["local_unemployment_rate"]["bounds"],
    }
    for field, bounds in bound_specs.items():
        valid = bool(data[field].between(bounds["minimum"], bounds["maximum"]).all())
        _record_check(
            checks, errors, f"bounds_{field}", valid, f"{field} violates configured bounds."
        )
    for field in ("annual_income", "loan_amount", "property_value"):
        _record_check(
            checks,
            errors,
            f"positive_{field}",
            bool((data[field] > 0.0).all()),
            f"{field} must be strictly positive.",
        )

    _record_check(
        checks,
        errors,
        "loan_amount_identity",
        bool(
            np.allclose(
                data["loan_amount"],
                data["property_value"] * data["loan_to_value_ratio"],
                rtol=1e-12,
                atol=1e-8,
            )
        ),
        "loan_amount != property_value * loan_to_value_ratio.",
    )
    _record_check(
        checks,
        errors,
        "income_to_loan_identity",
        bool(
            np.allclose(
                data["income_to_loan_ratio"],
                data["annual_income"] / data["loan_amount"],
                rtol=1e-12,
                atol=1e-12,
            )
        ),
        "income_to_loan_ratio identity is corrupted.",
    )

    category_specs = {
        name: set(spec["shares"])
        for name, spec in config["population"]["demographics"].items()
    }
    category_specs.update(
        {
            name: set(loan[name]["shares"])
            for name in ("loan_purpose", "loan_type", "occupancy_type")
        }
    )
    for field, allowed in category_specs.items():
        observed = set(data[field].astype(object).unique())
        _record_check(
            checks,
            errors,
            f"categories_{field}",
            observed.issubset(allowed),
            f"{field} contains unknown categories: {sorted(observed - allowed)}",
        )

    employment_valid = np.ones(len(data), dtype=bool)
    for group, maximum in AGE_GROUP_MAX_EMPLOYMENT.items():
        mask = data["age_group"].astype(object).to_numpy() == group
        employment_valid[mask] = data.loc[mask, "employment_years"].between(0.0, maximum)
    _record_check(
        checks,
        errors,
        "age_employment_consistency",
        bool(employment_valid.all()),
        "Employment history exceeds an age-group feasible maximum.",
    )

    _record_check(
        checks,
        errors,
        "approval_probability_bounds",
        bool(data["approval_probability_true"].between(0.0, 1.0).all()),
        "True approval probabilities are outside [0, 1].",
    )
    _record_check(
        checks,
        errors,
        "binary_approved",
        set(data["approved"].unique()).issubset({0, 1}),
        "approved contains values outside {0, 1}.",
    )

    demographics: dict[str, Any] = {}
    for field, spec in config["population"]["demographics"].items():
        expected = spec["shares"]
        realized = data[field].astype(object).value_counts(normalize=True)
        field_results = {}
        for category, probability in expected.items():
            observed = float(realized.get(category, 0.0))
            standard_error = np.sqrt(probability * (1.0 - probability) / len(data))
            tolerance = 4.0 * standard_error + 0.002
            field_results[category] = {
                "expected": float(probability),
                "realized": observed,
                "difference": observed - float(probability),
                "diagnostic_tolerance": float(tolerance),
            }
            if abs(observed - probability) > tolerance:
                warnings.append(
                    f"{field} share for {category} is outside its diagnostic tolerance."
                )
        demographics[field] = field_results

    summary_fields = [
        "annual_income",
        "credit_score",
        "liquid_assets",
        "property_value",
        "loan_amount",
        "debt_to_income_ratio",
        "loan_to_value_ratio",
    ]
    distribution_summary = {
        field: {
            "mean": float(data[field].mean()),
            "median": float(data[field].median()),
        }
        for field in summary_fields
    }
    correlations = {
        "income_assets": float(data["annual_income"].corr(data["liquid_assets"])),
        "income_property_value": float(
            data["annual_income"].corr(data["property_value"])
        ),
    }
    for name, value in correlations.items():
        if not np.isfinite(value) or value <= 0.0:
            warnings.append(f"Expected positive dependence diagnostic failed: {name}={value}.")

    probabilities = data["approval_probability_true"]
    approval = {
        "mean_probability": float(probabilities.mean()),
        "realized_rate": float(data["approved"].mean()),
        "probability_quantiles": {
            str(quantile): float(probabilities.quantile(quantile))
            for quantile in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "fraction_below_0_01": float((probabilities < 0.01).mean()),
        "fraction_above_0_99": float((probabilities > 0.99).mean()),
        "mean_probability_by_race": {
            str(key): float(value)
            for key, value in data.groupby("race", observed=False)[
                "approval_probability_true"
            ].mean().items()
        },
        "realized_rate_by_race": {
            str(key): float(value)
            for key, value in data.groupby("race", observed=False)["approved"].mean().items()
        },
    }
    if config["simulation"]["scenario"] == "fair_baseline":
        target = config["approval_model"]["intercept"]["target_mean_probability"]
        tolerance_spec = config["validation"]["realized_approval_rate_tolerance"]
        tolerance = (
            tolerance_spec["standard_error_multiplier"]
            * np.sqrt(target * (1.0 - target) / len(data))
            + tolerance_spec["absolute_margin"]
        )
        if abs(approval["mean_probability"] - target) > tolerance:
            warnings.append(
                "Fair-baseline mean true approval probability is outside the run-level tolerance."
            )

    clipping_rates = metadata["population_diagnostics"]["clipping_rates"]
    maximum_clipped = config["validation"]["maximum_clipped_share_per_variable"]
    for field, rates in clipping_rates.items():
        if (
            field != "debt_to_income_ratio"
            and rates["total_clipped_share"] > maximum_clipped
        ):
            warnings.append(
                f"{field} clipped share {rates['total_clipped_share']:.4%} exceeds "
                f"the configured {maximum_clipped:.2%} threshold."
            )
    dti_boundary = float(
        clipping_rates["debt_to_income_ratio"]["upper_boundary_share"]
    )
    dti_threshold = financial["debt_to_income_ratio"][
        "maximum_allowed_boundary_share"
    ]
    if dti_boundary > dti_threshold:
        warnings.append(
            f"DTI upper-bound mass {dti_boundary:.4%} exceeds the configured "
            f"{dti_threshold:.2%} threshold."
        )

    scenario = config["simulation"]["scenario"]
    expected_upstream, expected_direct = EXPECTED_SCENARIO_SWITCHES[scenario]
    effects = config["scenario_effects"]
    upstream_active = bool(effects["upstream"]["enabled"])
    direct_active = bool(effects["direct"]["enabled"])
    upstream_nonzero = any(
        any(value != 0.0 for value in treatment["additive_shift_by_race"].values())
        for treatment in effects["upstream"]["treatments"].values()
    )
    direct_nonzero = any(
        value != 0.0 for value in effects["direct"]["log_odds_by_race"].values()
    )
    scenario_invariants = {
        "expected_upstream_enabled": expected_upstream,
        "actual_upstream_enabled": upstream_active,
        "upstream_nonzero": upstream_nonzero,
        "expected_direct_enabled": expected_direct,
        "actual_direct_enabled": direct_active,
        "direct_nonzero": direct_nonzero,
        "context_race_conditioned": bool(
            context["neighborhood_minority_share"]["baseline_race_dependency"]
        ),
        "race_in_baseline_approval_terms": "race"
        in config["approval_model"]["continuous_terms"]
        or "race" in config["approval_model"]["categorical_terms"],
    }
    mechanism_valid = (
        upstream_active is expected_upstream
        and direct_active is expected_direct
        and upstream_nonzero is expected_upstream
        and direct_nonzero is expected_direct
        and not scenario_invariants["context_race_conditioned"]
        and not scenario_invariants["race_in_baseline_approval_terms"]
    )
    _record_check(
        checks,
        errors,
        "scenario_mechanism",
        mechanism_valid,
        "Resolved scenario mechanisms do not match the four-world design.",
    )

    counterfactual: dict[str, Any] | None = None
    if direct_active:
        values = counterfactual_direct_probabilities(
            data, config, float(metadata["intercept"])
        )
        reference_probability = values["reference_probability"]
        comparison_probability = values["comparison_probability"]
        counterfactual = {
            "configured_log_odds_difference": float(
                values["configured_log_odds_difference"]
            ),
            "verified_score_difference": float(values["score_difference"]),
            "mean_probability_difference": float(
                np.mean(comparison_probability - reference_probability)
            ),
            "all_comparison_probabilities_lower": bool(
                np.all(comparison_probability < reference_probability)
            ),
            "maximum_absolute_log_odds_error": float(
                np.max(
                    np.abs(
                        (logit(comparison_probability) - logit(reference_probability))
                        - float(values["configured_log_odds_difference"])
                    )
                )
            ),
        }
        _record_check(
            checks,
            errors,
            "direct_counterfactual_direction",
            counterfactual["all_comparison_probabilities_lower"],
            "Direct Black penalty did not reduce probability at fixed features.",
        )
        _record_check(
            checks,
            errors,
            "direct_counterfactual_exact_log_odds",
            counterfactual["maximum_absolute_log_odds_error"] < 1e-10,
            "Direct counterfactual does not match the configured log-odds penalty.",
        )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "scenario": scenario,
        "effect_level": config["simulation"]["effect_level"],
        "n_rows": int(len(data)),
        "seed": int(config["simulation"]["random_seed"]),
        "demographic_proportions": demographics,
        "distribution_summary": distribution_summary,
        "dependence_diagnostics": correlations,
        "approval": approval,
        "boundary_and_clipping_rates": clipping_rates,
        "scenario_invariants": scenario_invariants,
        "counterfactual_direct_effect": counterfactual,
    }


def validation_summary_row(report: dict[str, Any]) -> dict[str, Any]:
    distributions = report["distribution_summary"]
    return {
        "scenario": report["scenario"],
        "effect_level": report["effect_level"],
        "n_rows": report["n_rows"],
        "seed": report["seed"],
        "valid": report["valid"],
        "warning_count": len(report["warnings"]),
        "mean_approval_probability": report["approval"]["mean_probability"],
        "realized_approval_rate": report["approval"]["realized_rate"],
        "median_income": distributions["annual_income"]["median"],
        "median_credit_score": distributions["credit_score"]["median"],
        "median_dti": distributions["debt_to_income_ratio"]["median"],
        "median_ltv": distributions["loan_to_value_ratio"]["median"],
        "dti_upper_boundary_share": report["boundary_and_clipping_rates"][
            "debt_to_income_ratio"
        ]["upper_boundary_share"],
    }


def save_validation_outputs(
    report: dict[str, Any], dataset_path: Path | str
) -> tuple[Path, Path]:
    """Save structured JSON metrics and a concise one-row CSV table."""
    stem = Path(dataset_path).stem
    metrics_path = PROJECT_ROOT / "results" / "metrics" / f"validation_{stem}.json"
    table_path = PROJECT_ROOT / "results" / "tables" / f"summary_{stem}.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame([validation_summary_row(report)]).to_csv(table_path, index=False)
    return metrics_path, table_path


def save_validation_comparison(
    reports: list[dict[str, Any]],
    output_path: Path | str | None = None,
) -> Path:
    """Save a compact comparison table for a declared set of validation runs."""
    path = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "results" / "tables" / "generator_validation_summary.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([validation_summary_row(report) for report in reports]).to_csv(
        path, index=False
    )
    return path
