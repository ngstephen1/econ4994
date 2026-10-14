"""Equation validation, descriptive risk distributions, and evaluator audits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fair_lending.professor_model.risk import (
    TRUTH_COLUMNS, TRUE_RISK_SCHEMA_VERSION, compute_true_risk, validate_true_risk_config,
)


def _safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = left.corr(right, method=method)
    return float(value) if np.isfinite(value) else None


def validate_true_risk(
    truth: pd.DataFrame, applicants: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Check exact equations/IDs and summarize a frozen applicant population.

    Fixed-feature derivatives describe the constructed mechanism. Marginal
    correlations describe the sample and need not have the same sign.
    """
    validate_true_risk_config(config)
    report: dict[str, Any] = {
        "schema_version": TRUE_RISK_SCHEMA_VERSION, "information_scope": "EVALUATOR_ONLY",
        "valid": False, "checks": {}, "errors": [], "warnings": [],
    }

    def check(name: str, value: bool) -> None:
        report["checks"][name] = bool(value)
        if not value:
            report["errors"].append(name)

    expected_n = config["expected_input_rows"]
    check("exact_truth_schema", tuple(truth.columns) == TRUTH_COLUMNS)
    check("exact_row_count", len(truth) == len(applicants) == expected_n)
    check("no_missing_truth", not truth.isna().any().any())
    if report["errors"]:
        return report
    check("unique_truth_ids", truth.applicant_id.is_unique)
    check("unique_input_ids", applicants.applicant_id.is_unique)
    check("matching_ids", set(truth.applicant_id) == set(applicants.applicant_id))
    try:
        finite = np.isfinite(truth.iloc[:, 1:].to_numpy(dtype=float)).all()
    except (ValueError, TypeError):
        finite = False
    check("finite_truth", bool(finite))
    if report["errors"]:
        return report
    joined = applicants.merge(truth, on="applicant_id", validate="one_to_one")
    for name in TRUTH_COLUMNS[1:]:
        check(f"strict_bounds_{name}", bool(((joined[name] > 0) & (joined[name] < 1)).all()))
    calculated = compute_true_risk(applicants, config).set_index("applicant_id").loc[joined.applicant_id]
    for name in TRUTH_COLUMNS[1:]:
        check(f"equation_{name}", bool(np.allclose(joined[name], calculated[name], rtol=1e-12, atol=0)))
    check("product_identity", bool(np.allclose(joined.true_PD, joined.P_income * joined.P_other, rtol=1e-12, atol=0)))
    recomputed = compute_true_risk(applicants, config)
    check("deterministic_recalculation", calculated.reset_index().equals(recomputed.set_index("applicant_id").loc[joined.applicant_id].reset_index()))

    report["n_applicants"] = len(truth)
    report["unique_applicant_ids"] = truth.applicant_id.nunique()
    report["distributions"] = {}
    for name in TRUTH_COLUMNS[1:]:
        values = joined[name]
        report["distributions"][name] = {
            "mean": float(values.mean()), "median": float(values.median()),
            "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "minimum": float(values.min()), "maximum": float(values.max()),
            "quantiles": {f"p{int(q * 100):02d}": float(values.quantile(q)) for q in (.01, .05, .25, .50, .75, .95, .99)},
        }
    bins = [
        ("true_PD < 0.001", joined.true_PD < .001),
        ("0.001 <= true_PD < 0.01", (joined.true_PD >= .001) & (joined.true_PD < .01)),
        ("0.01 <= true_PD < 0.05", (joined.true_PD >= .01) & (joined.true_PD < .05)),
        ("true_PD >= 0.05", joined.true_PD >= .05),
    ]
    report["risk_bins"] = [{"bin": label, "count": int(mask.sum()), "fraction": float(mask.mean())} for label, mask in bins]
    report["correlations"] = [
        {"variable": name, "pearson": _safe_correlation(joined[name], joined.true_PD, "pearson"),
         "spearman": _safe_correlation(joined[name], joined.true_PD, "spearman")}
        for name in ("Y", "log_Y", "D", "epsilon", "C")
    ]
    report["group_comparison"] = []
    for group in (0, 1):
        values = joined.loc[joined.B == group, "true_PD"]
        report["group_comparison"].append({
            "B": group, "n": len(values),
            "mean_true_PD": float(values.mean()) if len(values) else None,
            "median_true_PD": float(values.median()) if len(values) else None,
            "p95_true_PD": float(values.quantile(.95)) if len(values) else None,
        })
    income_slope = config["equation"]["log_income_coefficient"] * joined.P_income * (1 - joined.P_income) * joined.P_other
    other_slope = joined.P_income * joined.P_other * (1 - joined.P_other)
    report["fixed_feature_direction_checks"] = {
        "higher_income_lowers_true_PD": bool((income_slope < 0).all()),
        "higher_D_raises_true_PD": bool((config["equation"]["dti_coefficient"] * other_slope > 0).all()),
        "higher_epsilon_raises_true_PD": bool((other_slope > 0).all()),
    }
    for name, passed in report["fixed_feature_direction_checks"].items():
        check(name, passed)
    mean_risk = report["distributions"]["true_PD"]["mean"]
    report["warnings"].append(f"Observed mean true_PD is {100 * mean_risk:.6f}%; review the supplied equation's scale before Step 4. No recalibration or probability floor was applied.")
    dti_correlation = next(item["pearson"] for item in report["correlations"] if item["variable"] == "D")
    if dti_correlation is not None and dti_correlation <= 0:
        report["warnings"].append("The marginal DTI correlation is not positive in this sample despite a positive fixed-feature derivative; other varying inputs can dominate the weak DTI effect.")
    report["sd_convention"] = "sample standard deviation, ddof=1"
    report["default_event_horizon"] = "unresolved; no period-survival interpretation assigned"
    report["valid"] = not report["errors"]
    return report
