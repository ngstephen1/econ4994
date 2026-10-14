"""Identity checks and descriptive diagnostics for the Step 2 population."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fair_lending.professor_model.config import BANK_COLUMNS, PRIVATE_COLUMNS, SCHEMA_VERSION, validate_config

QUANTILES = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)


def _number(value: Any) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _summary(values: pd.Series) -> dict[str, Any]:
    return {
        "mean": float(values.mean()), "median": float(values.median()),
        "sd": _number(values.std(ddof=1)), "minimum": float(values.min()), "maximum": float(values.max()),
        "quantiles": {f"p{int(q * 100):02d}": float(values.quantile(q)) for q in QUANTILES},
    }


def validate_population(
    bank: pd.DataFrame, private: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    """Validate structural identities; report permitted Normal tails as warnings.

    Finite-sample distribution/correlation diagnostics do not assert that sample
    means equal configured means or that audit groups are exactly balanced.
    All SD summaries use the sample convention (ddof=1).
    """
    validate_config(config)
    report: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "valid": False, "errors": [], "warnings": [], "checks": {}}

    def check(name: str, condition: bool) -> None:
        report["checks"][name] = bool(condition)
        if not condition:
            report["errors"].append(name)

    expected = config["simulation"]["n_samples"]
    check("exact_bank_schema", tuple(bank.columns) == BANK_COLUMNS)
    check("exact_private_schema", tuple(private.columns) == PRIVATE_COLUMNS)
    # A strict allowlist rejects any hidden/future field, not just known names.
    check("bank_information_allowlist", set(bank.columns) == set(BANK_COLUMNS))
    check("exact_bank_count", len(bank) == expected)
    check("exact_private_count", len(private) == expected)
    check("no_missing_values", not bank.isna().any().any() and not private.isna().any().any())
    if report["errors"]:
        return report
    check("unique_bank_ids", bank["applicant_id"].is_unique)
    check("unique_private_ids", private["applicant_id"].is_unique)
    check("matching_applicant_ids", set(bank["applicant_id"]) == set(private["applicant_id"]))
    for name, frame in (("bank", bank), ("private", private)):
        numeric = frame.drop(columns="applicant_id")
        check(f"{name}_numeric_columns", all(is_numeric_dtype(dtype) for dtype in numeric.dtypes))
        if report["checks"][f"{name}_numeric_columns"]:
            check(f"{name}_all_finite", bool(np.isfinite(numeric.to_numpy()).all()))
    if report["errors"]:
        return report
    combined = bank.merge(private, on="applicant_id", validate="one_to_one")
    check("positive_income", bool((combined["Y"] > 0).all()))
    check("binary_group", bool(combined["B"].isin([0, 1]).all()))
    check("income_exponential_identity", bool(np.allclose(combined["Y"], np.exp(combined["log_Y"]), rtol=1e-12, atol=0)))
    coefficient = config["credit_signal"]["epsilon_coefficient"]
    check("credit_signal_identity", bool(np.allclose(combined["C"], coefficient * combined["epsilon"] + combined["omega"], rtol=1e-12, atol=1e-12)))

    report["population"] = {"n_applicants": len(bank), "unique_bank_ids": bank.applicant_id.nunique(), "unique_private_ids": private.applicant_id.nunique()}
    for variable in ("Y", "log_Y", "D", "epsilon", "omega", "C"):
        report[variable] = _summary(combined[variable])
    below = int((combined.D < 0).sum())
    above = int((combined.D > 1).sum())
    report["D"].update({"count_below_zero": below, "fraction_below_zero": below / expected, "percent_below_zero": 100 * below / expected, "count_above_one": above, "fraction_above_one": above / expected, "percent_above_one": 100 * above / expected, "clipping_applied": False})
    if below:
        report["warnings"].append(f"D < 0 for {below} applicants ({100 * below / expected:.4f}%); retained under the untruncated Normal specification.")
    if above:
        report["warnings"].append(f"D > 1 for {above} applicants ({100 * above / expected:.4f}%); retained under the untruncated Normal specification.")
    report["signal_correlations"] = {"C_epsilon": _number(combined.C.corr(combined.epsilon)), "C_omega": _number(combined.C.corr(combined.omega))}
    report["group_comparison"] = []
    for group in (0, 1):
        subset = combined.loc[combined.B == group]
        report["group_comparison"].append({"B": group, "n": len(subset), "share": len(subset) / expected, **{f"mean_{v}": _number(subset[v].mean()) for v in ("Y", "D", "epsilon", "C")}})
    a, b = report["group_comparison"]
    report["group_1_minus_0_mean_differences"] = {
        variable: b[f"mean_{variable}"] - a[f"mean_{variable}"] if a[f"mean_{variable}"] is not None and b[f"mean_{variable}"] is not None else None
        for variable in ("Y", "D", "epsilon", "C")
    }
    primitives = ("log_Y", "D", "B", "epsilon", "omega")
    correlations = combined.loc[:, primitives].corr()
    off_diagonal = correlations.to_numpy()[~np.eye(len(primitives), dtype=bool)]
    finite_correlations = off_diagonal[np.isfinite(off_diagonal)]
    report["independence_diagnostics"] = {
        "primitive_correlations": {row: {col: _number(correlations.loc[row, col]) for col in primitives} for row in primitives},
        "max_absolute_primitive_correlation": float(np.abs(finite_correlations).max()) if len(finite_correlations) else None,
        "interpretation": "Finite-sample diagnostic; independent RNG streams and group-invariance tests establish the generation mechanism.",
    }
    eps_sd = config["population"]["epsilon"]["sd"]
    omega_sd = config["population"]["omega"]["sd"]
    signal_sd = math.sqrt(coefficient**2 * eps_sd**2 + omega_sd**2)
    report["configured_signal_moments"] = {
        "mean_C": coefficient * config["population"]["epsilon"]["mean"] + config["population"]["omega"]["mean"],
        "sd_C": signal_sd, "correlation_C_epsilon": coefficient * eps_sd / signal_sd, "correlation_C_omega": omega_sd / signal_sd,
    }
    report["sd_convention"] = "sample standard deviation, ddof=1"
    report["valid"] = not report["errors"]
    return report
