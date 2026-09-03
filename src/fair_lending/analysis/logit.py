"""Statsmodels logistic specifications matched to the synthetic DGP."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from fair_lending.simulation.config import approval_transform_parameters


MODEL_ORDER = ("model_0", "model_1", "model_2", "model_3")
MODEL_LABELS = {
    "model_0": "Model 0: race only",
    "model_1": "Model 1: core financial controls",
    "model_2": "Model 2: financial + loan controls",
    "model_3": "Model 3: expanded demographic sensitivity",
}
TRANSFORMED_TERMS = {
    "annual_income": "log_income",
    "credit_score": "credit_score_50",
    "employment_years": "employment_years_10",
    "liquid_assets": "log_liquid_assets",
    "debt_to_income_ratio": "dti_10pp",
    "loan_to_value_ratio": "ltv_10pp",
}
CORE_CONTROL_COLUMNS = list(TRANSFORMED_TERMS.values())
LOAN_REFERENCES = {
    "loan_purpose": "Home purchase",
    "loan_type": "Conventional",
    "occupancy_type": "Principal residence",
}
DEMOGRAPHIC_REFERENCES = {
    "age_group": "35-44",
    "sex": "Male",
    "ethnicity": "Not Hispanic or Latino",
}
FORBIDDEN_PRIMARY_FIELDS = {
    "neighborhood_minority_share",
    "approval_probability_true",
    "denial_reason",
    "property_value",
    "loan_amount",
    "income_to_loan_ratio",
}


def _transform(values: pd.Series, parameters: dict[str, float | str]) -> np.ndarray:
    array = values.to_numpy(dtype=float)
    scale = float(parameters["scale"])
    if parameters["kind"] == "linear_centered":
        return (array - float(parameters["center"])) / scale
    if parameters["kind"] == "log_centered":
        return (np.log(array) - np.log(float(parameters["center"]))) / scale
    if parameters["kind"] == "log_offset_centered":
        return (
            np.log(array + float(parameters["offset"]))
            - np.log(float(parameters["center"]))
        ) / scale
    raise ValueError(f"Unsupported transform kind: {parameters['kind']}")


def prepare_regression_sample(
    data: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Restrict to White/Black and construct exact DGP-scaled covariates."""
    race = data["race"].astype(object)
    sample = data.loc[race.isin(["White", "Black"])].copy().reset_index(drop=True)
    sample["black"] = (sample["race"].astype(object) == "Black").astype(float)
    for source, transformed in TRANSFORMED_TERMS.items():
        sample[transformed] = _transform(
            sample[source], approval_transform_parameters(config, source)
        )
    return sample


def _safe_category_name(value: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")


def _add_reference_dummies(
    matrix: pd.DataFrame,
    sample: pd.DataFrame,
    config: dict[str, Any],
    references: dict[str, str],
) -> None:
    for field, reference in references.items():
        if field in config["loan_property_variables"]:
            categories = list(config["loan_property_variables"][field]["shares"])
        else:
            categories = list(config["population"]["demographics"][field]["shares"])
        if reference not in categories:
            raise ValueError(f"Reference category {reference!r} missing for {field}")
        values = sample[field].astype(object)
        for category in categories:
            if category != reference:
                name = f"{field}__{_safe_category_name(category)}"
                matrix[name] = (values == category).astype(float)


def build_design_matrix(
    sample: pd.DataFrame,
    config: dict[str, Any],
    model_name: str,
) -> pd.DataFrame:
    """Build one of the four predeclared model matrices with explicit references."""
    if model_name not in MODEL_ORDER:
        raise ValueError(f"Unknown model: {model_name}")
    matrix = pd.DataFrame({"const": 1.0, "black": sample["black"]}, index=sample.index)
    if model_name in {"model_1", "model_2", "model_3"}:
        for field in CORE_CONTROL_COLUMNS:
            matrix[field] = sample[field].to_numpy(dtype=float)
    if model_name in {"model_2", "model_3"}:
        _add_reference_dummies(matrix, sample, config, LOAN_REFERENCES)
    if model_name == "model_3":
        _add_reference_dummies(matrix, sample, config, DEMOGRAPHIC_REFERENCES)

    leaked = FORBIDDEN_PRIMARY_FIELDS.intersection(matrix.columns)
    if leaked:
        raise ValueError(f"Forbidden variables leaked into model matrix: {sorted(leaked)}")
    return matrix.astype(float)


def design_matrix_diagnostics(matrix: pd.DataFrame) -> dict[str, Any]:
    """Check rank, duplicates, constants, conditioning, and finite values."""
    duplicate_pairs: list[list[str]] = []
    columns = list(matrix.columns)
    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            if np.array_equal(matrix[left].to_numpy(), matrix[right].to_numpy()):
                duplicate_pairs.append([left, right])
    zero_variance = [
        column
        for column in columns
        if column != "const" and float(matrix[column].var()) == 0.0
    ]
    array = matrix.to_numpy(dtype=float)
    rank = int(np.linalg.matrix_rank(array))
    return {
        "n_columns": int(matrix.shape[1]),
        "rank": rank,
        "full_rank": rank == matrix.shape[1],
        "condition_number": float(np.linalg.cond(array)),
        "duplicate_column_pairs": duplicate_pairs,
        "zero_variance_columns": zero_variance,
        "all_finite": bool(np.isfinite(array).all()),
    }


def fit_logit_model(
    data: pd.DataFrame,
    config: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Fit one statsmodels Logit model and return inference plus internals."""
    sample = prepare_regression_sample(data, config)
    matrix = build_design_matrix(sample, config, model_name)
    diagnostics = design_matrix_diagnostics(matrix)
    if not diagnostics["full_rank"] or not diagnostics["all_finite"]:
        raise ValueError(f"Invalid design matrix for {model_name}: {diagnostics}")

    result = sm.Logit(sample["approved"].astype(float), matrix).fit(
        method="newton", maxiter=100, disp=False
    )
    confidence = result.conf_int(alpha=0.05).loc["black"]
    coefficient = float(result.params["black"])
    standard_error = float(result.bse["black"])
    summary = {
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "n": int(result.nobs),
        "black_coefficient": coefficient,
        "black_se": standard_error,
        "black_z": float(result.tvalues["black"]),
        "black_p_value": float(result.pvalues["black"]),
        "black_ci_low": float(confidence.iloc[0]),
        "black_ci_high": float(confidence.iloc[1]),
        "black_odds_ratio": float(np.exp(coefficient)),
        "black_odds_ratio_ci_low": float(np.exp(confidence.iloc[0])),
        "black_odds_ratio_ci_high": float(np.exp(confidence.iloc[1])),
        "converged": bool(result.mle_retvals.get("converged", False)),
        "iterations": int(result.mle_retvals.get("iterations", 0)),
        "log_likelihood": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "pseudo_r_squared_mcfadden": float(result.prsquared),
        "matrix_columns": list(matrix.columns),
        "matrix_rank": diagnostics["rank"],
        "condition_number": diagnostics["condition_number"],
        "duplicate_column_pairs": diagnostics["duplicate_column_pairs"],
        "zero_variance_columns": diagnostics["zero_variance_columns"],
        "maximum_standard_error": float(result.bse.max()),
    }
    return {
        "result": result,
        "sample": sample,
        "design_matrix": matrix,
        "diagnostics": diagnostics,
        "summary": summary,
    }


def fit_logit_sequence(
    data: pd.DataFrame, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fit all four nested, predeclared specifications."""
    return [fit_logit_model(data, config, model_name) for model_name in MODEL_ORDER]
