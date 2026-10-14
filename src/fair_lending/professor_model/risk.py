"""Deterministic evaluator-only default risk under the professor's equations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit

from fair_lending.professor_model.config import PROJECT_ROOT

TRUE_RISK_SCHEMA_VERSION = "professor-true-risk-v1"
TRUE_RISK_CONFIG_PATH = PROJECT_ROOT / "configs/professor_model/true_risk.yaml"
TRUE_RISK_INPUTS = ("log_Y", "D", "epsilon")
TRUTH_COLUMNS = ("applicant_id", "P_income", "P_other", "true_PD")
TRUTH_PREVIEW_COLUMNS = (
    "applicant_id", "Y", "log_Y", "D", "B", "epsilon", "C", "P_income", "P_other", "true_PD"
)
FROZEN_EQUATION = {
    "income_intercept": 0.52, "log_income_coefficient": -0.65,
    "other_intercept": 4.0, "dti_coefficient": 0.06,
}


def validate_true_risk_config(config: dict[str, Any]) -> None:
    """Reject equation retuning and writes outside the separate model family."""
    try:
        if config["schema_version"] != TRUE_RISK_SCHEMA_VERSION:
            raise ValueError("unsupported true-risk schema")
        if config["equation"] != FROZEN_EQUATION:
            raise ValueError("true-risk coefficients must match the supplied frozen equation")
        for key in ("expected_input_rows", "preview_rows"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if config["preview_rows"] > 200:
            raise ValueError("preview_rows must not exceed 200")
        for value, prefix in (
            (config["input_directory"], "data/synthetic/professor_model"),
            (config["outputs"]["validation_report"], "results/metrics/professor_model"),
            (config["outputs"]["summary"], "results/tables/professor_model"),
            (config["outputs"]["group_comparison"], "results/tables/professor_model"),
            (config["outputs"]["correlations"], "results/tables/professor_model"),
            (config["outputs"]["figures_directory"], "results/figures/professor_model"),
        ):
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or not path.is_relative_to(prefix):
                raise ValueError(f"path must remain inside {prefix}")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing or malformed true-risk configuration: {exc}") from exc


def load_true_risk_config(path: Path | str = TRUE_RISK_CONFIG_PATH) -> dict[str, Any]:
    """Load Step 3 settings without changing Step 2 configuration or metadata."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("true-risk config must be a YAML mapping")
    validate_true_risk_config(config)
    return config


def compute_true_risk(applicants: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Evaluate exactly two stable sigmoid factors, without sampling or clipping.

    The allowlist makes B, C, Y in levels, and omega irrelevant to calculation.
    Natural log income is consumed directly from the saved private view.
    """
    validate_true_risk_config(config)
    required = ("applicant_id", *TRUE_RISK_INPUTS)
    if not set(required).issubset(applicants.columns):
        raise ValueError(f"required risk inputs: {required}")
    source = applicants.loc[:, required]
    if source.empty or source.isna().any().any() or not source.applicant_id.is_unique:
        raise ValueError("risk inputs require nonempty data, unique IDs, and no missing values")
    values = source.loc[:, TRUE_RISK_INPUTS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("risk inputs must be finite")
    log_income, dti, epsilon = values.T
    coefficients = config["equation"]
    income = expit(coefficients["income_intercept"] + coefficients["log_income_coefficient"] * log_income)
    other = expit(coefficients["other_intercept"] + coefficients["dti_coefficient"] * dti + epsilon)
    return pd.DataFrame({
        "applicant_id": source.applicant_id.to_numpy(copy=True),
        "P_income": income, "P_other": other, "true_PD": income * other,
    }).loc[:, TRUTH_COLUMNS]
