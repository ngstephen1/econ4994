"""Researcher-frozen nonlinear repayment-risk sensitivity world."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

from fair_lending.economic_lending.config import PROJECT_ROOT, config_fingerprint
from fair_lending.economic_lending.repayment import (
    TRUE_RISK_PREDICTORS,
    sigmoid,
    survival_probability,
    transformed_true_risk_predictors,
)
from fair_lending.economic_lending.schema import SIMULATION_TRUTH_COLUMNS


NONLINEAR_CONFIG_PATH = PROJECT_ROOT / "configs" / "economic_lending" / "nonlinear_risk.yaml"
NONLINEAR_WORLD_ID = "nonlinear_v1"
NONLINEAR_ALLOWED_OBSERVABLES = frozenset(
    {
        "annual_income",
        "credit_score",
        "employment_years",
        "liquid_assets",
        "first_period_dti",
        "requested_ltv",
    }
)


def load_nonlinear_config(path: Path | str = NONLINEAR_CONFIG_PATH) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("risk_world") != NONLINEAR_WORLD_ID:
        raise ValueError("nonlinear config must declare risk_world=nonlinear_v1")
    config = copy.deepcopy(loaded)
    config["metadata"] = {
        "config_path": str(path),
        "config_fingerprint": config_fingerprint(loaded),
    }
    return config


def nonlinear_score_components(transformed: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return the three declared nonlinear score contributions."""

    terms = config["nonlinear_terms"]
    dti = transformed["first_period_dti_10pp"].to_numpy(dtype=float)
    credit = transformed["credit_score_50"].to_numpy(dtype=float)
    ltv = transformed["requested_ltv_10pp"].to_numpy(dtype=float)
    assets = transformed["log_liquid_assets"].to_numpy(dtype=float)
    threshold = float(terms["high_dti_convex_penalty"]["threshold_scaled_dti"])
    return pd.DataFrame(
        {
            "high_dti_convex_penalty": float(terms["high_dti_convex_penalty"]["coefficient"])
            * np.square(np.maximum(dti - threshold, 0.0)),
            "weak_credit_high_ltv_interaction": float(
                terms["weak_credit_high_ltv_interaction"]["coefficient"]
            )
            * np.maximum(-credit, 0.0)
            * np.maximum(ltv, 0.0),
            "bounded_asset_buffer": float(terms["bounded_asset_buffer"]["coefficient"])
            * np.tanh(assets),
        },
        index=transformed.index,
    )


def nonlinear_score_without_intercept(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    baseline_true_risk: dict[str, Any],
    nonlinear_config: dict[str, Any],
) -> np.ndarray:
    transformed = transformed_true_risk_predictors(applicants, loan_options, baseline_true_risk)
    linear = np.zeros(len(transformed), dtype=float)
    for predictor in TRUE_RISK_PREDICTORS:
        linear += transformed[predictor].to_numpy(dtype=float) * float(
            baseline_true_risk["predictors"][predictor]["coefficient"]
        )
    return linear + nonlinear_score_components(transformed, nonlinear_config).sum(axis=1).to_numpy()


def nonlinear_repayment_probabilities(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    baseline_true_risk: dict[str, Any],
    nonlinear_config: dict[str, Any],
    *,
    intercept: float | None = None,
) -> pd.DataFrame:
    """Calculate nonlinear truth using only the baseline six observables."""

    alpha = nonlinear_config["calibration"]["solved_intercept"] if intercept is None else intercept
    if alpha is None:
        raise ValueError("nonlinear intercept must be calibrated before truth generation")
    score = float(alpha) + nonlinear_score_without_intercept(
        applicants, loan_options, baseline_true_risk, nonlinear_config
    )
    rho = np.asarray(sigmoid(score), dtype=float)
    term = loan_options["term_periods"].to_numpy(dtype=int)
    return pd.DataFrame(
        {
            "applicant_id": applicants["applicant_id"].to_numpy(),
            "repayment_probability_per_period_true": rho,
            "full_repayment_probability_true": survival_probability(rho, term),
        }
    ).loc[:, SIMULATION_TRUTH_COLUMNS]


def calibrate_nonlinear_intercept(
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
    baseline_true_risk: dict[str, Any],
    nonlinear_config: dict[str, Any],
) -> float:
    """Solve only alpha to match the declared mean full-repayment target."""

    score = nonlinear_score_without_intercept(
        applicants, loan_options, baseline_true_risk, nonlinear_config
    )
    term = loan_options["term_periods"].to_numpy(dtype=int)
    target = float(nonlinear_config["calibration"]["target_mean_full_repayment_probability"])

    def objective(alpha: float) -> float:
        return float(np.mean(np.power(sigmoid(alpha + score), term)) - target)

    return float(brentq(objective, -5.0, 15.0, xtol=float(nonlinear_config["calibration"]["tolerance"])))
