"""Transparent logistic approval mechanism and direct-effect checks."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from fair_lending.simulation.config import approval_transform_parameters


def _mapped_coefficients(values: pd.Series, mapping: dict[str, float]) -> np.ndarray:
    mapped = values.astype(object).map(mapping)
    if mapped.isna().any():
        unknown = values[mapped.isna()].astype(str).unique().tolist()
        raise ValueError(f"Approval coefficient missing for categories: {unknown}")
    return mapped.to_numpy(dtype=float)


def baseline_linear_predictor(
    applications: pd.DataFrame, config: dict[str, Any]
) -> np.ndarray:
    """Compute the documented underwriting score without intercept or race."""
    terms = config["approval_model"]["continuous_terms"]
    score = np.zeros(len(applications), dtype=float)
    for field, term in terms.items():
        values = applications[field].to_numpy(dtype=float)
        parameters = approval_transform_parameters(config, field)
        if parameters["kind"] == "linear_centered":
            transformed = (values - float(parameters["center"])) / float(
                parameters["scale"]
            )
        elif parameters["kind"] == "log_centered":
            transformed = (
                np.log(values) - np.log(float(parameters["center"]))
            ) / float(parameters["scale"])
        elif parameters["kind"] == "log_offset_centered":
            transformed = (
                np.log(values + float(parameters["offset"]))
                - np.log(float(parameters["center"]))
            ) / float(parameters["scale"])
        else:  # pragma: no cover - the config parser rejects this first
            raise ValueError(f"Unsupported transform kind for {field}")
        score += float(term["coefficient"]) * transformed
    for field in ("loan_purpose", "loan_type", "occupancy_type"):
        score += _mapped_coefficients(
            applications[field],
            config["approval_model"]["categorical_terms"][field]["coefficients"],
        )
    return score


def direct_effect_vector(
    race: pd.Series, config: dict[str, Any]
) -> np.ndarray:
    """Return the configured direct race log-odds term for each row."""
    return _mapped_coefficients(
        race, config["scenario_effects"]["direct"]["log_odds_by_race"]
    )


def approval_probabilities(
    applications: pd.DataFrame,
    config: dict[str, Any],
    intercept: float,
) -> np.ndarray:
    """Compute true approval probabilities from baseline and direct scores."""
    score = (
        float(intercept)
        + baseline_linear_predictor(applications, config)
        + direct_effect_vector(applications["race"], config)
    )
    return expit(score)


def counterfactual_direct_probabilities(
    applications: pd.DataFrame,
    config: dict[str, Any],
    intercept: float,
    *,
    reference_category: str = "White",
    comparison_category: str = "Black",
) -> dict[str, np.ndarray | float]:
    """Apply two race terms to identical non-race features for validation."""
    effects = config["scenario_effects"]["direct"]["log_odds_by_race"]
    reference_effect = float(effects[reference_category])
    comparison_effect = float(effects[comparison_category])
    common_score = float(intercept) + baseline_linear_predictor(applications, config)
    return {
        "reference_probability": expit(common_score + reference_effect),
        "comparison_probability": expit(common_score + comparison_effect),
        "configured_log_odds_difference": comparison_effect - reference_effect,
        "score_difference": comparison_effect - reference_effect,
    }
