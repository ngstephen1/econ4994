"""Probability-scale adjusted contrasts and synthetic direct-effect truth."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fair_lending.simulation.approval import counterfactual_direct_probabilities


def standardized_black_white_contrast(
    fitted_result: Any, design_matrix: pd.DataFrame
) -> float:
    """Average p(Black) - p(White), holding every non-race column fixed."""
    if "black" not in design_matrix:
        raise ValueError("Design matrix must contain a black indicator")
    white_matrix = design_matrix.copy()
    black_matrix = design_matrix.copy()
    white_matrix["black"] = 0.0
    black_matrix["black"] = 1.0
    white_probability = np.asarray(fitted_result.predict(white_matrix), dtype=float)
    black_probability = np.asarray(fitted_result.predict(black_matrix), dtype=float)
    return float(np.mean(black_probability - white_probability))


def true_direct_effect(
    regression_sample: pd.DataFrame,
    config: dict[str, Any],
    intercept: float,
) -> dict[str, float]:
    """Return configured log-odds truth and its fixed-feature probability scale."""
    race_effects = config["scenario_effects"]["direct"]["log_odds_by_race"]
    configured = float(race_effects["Black"] - race_effects["White"])
    probabilities = counterfactual_direct_probabilities(
        regression_sample,
        config,
        intercept,
        reference_category="White",
        comparison_category="Black",
    )
    probability_gap = float(
        np.mean(
            np.asarray(probabilities["comparison_probability"])
            - np.asarray(probabilities["reference_probability"])
        )
    )
    return {
        "true_direct_log_odds": configured,
        "true_direct_probability_gap": probability_gap,
        "true_direct_probability_gap_percentage_points": 100.0 * probability_gap,
    }
