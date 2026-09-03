"""Dashboard adapter around the validated statsmodels recovery analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fair_lending.analysis.estimands import standardized_black_white_contrast
from fair_lending.analysis.logit import fit_logit_sequence


def run_statistical_analysis(
    data: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Fit all predeclared models and return compact inference results."""
    rows = []
    for fitted in fit_logit_sequence(data, config):
        summary = dict(fitted["summary"])
        summary["adjusted_probability_gap"] = standardized_black_white_contrast(
            fitted["result"], fitted["design_matrix"]
        )
        rows.append(summary)
    return pd.DataFrame(rows)


def statistical_interpretation(results: pd.DataFrame) -> str:
    """Give cautious, result-dependent interpretation of the coefficient path."""
    indexed = results.set_index("model")
    raw = float(indexed.loc["model_0", "adjusted_probability_gap"])
    adjusted = float(indexed.loc["model_2", "adjusted_probability_gap"])
    if abs(adjusted) < 0.01 and abs(raw) >= 0.01:
        return (
            "The unadjusted association becomes small after measured borrower and "
            "loan characteristics are included. In this synthetic sample, those "
            "variables account for much of the conditional gap; this is not a causal "
            "claim about real lending."
        )
    if adjusted < -0.01:
        return (
            "A negative conditional Black–White difference remains after accounting "
            "for the included borrower and loan variables. Its interpretation depends "
            "on the configured synthetic mechanism and the model specification."
        )
    if adjusted > 0.01:
        return (
            "A positive conditional Black–White difference remains after accounting "
            "for the included borrower and loan variables. Finite-sample variation and "
            "the configured mechanism should be checked before interpreting it."
        )
    return (
        "Both the unadjusted and adjusted Black–White contrasts are small in this "
        "sample. Finite samples can still show nonzero estimates when no direct effect "
        "is configured."
    )
