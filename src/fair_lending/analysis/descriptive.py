"""Descriptive application outcomes and unadjusted race disparities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm


def descriptive_outcomes(
    data: pd.DataFrame, config: dict[str, Any], scenario: str
) -> pd.DataFrame:
    """Summarize outcomes overall and for every configured race category."""
    rows: list[dict[str, Any]] = []

    def summarize(group: str, subset: pd.DataFrame, row_type: str) -> None:
        n = int(len(subset))
        approvals = int(subset["approved"].sum())
        rows.append(
            {
                "scenario": scenario,
                "row_type": row_type,
                "race": group,
                "n_applications": n,
                "sample_share": float(n / len(data)),
                "approval_count": approvals,
                "denial_count": n - approvals,
                "approval_rate": float(approvals / n),
                "denial_rate": float(1.0 - approvals / n),
                "mean_approval_probability_true": float(
                    subset["approval_probability_true"].mean()
                ),
            }
        )

    summarize("All races", data, "overall")
    for race in config["population"]["demographics"]["race"]["shares"]:
        subset = data.loc[data["race"].astype(object) == race]
        summarize(race, subset, "race")
    return pd.DataFrame(rows)


def independent_proportion_difference(
    successes_group: int,
    n_group: int,
    successes_reference: int,
    n_reference: int,
    *,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Wald CI for a difference between two independent proportions.

    The standard error is unpooled because the estimand is the difference in
    the two population proportions, not a null-restricted hypothesis test.
    """
    if n_group <= 0 or n_reference <= 0:
        raise ValueError("Both groups must contain observations")
    group_rate = successes_group / n_group
    reference_rate = successes_reference / n_reference
    difference = group_rate - reference_rate
    standard_error = np.sqrt(
        group_rate * (1.0 - group_rate) / n_group
        + reference_rate * (1.0 - reference_rate) / n_reference
    )
    critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
    return {
        "group_rate": float(group_rate),
        "reference_rate": float(reference_rate),
        "difference": float(difference),
        "standard_error": float(standard_error),
        "ci_low": float(difference - critical_value * standard_error),
        "ci_high": float(difference + critical_value * standard_error),
        "confidence_level": float(confidence_level),
    }


def black_white_raw_gap(data: pd.DataFrame, scenario: str) -> dict[str, Any]:
    """Return the unadjusted Black-minus-White approval disparity and CI."""
    race = data["race"].astype(object)
    black = data.loc[race == "Black", "approved"]
    white = data.loc[race == "White", "approved"]
    approval = independent_proportion_difference(
        int(black.sum()), len(black), int(white.sum()), len(white)
    )
    return {
        "scenario": scenario,
        "reference_group": "White",
        "comparison_group": "Black",
        "n_white": int(len(white)),
        "n_black": int(len(black)),
        "white_approval_rate": approval["reference_rate"],
        "black_approval_rate": approval["group_rate"],
        "raw_approval_gap": approval["difference"],
        "raw_approval_gap_percentage_points": 100.0 * approval["difference"],
        "standard_error": approval["standard_error"],
        "ci_low": approval["ci_low"],
        "ci_high": approval["ci_high"],
        "ci_low_percentage_points": 100.0 * approval["ci_low"],
        "ci_high_percentage_points": 100.0 * approval["ci_high"],
        "black_denial_rate": 1.0 - approval["group_rate"],
        "white_denial_rate": 1.0 - approval["reference_rate"],
        "raw_denial_gap": -approval["difference"],
        "raw_denial_gap_percentage_points": -100.0 * approval["difference"],
        "denial_ci_low": -approval["ci_high"],
        "denial_ci_high": -approval["ci_low"],
        "ci_method": "unpooled Wald difference in independent proportions",
    }
