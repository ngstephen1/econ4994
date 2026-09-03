"""Predeclared feature regimes for the synthetic ML benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


RACE_BLIND = "race_blind"
RACE_AWARE = "race_aware_sensitivity"

NUMERIC_FEATURES = (
    "annual_income",
    "credit_score",
    "employment_years",
    "liquid_assets",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
)
CATEGORICAL_FEATURES = (
    "loan_purpose",
    "loan_type",
    "occupancy_type",
)
PROTECTED_ATTRIBUTES = {"race", "ethnicity", "sex", "age_group"}
PROXY_FIELDS = {"neighborhood_minority_share"}
OUTCOME_AND_TRUTH_FIELDS = {"approved", "approval_probability_true", "denial_reason"}
IDENTIFIER_FIELDS = {"application_id"}
FORBIDDEN_RACE_BLIND_FIELDS = (
    PROTECTED_ATTRIBUTES | PROXY_FIELDS | OUTCOME_AND_TRUTH_FIELDS | IDENTIFIER_FIELDS
)


@dataclass(frozen=True)
class FeatureSpec:
    """Simple description of one experimental predictor regime."""

    name: str
    label: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


FEATURE_REGIMES = {
    RACE_BLIND: FeatureSpec(
        name=RACE_BLIND,
        label="Race-blind primary model",
        numeric=NUMERIC_FEATURES,
        categorical=CATEGORICAL_FEATURES,
    ),
    RACE_AWARE: FeatureSpec(
        name=RACE_AWARE,
        label="Race-aware sensitivity model",
        numeric=NUMERIC_FEATURES,
        categorical=CATEGORICAL_FEATURES + ("race",),
    ),
}


def get_feature_spec(regime: str) -> FeatureSpec:
    """Return a validated feature specification."""
    try:
        return FEATURE_REGIMES[regime]
    except KeyError as error:
        raise ValueError(
            f"Unknown feature regime {regime!r}; expected {tuple(FEATURE_REGIMES)}"
        ) from error


def select_features(data: pd.DataFrame, regime: str) -> pd.DataFrame:
    """Select declared predictors and reject missing or prohibited inputs."""
    spec = get_feature_spec(regime)
    missing = set(spec.columns).difference(data.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    if regime == RACE_BLIND:
        leaked = FORBIDDEN_RACE_BLIND_FIELDS.intersection(spec.columns)
        if leaked:
            raise ValueError(f"Forbidden race-blind features: {sorted(leaked)}")
    predictors = data.loc[:, list(spec.columns)].copy()
    if predictors.isna().any().any():
        raise ValueError(
            "Version-1 benchmark features contain missing values; no imputer is "
            "included because missingness is not part of this DGP"
        )
    return predictors
