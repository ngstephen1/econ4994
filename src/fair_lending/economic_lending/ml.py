"""Flexible repayment-risk benchmark with an identical observable information set."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from fair_lending.economic_lending.config import PROJECT_ROOT, config_fingerprint
from fair_lending.economic_lending.history import FORBIDDEN_LENDER_FIELDS


ML_CONFIG_PATH = PROJECT_ROOT / "configs" / "economic_lending" / "ml_benchmark.yaml"
ML_FEATURES = (
    "annual_income",
    "credit_score",
    "employment_years",
    "liquid_assets",
    "first_period_dti",
    "requested_ltv",
)
ML_POLICY_ID = "ml_histgb_v1"


@dataclass(frozen=True)
class MLSelectionResult:
    """Selected fitted estimator and every validation trial."""

    estimator: HistGradientBoostingClassifier
    selected_parameters: dict[str, Any]
    trials: pd.DataFrame
    config: dict[str, Any]


def load_ml_config(path: Path | str = ML_CONFIG_PATH) -> dict[str, Any]:
    """Load the separate benchmark configuration without changing the true DGP."""

    config_path = Path(path)
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("ML benchmark config must be a YAML mapping")
    config = copy.deepcopy(loaded)
    if tuple(config["features"]) != ML_FEATURES:
        raise ValueError("ML feature config must exactly match the declared allowlist")
    if config["model_family"] != "HistGradientBoostingClassifier":
        raise ValueError("Prompt 12 permits only HistGradientBoostingClassifier")
    config["metadata"] = {
        "config_path": str(config_path),
        "config_fingerprint": config_fingerprint(loaded),
    }
    return config


def ml_feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Select exactly the six substantive raw numeric features."""

    leaked = FORBIDDEN_LENDER_FIELDS.intersection(frame.columns)
    if leaked:
        raise ValueError(f"hidden truth leaked into ML input: {sorted(leaked)}")
    missing = [feature for feature in ML_FEATURES if feature not in frame.columns]
    if missing:
        raise ValueError(f"ML input is missing features: {missing}")
    design = frame.loc[:, ML_FEATURES].astype(float).copy()
    forbidden = {
        "group",
        "applicant_id",
        "cohort",
        "period_index",
        "paid_this_period",
        "default_period",
    }
    if forbidden.intersection(design.columns):
        raise AssertionError("non-feature information leaked into ML design")
    return design


def _candidate_parameters(config: dict[str, Any]) -> list[dict[str, Any]]:
    grid = config["grid"]
    names = list(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(grid[name] for name in names))
    ]


def tune_hist_gradient_boosting(
    training_history: pd.DataFrame,
    validation_history: pd.DataFrame,
    config: dict[str, Any],
) -> MLSelectionResult:
    """Fit on historical_train and select only by historical_validation."""

    if set(training_history["cohort"].unique()) != {"historical_train"}:
        raise ValueError("ML fitting accepts historical_train rows only")
    if set(validation_history["cohort"].unique()) != {"historical_validation"}:
        raise ValueError("ML selection accepts historical_validation rows only")
    x_train = ml_feature_matrix(training_history)
    y_train = training_history["paid_this_period"].to_numpy(dtype=int)
    x_validation = ml_feature_matrix(validation_history)
    y_validation = validation_history["paid_this_period"].to_numpy(dtype=int)

    fitted_candidates: list[HistGradientBoostingClassifier] = []
    rows: list[dict[str, Any]] = []
    fixed = config["fixed_parameters"]
    for candidate_index, parameters in enumerate(_candidate_parameters(config)):
        estimator = HistGradientBoostingClassifier(
            **fixed,
            **parameters,
            random_state=int(config["random_state"]),
        )
        estimator.fit(x_train, y_train)
        probability = estimator.predict_proba(x_validation)[:, 1]
        rows.append(
            {
                "candidate_index": candidate_index,
                **parameters,
                "validation_log_loss": float(
                    log_loss(y_validation, probability, labels=[0, 1])
                ),
                "validation_brier_score": float(
                    brier_score_loss(y_validation, probability)
                ),
                "validation_roc_auc": float(roc_auc_score(y_validation, probability)),
                "validation_average_precision": float(
                    average_precision_score(y_validation, probability)
                ),
            }
        )
        fitted_candidates.append(estimator)
    trials = pd.DataFrame(rows).sort_values(
        ["validation_log_loss", "validation_brier_score", "validation_roc_auc"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    trials["selected"] = False
    trials.loc[0, "selected"] = True
    selected_index = int(trials.loc[0, "candidate_index"])
    selected_parameters = {
        name: trials.loc[0, name] for name in config["grid"]
    }
    return MLSelectionResult(
        estimator=fitted_candidates[selected_index],
        selected_parameters=selected_parameters,
        trials=trials,
        config=config,
    )


def predict_ml_applicant_risk(
    estimator: HistGradientBoostingClassifier,
    applicants: pd.DataFrame,
    loan_options: pd.DataFrame,
) -> pd.DataFrame:
    """Produce one constant conditional risk estimate per applicant/request."""

    source = applicants.merge(
        loan_options.loc[:, ["applicant_id", "first_period_dti", "requested_ltv"]],
        on="applicant_id",
        validate="one_to_one",
    )
    probability = estimator.predict_proba(ml_feature_matrix(source))[:, 1]
    return pd.DataFrame(
        {
            "applicant_id": source["applicant_id"].to_numpy(),
            "repayment_probability_ml": probability,
        }
    )
